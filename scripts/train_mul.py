import os

# ===== Thread env (same as original) =====
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)          # intra-op threads
torch.set_num_interop_threads(1)  # how many threads coordinate parallel regions

import shutil
import tempfile

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.nn.functional as F
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn import L1Loss
from tqdm import tqdm
import copy

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandSpatialCropd, RandFlipd, EnsureTyped, CropForegroundd, SpatialPadd
)
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader

from eval_utils import (
    psnr,
    ssim_3d,
    fid_from_features,
    plot_recon_loss,
    plot_adversarial_loss,
    plot_reconstructions,
    sample_ldm,
    sample_ldm_cond,
    plot_unet_loss,
)

from glob import glob
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import nibabel as nib
import numpy as np
import random
import argparse
from datetime import datetime
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Avoid shared-memory exhaustion in DataLoader workers (e.g., limited /dev/shm)
mp.set_sharing_strategy("file_system")


# ======================
# DDP helper functions
# ======================
def init_distributed():
    """
    Initialize distributed env if launched with torchrun.
    Returns (rank, world_size, local_rank).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"[DDP] Initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        print("[DDP] Single-process (non-distributed) mode.")
    return rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


# ======================
# Data loading
# ======================
def make_dataloaders_from_csv(
    csv_path,
    conditions=['age', 'sex', 'vol', 'part'],
    train_transforms=None,
    train_val_split=0.1,
    batch_size=1,
    num_workers=8,
    seed=1017,
    distributed=False,
    rank=0,
    world_size=1,
):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        sample = {"image": row["image"]}
        for c in conditions:
            sample[c] = row[c]
        data.append(sample)

    random.seed(seed)
    random.shuffle(data)
    random.seed()

    split = int(len(data) * train_val_split)
    train_data, val_data = data[:-split], data[-split:]

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=train_transforms)

    if is_main_process(rank):
        print("Example dataset sample: ", train_ds[0])
        print(f'Transformed data shape: {train_ds[0]["image"].shape}')
        print(f"Number of training samples: {len(train_ds)}")
        print(f"Number of validation samples: {len(val_ds)}")

    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def check_train_data(train_loader, idx=0, title="Train data example", save_path=None):
    # Plot axial, coronal and sagittal slices of a training sample
    check_data = first(train_loader)
    img = check_data["image"][idx, 0]
    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close("all")


# ======================
# Checkpoint utils
# ======================
def _load_ckpt_into_ae(ae, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    missing, unexpected = ae.load_state_dict(state, strict=False)
    print(f"[AE] load ckpt: missing={len(missing)}, unexpected={len(unexpected)}")


def _save_ckpt(
    path: str,
    unet: torch.nn.Module,
    opt,
    scaler,
    epoch: int,
    global_step: int,
    ema: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": unet.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    if ema is not None:
        payload["ema"] = {k: v.cpu() for k, v in ema.shadow.items()}
    torch.save(payload, path)


def _load_ckpt_into_unet(unet: torch.nn.Module, ckpt_path: str) -> Tuple[int, int]:
    """Loads either a raw state_dict or a packaged checkpoint. Returns (epoch, global_step) if present."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        missing, unexpected = unet.load_state_dict(sd["state_dict"], strict=False)
        print(f"[LDM] resumed UNet (packaged): missing={len(missing)} unexpected={len(unexpected)}")
        return int(sd.get("epoch", 0)), 0
    else:
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        print(f"[LDM] resumed UNet (raw): missing={len(missing)} unexpected={len(unexpected)}")
        return 0, 0


# ======================
# Losses
# ======================
def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=[1, 2, 3, 4],
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


# ======================
# AE training (DDP-aware)
# ======================
def train_ae(
    autoencoder,
    train_loader,
    val_loader=None,
    val_interval=1,
    ae_epochs=100,
    adv_weight=0.01,
    perceptual_weight=0.001,
    kl_weight=1e-6,
    lr=1e-4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    outdir="ckpts",
    rank=0,
):
    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
    )
    discriminator.to(device)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(
        spatial_dims=3,
        network_type="squeeze",
        is_fake_3d=True,
        fake_3d_ratio=0.2,
    )
    loss_perceptual.to(device)

    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), lr=lr)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=lr)

    n_epochs = ae_epochs
    autoencoder_warm_up_n_epochs = int(0.2 * n_epochs)  # 20% warmup
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_ssim_train_list, epoch_psnr_train_list = [], []
    val_recon_epoch_loss_list = []
    epoch_ssim_val_list, epoch_psnr_val_list = [], []

    ae_best = None
    ae_best_loss = float("inf")

    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0.0
        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0
        ssim_train, psnr_train = 0.0, 0.0
        ssim_val, psnr_val = 0.0, 0.0

        # DDP: make sampler deterministic per-epoch
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            ncols=110,
            disable=not is_main_process(rank),
        )
        if is_main_process(rank):
            progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            images = batch["image"].to(device)

            # Generator (AE) part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            kl_loss_val = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss_val + perceptual_weight * p_loss
            psnr_train += psnr(reconstruction.float(), images.float())
            ssim_train += ssim_3d(reconstruction.float(), images.float())

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(
                    logits_fake,
                    target_is_real=True,
                    for_discriminator=False,
                )
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(
                    logits_fake,
                    target_is_real=False,
                    for_discriminator=True,
                )
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(
                    logits_real,
                    target_is_real=True,
                    for_discriminator=True,
                )
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss
                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            if is_main_process(rank):
                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                        "SSIM_train": ssim_train / (step + 1),
                        "PSNR_train": psnr_train / (step + 1),
                    }
                )

        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
        epoch_ssim_train_list.append(np.mean(ssim_train / (step + 1)))
        epoch_psnr_train_list.append(np.mean(psnr_train / (step + 1)))

        if is_main_process(rank):
            plot_recon_loss(
                epoch_recon_loss_list,
                epoch_ssim_train_list,
                epoch_psnr_train_list,
                title=f"Train Reconstruction Loss Curve_ep{epoch + 1}",
                outdir=outdir,
                filename="AE_train_recon_loss.png",
            )
            plot_adversarial_loss(
                epoch_gen_loss_list,
                epoch_disc_loss_list,
                title=f"Adversarial Training Curves_ep{epoch + 1}",
                outdir=outdir,
                filename="AE_disc_loss.png",
            )
            plot_reconstructions(
                batch,
                reconstruction,
                idx=0,
                channel=0,
                title=f"Train Sample Reconstructions_ep{epoch + 1}",
                outdir=outdir,
                filename="AE_train_recons.png",
            )

        # ========= Validation =========
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            if isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            val_epoch_loss = 0.0
            val_progress_bar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                ncols=70,
                disable=not is_main_process(rank),
            )
            if is_main_process(rank):
                val_progress_bar.set_description(f"Val Epoch {epoch}")
            with torch.no_grad():
                for val_step, val_batch in val_progress_bar:
                    val_images = val_batch["image"].to(device)
                    val_reconstruction, _, _ = autoencoder(val_images)
                    val_recons_loss = l1_loss(
                        val_reconstruction.float(),
                        val_images.float(),
                    )
                    val_epoch_loss += val_recons_loss.item()
                    psnr_val += psnr(val_reconstruction.float(), val_images.float())
                    ssim_val += ssim_3d(val_reconstruction.float(), val_images.float())
                    if is_main_process(rank):
                        val_progress_bar.set_postfix(
                            {
                                "val_recons_loss": val_epoch_loss / (val_step + 1),
                                "SSIM_val": ssim_val / (val_step + 1),
                                "PSNR_val": psnr_val / (val_step + 1),
                            }
                        )
            val_recon_epoch_loss_list.append(val_epoch_loss / (val_step + 1))
            epoch_ssim_val_list.append(ssim_val / (val_step + 1))
            epoch_psnr_val_list.append(np.mean(psnr_val / (val_step + 1)))

            if is_main_process(rank):
                plot_recon_loss(
                    val_recon_epoch_loss_list,
                    epoch_ssim_val_list,
                    epoch_psnr_val_list,
                    title=f"Val Reconstruction Loss Curve_ep{epoch + 1}",
                    outdir=outdir,
                    filename="AE_val_recon_loss.png",
                )

                # best ckpt
                if val_epoch_loss / (val_step + 1) < ae_best_loss:
                    print("Updating best AE checkpoint...")
                    ae_best_loss = val_epoch_loss / (val_step + 1)
                    # copy underlying module if DDP
                    if isinstance(autoencoder, DDP):
                        ae_best = copy.deepcopy(autoencoder.module)
                    else:
                        ae_best = copy.deepcopy(autoencoder)

                    _save_ckpt(
                        os.path.join(outdir, "AE_best.pt"),
                        autoencoder,
                        opt=optimizer_g,
                        scaler=None,
                        epoch=epoch + 1,
                        global_step=None,
                        extra={
                            "opt_d": optimizer_d,
                            "train_rec_loss": epoch_recon_loss_list,
                            "train_disc_loss": epoch_disc_loss_list,
                            "train_gen_loss": epoch_gen_loss_list,
                            "train_ssim": epoch_ssim_train_list,
                            "train_psnr": epoch_psnr_train_list,
                            "val_rec_loss": val_recon_epoch_loss_list,
                            "val_ssim": epoch_ssim_val_list,
                            "val_psnr": epoch_psnr_val_list,
                        },
                    )
                    plot_reconstructions(
                        val_batch,
                        val_reconstruction,
                        idx=0,
                        channel=0,
                        title=f"Val Sample Reconstructions_ep{epoch + 1}",
                        outdir=outdir,
                        filename="AE_val_recons.png",
                    )

                if epoch + 1 == n_epochs:
                    print(
                        "Final epoch, saving last AE checkpoint and val reconstructions..."
                    )
                    plot_reconstructions(
                        val_batch,
                        val_reconstruction,
                        idx=0,
                        channel=0,
                        title=f"Val Sample Reconstructions_ep{epoch + 1}",
                        outdir=outdir,
                        filename="AE_last_val_recons.png",
                    )

        # always save last
        if is_main_process(rank):
            _save_ckpt(
                os.path.join(outdir, "AE_last.pt"),
                autoencoder,
                opt=optimizer_g,
                scaler=None,
                epoch=epoch + 1,
                global_step=None,
                extra={
                    "opt_d": optimizer_d,
                    "train_rec_loss": epoch_recon_loss_list,
                    "train_disc_loss": epoch_disc_loss_list,
                    "train_gen_loss": epoch_gen_loss_list,
                    "train_ssim": epoch_ssim_train_list,
                    "train_psnr": epoch_psnr_train_list,
                    "val_rec_loss": val_recon_epoch_loss_list,
                    "val_ssim": epoch_ssim_val_list,
                    "val_psnr": epoch_psnr_val_list,
                },
            )

    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()
    if is_main_process(rank):
        print("Returning the last and best AEs.")
    return autoencoder, ae_best


# ======================
# LDM training (DDP-aware)
# ======================
def train_ldm(
    unet,
    train_loader,
    autoencoder,
    ldm_epochs=150,
    ldm_num_parts=5,
    lr=1e-4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    outdir="ckpts",
    sample_every=25,
    rank=0,
):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )

    autoencoder.to(device)
    unet.to(device)

    # Compute scale_factor
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            check_data = next(iter(train_loader))
            check_img = check_data["image"].to(device)
            z_check = autoencoder.encode_stage_2_inputs(check_img)

    scale_factor = 1.0 / torch.std(z_check)
    if is_main_process(rank):
        print(f"Scaling factor set to {scale_factor.item():.4f}")

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    optimizer_diff = torch.optim.Adam(unet.parameters(), lr=lr)
    scaler = GradScaler()
    epoch_loss_list = []

    # sanity check on channels
    latent_channels = z_check.shape[1]
    expected_in_channels = latent_channels + ldm_num_parts
    if unet.in_channels != expected_in_channels:
        raise ValueError(
            f"UNet in_channels={unet.in_channels}, "
            f"but latent_channels({latent_channels}) + ldm_num_parts({ldm_num_parts}) = {expected_in_channels}. "
            "You must construct the UNet with in_channels = latent_channels + ldm_num_parts."
        )

    for epoch in range(ldm_epochs):
        unet.train()
        epoch_loss = 0.0

        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            ncols=70,
            disable=not is_main_process(rank),
        )
        if is_main_process(rank):
            progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            images = batch["image"].to(device)
            part_id = batch["part"].to(device)

            B = images.shape[0]

            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                with torch.no_grad():
                    latents = autoencoder.encode_stage_2_inputs(images)
                    latents = latents * scale_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.num_train_timesteps,
                    (B,),
                    device=device,
                ).long()

                noisy_latents = scheduler.add_noise(
                    original_samples=latents,
                    noise=noise,
                    timesteps=timesteps,
                )

                _, _, Dz, Hz, Wz = noisy_latents.shape
                part_onehot = F.one_hot(part_id, num_classes=ldm_num_parts).float()
                part_map = part_onehot.view(B, ldm_num_parts, 1, 1, 1)
                part_map = part_map.expand(-1, -1, Dz, Hz, Wz)

                noisy_latents_cond = torch.cat([noisy_latents, part_map], dim=1)
                noise_pred = unet(noisy_latents_cond, timesteps)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()
            if is_main_process(rank):
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % sample_every == 0 and is_main_process(rank):
            for i in range(ldm_num_parts):
                sample_ldm_cond(
                    train_loader,
                    autoencoder,
                    unet,
                    scheduler,
                    device,
                    idx=0,
                    channel=0,
                    outdir=outdir,
                    filename=f"synthetic_ep{epoch + 1}_part{i}",
                    part_id=i,
                )

        if is_main_process(rank):
            _save_ckpt(
                os.path.join(outdir, "UNET_last.pt"),
                unet,
                opt=optimizer_diff,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=None,
                extra={"train_loss": epoch_loss_list},
            )
            plot_unet_loss(
                epoch_loss_list,
                title=f"UNET Loss Curves_ep{epoch + 1}",
                outdir=outdir,
                filename="UNET_loss.png",
            )


# ======================
# Main
# ======================
def main():
    ap = argparse.ArgumentParser(
        description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with columns: image[/path], sex, age, vol, [target_label]",
    )
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train_val_split", type=float, default=0.1)
    ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="both")

    # AE config
    ap.add_argument("--ae_epochs", type=int, default=50)
    ap.add_argument("--ae_lr", type=float, default=1e-4)
    ap.add_argument("--ae_latent_ch", type=int, default=3)
    ap.add_argument("--ae_kl", type=float, default=1e-6)
    ap.add_argument("--ae_adv_weight", type=float, default=0.01)
    ap.add_argument("--ae_perceptual_weight", type=float, default=0.001)
    ap.add_argument("--ae_kl_weight", type=float, default=1e-6)
    ap.add_argument("--ae_num_channels", default="64,128,256,512")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt (optional)")

    # LDM config
    ap.add_argument("--ldm_epochs", type=int, default=150)
    ap.add_argument("--ldm_lr", type=float, default=1e-4)
    ap.add_argument("--ldm_use_cond", default="False")
    ap.add_argument("--ldm_num_parts", type=int, default=5)
    ap.add_argument("--ldm_num_channels", default="128,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_ckpt", default="", help="Resume UNet weights (optional)")
    ap.add_argument(
        "--ldm_sample_every", type=int, default=25, help="Synthesize samples every N epochs"
    )

    ap.add_argument("--outdir", default="ckpts")
    ap.add_argument("--out_prefix", default="")

    args = ap.parse_args()

    # ==== DDP init ====
    rank, world_size, local_rank = init_distributed()

    # ==== Output dir (synchronized across ranks) ====
    if is_main_process(rank):
        experiment_dir = os.path.join(
            args.outdir,
            f"run_{args.out_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    else:
        experiment_dir = None

    if world_size > 1:
        obj_list = [experiment_dir]
        dist.broadcast_object_list(obj_list, src=0)
        experiment_dir = obj_list[0]

    if is_main_process(rank):
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"\n✅ Output dir: {experiment_dir}\n")
        cfg = vars(args).copy()
        for k, v in list(cfg.items()):
            if isinstance(v, tuple):
                cfg[k] = list(v)
        with open(
            os.path.join(experiment_dir, "args.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

    if world_size > 1:
        dist.barrier()

    args.outdir = experiment_dir

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size = tuple(int(x) for x in args.size.split(","))

    channel = 0  # 0 = Flair
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            transforms.EnsureTyped(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear")),
            transforms.CropForegroundd(keys="image", source_key="image"),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=size),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=size,
                mode="constant",
                constant_values=-1.0,
            ),
        ]
    )

    train_loader, val_loader = make_dataloaders_from_csv(
        args.csv,
        conditions=["age", "sex", "vol", "part"],
        train_transforms=train_transforms,
        train_val_split=args.train_val_split,
        batch_size=args.batch,
        num_workers=args.workers,
        seed=1017,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )

    ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
    ae_latent_ch = int(args.ae_latent_ch)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if is_main_process(rank):
        print(f"\n⚙️ Using {device} | world_size={world_size}, rank={rank}\n")

    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=ae_num_channels,
        latent_channels=ae_latent_ch,
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=(False, False, True, True),
    )
    autoencoder.to(device)

    if args.ae_ckpt != "":
        _load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)

    # wrap AE in DDP (allow unused parameters!)
    if world_size > 1:
        autoencoder = DDP(
            autoencoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # <-- important for AutoencoderKL
        )

    # ---- AE stage ----
    if args.stage in ["ae", "both"]:
        last_ae, best_ae = train_ae(
            autoencoder,
            train_loader,
            val_loader,
            val_interval=1,
            ae_epochs=args.ae_epochs,
            adv_weight=args.ae_adv_weight,
            perceptual_weight=args.ae_perceptual_weight,
            kl_weight=args.ae_kl_weight,
            lr=args.ae_lr,
            device=device,
            outdir=args.outdir,
            rank=rank,
        )

        # After AE training, load the best weights from disk on all ranks
        if world_size > 1:
            dist.barrier()
        best_ckpt_path = os.path.join(args.outdir, "AE_best.pt")
        target_ae = autoencoder.module if isinstance(autoencoder, DDP) else autoencoder
        if os.path.exists(best_ckpt_path):
            _load_ckpt_into_ae(target_ae, best_ckpt_path, device)
        if world_size > 1:
            dist.barrier()

    # ---- LDM (UNet) ----
    ldm_num_channels = tuple(int(x) for x in args.ldm_num_channels.split(","))
    ldm_num_head_channels = tuple(int(x) for x in args.ldm_num_head_channels.split(","))
    ldm_num_parts = int(args.ldm_num_parts)

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3 + ldm_num_parts,
        out_channels=3,
        num_res_blocks=2,
        num_channels=ldm_num_channels,
        attention_levels=(False, True, True),
        num_head_channels=ldm_num_head_channels,
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        upcast_attention=True,
        use_flash_attention=False,
    )
    unet.to(device)

    if args.ldm_ckpt != "":
        _load_ckpt_into_unet(unet, args.ldm_ckpt)

    if world_size > 1:
        unet = DDP(
            unet,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # safer
        )

    if args.stage in ["ldm", "both"]:
        train_ldm(
            unet,
            train_loader,
            autoencoder,
            ldm_epochs=args.ldm_epochs,
            ldm_num_parts=ldm_num_parts,
            lr=args.ldm_lr,
            device=device,
            outdir=args.outdir,
            sample_every=args.ldm_sample_every,
            rank=rank,
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
