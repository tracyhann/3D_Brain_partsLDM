"""
DDP trainer for train_3d_VAE.py (AE stage only).

Example (4 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_3d_VAE_ddp.py \
  --csv data/processed_parts/whole_brain_0206.csv \
  --spacing 1.5,1.5,1.5 \
  --batch 2 \
  --n_samples ALL \
  --workers 0 \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --outdir ckpts/AE \
  --out_prefix whole_brain_AE \
  --out_postfix 0214
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from monai import transforms
from monai.data import DataLoader, Dataset
from torch.nn import L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

# Import base single-GPU script utilities
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_3d_VAE as base  # noqa: E402


def parse_bool(val):
    if isinstance(val, bool):
        return val
    return str(val).lower() in {"1", "true", "yes", "y"}


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def broadcast_str(msg: str, src: int = 0) -> str:
    payload = [msg]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().mean().item())
    return float(x)


def build_transforms(keys, spacing):
    channel = 0
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys, track_meta=False),
            transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
            transforms.DivisiblePadd(keys=keys, k=32, mode="constant", constant_values=-1.0),
        ]
    )


def make_dataloaders_ddp(
    csv_path,
    conditions=None,
    train_transforms=None,
    n_samples="ALL",
    data_split_json_path="data/patient_splits_image_ids_75_10_15.json",
    batch_size=1,
    num_workers=8,
    seed=1017,
):
    if conditions is None:
        conditions = ["age", "sex", "vol", "group"]

    with open(data_split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    image_ids = {}
    for image_id in splits["train"]:
        image_ids[image_id] = "train"
    for image_id in splits["val"]:
        image_ids[image_id] = "val"
    for image_id in splits["test"]:
        image_ids[image_id] = "test"

    df = pd.read_csv(csv_path)
    train_data, val_data, test_data = [], [], []
    train_added = 0

    for _, row in df.iterrows():
        image_id = row["imageID"]
        if image_id not in image_ids:
            continue
        sample = {"image": row["image"]}
        for c in conditions:
            sample[c] = row[c]
        split = image_ids[image_id]
        if split == "train":
            train_data.append(sample)
            train_added += 1
            if n_samples != "ALL" and train_added >= n_samples:
                break
        elif split == "val":
            val_data.append(sample)
        elif split == "test":
            test_data.append(sample)

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=train_transforms)
    test_ds = Dataset(data=test_data, transform=train_transforms)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False, seed=seed)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False, seed=seed)
    test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


def train_ae_ddp(
    ddp_autoencoder,
    ddp_discriminator,
    train_loader,
    train_sampler,
    val_loader,
    val_sampler,
    val_interval=1,
    ae_epochs=100,
    adv_weight=0.01,
    perceptual_weight=0.001,
    kl_weight=1e-6,
    lr=1e-4,
    torch_autocast=True,
    device=None,
    outdir="ckpts",
    rank=0,
    resume_ae_ckpt="",
):
    if device is None:
        device = torch.device("cuda")

    l1_loss = L1Loss()
    adv_loss = base.PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = base.PerceptualLoss(
        spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
    ).to(device)

    optimizer_g = torch.optim.AdamW(ddp_autoencoder.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.AdamW(ddp_discriminator.parameters(), lr=lr*0.5, betas=(0.5, 0.9))
    scaler = torch.amp.GradScaler("cuda", enabled=torch_autocast)

    n_epochs = ae_epochs
    start_epoch = 0
    autoencoder_warm_up_n_epochs = int(0.2 * n_epochs)
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_ssim_train_list, epoch_psnr_train_list = [], []
    val_recon_epoch_loss_list = []
    val_recon_ema_list = []
    epoch_ssim_val_list, epoch_psnr_val_list = [], []

    ae_best_ema = float("inf")
    val_loss_ema = None
    ema_alpha = 0.2

    if resume_ae_ckpt:
        ckpt = torch.load(resume_ae_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = ddp_autoencoder.module.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(
                f"[Resume] Loaded AE from {resume_ae_ckpt}. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        opt_g_state = ckpt.get("optimizer")
        if isinstance(opt_g_state, dict):
            optimizer_g.load_state_dict(opt_g_state)
            if rank == 0:
                print("[Resume] Loaded generator optimizer state.")

        scaler_state = ckpt.get("scaler")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
            if rank == 0:
                print("[Resume] Loaded AMP scaler state.")

        extra = ckpt.get("extra") or {}

        opt_d_state = extra.get("opt_d")
        if hasattr(opt_d_state, "state_dict"):
            opt_d_state = opt_d_state.state_dict()
        if isinstance(opt_d_state, dict):
            optimizer_d.load_state_dict(opt_d_state)
            if rank == 0:
                print("[Resume] Loaded discriminator optimizer state.")

        disc_state = extra.get("disc_state_dict")
        if isinstance(disc_state, dict):
            disc_missing, disc_unexpected = ddp_discriminator.module.load_state_dict(disc_state, strict=False)
            if rank == 0:
                print(
                    "[Resume] Loaded discriminator weights. "
                    f"missing={len(disc_missing)} unexpected={len(disc_unexpected)}"
                )
        elif rank == 0:
            print("[Resume] No discriminator weights found in checkpoint; starting discriminator from scratch.")

        epoch_recon_loss_list = list(extra.get("train_rec_loss", []))
        epoch_disc_loss_list = list(extra.get("train_disc_loss", []))
        epoch_gen_loss_list = list(extra.get("train_gen_loss", []))
        epoch_ssim_train_list = list(extra.get("train_ssim", []))
        epoch_psnr_train_list = list(extra.get("train_psnr", []))
        val_recon_epoch_loss_list = list(extra.get("val_rec_loss", []))
        val_recon_ema_list = list(extra.get("val_rec_ema", []))
        epoch_ssim_val_list = list(extra.get("val_ssim", []))
        epoch_psnr_val_list = list(extra.get("val_psnr", []))

        start_epoch = int(ckpt.get("epoch") or 0)
        if start_epoch < 0:
            start_epoch = 0

        if val_recon_ema_list:
            val_recon_ema_list = [float(v) for v in val_recon_ema_list]
            val_loss_ema = float(val_recon_ema_list[-1])
            ae_best_ema = float(min(val_recon_ema_list))
        else:
            for val_loss in val_recon_epoch_loss_list:
                val_loss = float(val_loss)
                if val_loss_ema is None:
                    val_loss_ema = val_loss
                else:
                    val_loss_ema = ema_alpha * val_loss + (1.0 - ema_alpha) * val_loss_ema
                if val_loss_ema < ae_best_ema:
                    ae_best_ema = val_loss_ema

        if rank == 0:
            print(
                f"[Resume] start_epoch={start_epoch}, target_epochs={n_epochs}, "
                f"history(train_rec={len(epoch_recon_loss_list)}, val_rec={len(val_recon_epoch_loss_list)})"
            )

    if start_epoch >= n_epochs:
        if rank == 0:
            print(
                f"[Resume] start_epoch ({start_epoch}) >= ae_epochs ({n_epochs}). "
                "Nothing to train."
            )
        del loss_perceptual
        torch.cuda.empty_cache()
        return

    def _build_extra_payload():
        return {
            "opt_d": optimizer_d,
            "disc_state_dict": ddp_discriminator.module.state_dict(),
            "train_rec_loss": epoch_recon_loss_list,
            "train_disc_loss": epoch_disc_loss_list,
            "train_gen_loss": epoch_gen_loss_list,
            "train_ssim": epoch_ssim_train_list,
            "train_psnr": epoch_psnr_train_list,
            "val_rec_loss": val_recon_epoch_loss_list,
            "val_rec_ema": val_recon_ema_list,
            "val_ssim": epoch_ssim_val_list,
            "val_psnr": epoch_psnr_val_list,
        }
    for epoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(epoch)
        ddp_autoencoder.train()
        ddp_discriminator.train()

        recon_sum = 0.0
        gen_sum = 0.0
        disc_sum = 0.0
        ssim_sum = 0.0
        psnr_sum = 0.0
        num_steps = 0
        last_batch = None
        last_reconstruction = None

        progress = tqdm(
            total=len(train_loader),
            ncols=110,
            file=sys.__stdout__,
            disable=rank != 0,
        )
        progress.set_description(f"Epoch {epoch}")

        for batch in train_loader:
            images = batch["image"].to(device)
            last_batch = batch

            optimizer_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch_autocast):
                reconstruction, z_mu, z_sigma = ddp_autoencoder(images)
                kl_loss = base.KL_loss(z_mu, z_sigma)
                recons_loss = l1_loss(reconstruction.float(), images.float())
                if perceptual_weight > 0:
                    p_loss = loss_perceptual(reconstruction.float(), images.float())
                else:
                    p_loss = torch.zeros((), device=device, dtype=recons_loss.dtype)
                loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            last_reconstruction = reconstruction.detach()
            psnr_val = _to_float(base.psnr(reconstruction.float(), images.float()))
            ssim_val = _to_float(base.ssim_3d(reconstruction.float(), images.float()))

            generator_loss = torch.tensor(0.0, device=device)
            discriminator_loss = torch.tensor(0.0, device=device)

            if epoch > autoencoder_warm_up_n_epochs:
                for p in ddp_discriminator.module.parameters():
                    p.requires_grad_(False)
                with torch.amp.autocast("cuda", enabled=torch_autocast):
                    logits_fake = ddp_discriminator.module(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g = loss_g + adv_weight * generator_loss

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)

            if epoch > autoencoder_warm_up_n_epochs:
                for p in ddp_discriminator.module.parameters():
                    p.requires_grad_(True)
                optimizer_d.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=torch_autocast):
                    logits_fake = ddp_discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = ddp_discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)

            scaler.update()

            recon_sum += _to_float(recons_loss)
            gen_sum += _to_float(generator_loss)
            disc_sum += _to_float(discriminator_loss)
            ssim_sum += ssim_val
            psnr_sum += psnr_val
            num_steps += 1

            if rank == 0:
                progress.set_postfix(
                    {
                        "recons_loss": recon_sum / max(num_steps, 1),
                        "gen_loss": gen_sum / max(num_steps, 1),
                        "disc_loss": disc_sum / max(num_steps, 1),
                        "SSIM_train": ssim_sum / max(num_steps, 1),
                        "PSNR_train": psnr_sum / max(num_steps, 1),
                    }
                )
                progress.update(1)

        if rank == 0:
            progress.close()

        train_metrics = torch.tensor(
            [recon_sum, gen_sum, disc_sum, ssim_sum, psnr_sum, float(num_steps)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)

        global_steps = int(train_metrics[5].item())
        train_recon_mean = float(train_metrics[0].item() / max(global_steps, 1))
        train_gen_mean = float(train_metrics[1].item() / max(global_steps, 1))
        train_disc_mean = float(train_metrics[2].item() / max(global_steps, 1))
        train_ssim_mean = float(train_metrics[3].item() / max(global_steps, 1))
        train_psnr_mean = float(train_metrics[4].item() / max(global_steps, 1))

        if rank == 0:
            epoch_recon_loss_list.append(train_recon_mean)
            epoch_gen_loss_list.append(train_gen_mean)
            epoch_disc_loss_list.append(train_disc_mean)
            epoch_ssim_train_list.append(train_ssim_mean)
            epoch_psnr_train_list.append(train_psnr_mean)
            base.plot_recon_loss(
                epoch_recon_loss_list,
                epoch_ssim_train_list,
                epoch_psnr_train_list,
                title=f"Train Reconstruction Loss Curve_ep{epoch+1}",
                outdir=outdir,
                filename="AE_train_recon_loss.png",
            )
            base.plot_adversarial_loss(
                epoch_gen_loss_list,
                epoch_disc_loss_list,
                title=f"Adversarial Training Curves_ep{epoch+1}",
                outdir=outdir,
                filename="AE_disc_loss.png",
            )
            if last_batch is not None and last_reconstruction is not None:
                base.plot_reconstructions(
                    last_batch,
                    last_reconstruction,
                    idx=0,
                    channel=0,
                    title=f"Train Sample Reconstructions_ep{epoch+1}",
                    outdir=outdir,
                    filename="AE_train_recons.png",
                )

        val_epoch_loss = None
        val_batch_last = None
        val_reconstruction_last = None
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            val_sampler.set_epoch(epoch)
            ddp_autoencoder.eval()

            val_loss_sum = 0.0
            val_ssim_sum = 0.0
            val_psnr_sum = 0.0
            val_steps = 0

            val_progress = tqdm(
                total=len(val_loader),
                ncols=70,
                file=sys.__stdout__,
                disable=rank != 0,
            )
            val_progress.set_description(f"Val Epoch {epoch}")

            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device)
                    with torch.amp.autocast("cuda", enabled=torch_autocast):
                        val_reconstruction, _, _ = ddp_autoencoder(val_images)
                    val_batch_last = val_batch
                    val_reconstruction_last = val_reconstruction.detach()

                    val_recons_loss = l1_loss(val_reconstruction.float(), val_images.float())
                    val_psnr = _to_float(base.psnr(val_reconstruction.float(), val_images.float()))
                    val_ssim = _to_float(base.ssim_3d(val_reconstruction.float(), val_images.float()))

                    val_loss_sum += _to_float(val_recons_loss)
                    val_ssim_sum += val_ssim
                    val_psnr_sum += val_psnr
                    val_steps += 1

                    if rank == 0:
                        val_progress.set_postfix(
                            {
                                "val_recons_loss": val_loss_sum / max(val_steps, 1),
                                "SSIM_val": val_ssim_sum / max(val_steps, 1),
                                "PSNR_val": val_psnr_sum / max(val_steps, 1),
                            }
                        )
                        val_progress.update(1)

            if rank == 0:
                val_progress.close()

            val_metrics = torch.tensor(
                [val_loss_sum, val_ssim_sum, val_psnr_sum, float(val_steps)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
            global_val_steps = int(val_metrics[3].item())
            val_epoch_loss = float(val_metrics[0].item() / max(global_val_steps, 1))
            val_epoch_ssim = float(val_metrics[1].item() / max(global_val_steps, 1))
            val_epoch_psnr = float(val_metrics[2].item() / max(global_val_steps, 1))

            if rank == 0:
                val_recon_epoch_loss_list.append(val_epoch_loss)
                epoch_ssim_val_list.append(val_epoch_ssim)
                epoch_psnr_val_list.append(val_epoch_psnr)
                base.plot_recon_loss(
                    val_recon_epoch_loss_list,
                    epoch_ssim_val_list,
                    epoch_psnr_val_list,
                    title=f"Val Reconstruction Loss Curve_ep{epoch+1}",
                    outdir=outdir,
                    filename="AE_val_recon_loss.png",
                )

                if val_loss_ema is None:
                    val_loss_ema = val_epoch_loss
                else:
                    val_loss_ema = ema_alpha * val_epoch_loss + (1.0 - ema_alpha) * val_loss_ema
                val_recon_ema_list.append(float(val_loss_ema))

                print(
                    f"Val epoch {epoch}: raw={val_epoch_loss:.6f}, "
                    f"ema={val_loss_ema:.6f}, best_ema={ae_best_ema:.6f}"
                )

                if val_loss_ema < ae_best_ema:
                    ae_best_ema = val_loss_ema
                    print(
                        f"Updating best AE checkpoint (EMA improved): "
                        f"raw={val_epoch_loss:.6f}, ema={val_loss_ema:.6f}"
                    )
                    base._save_ckpt(
                        os.path.join(outdir, "AE_best.pt"),
                        ddp_autoencoder.module,
                        opt=optimizer_g,
                        scaler=None,
                        epoch=epoch + 1,
                        global_step=None,
                        extra=_build_extra_payload(),
                    )
                    if val_batch_last is not None and val_reconstruction_last is not None:
                        base.plot_reconstructions(
                            val_batch_last,
                            val_reconstruction_last,
                            idx=0,
                            channel=0,
                            title=f"Val Sample Reconstructions_ep{epoch+1}",
                            outdir=outdir,
                            filename="AE_val_recons.png",
                        )

                if epoch + 1 == n_epochs and val_batch_last is not None and val_reconstruction_last is not None:
                    base.plot_reconstructions(
                        val_batch_last,
                        val_reconstruction_last,
                        idx=0,
                        channel=0,
                        title=f"Val Sample Reconstructions_ep{epoch+1}",
                        outdir=outdir,
                        filename="AE_last_val_recons.png",
                    )

        if rank == 0:
            base._save_ckpt(
                os.path.join(outdir, "AE_last.pt"),
                ddp_autoencoder.module,
                opt=optimizer_g,
                scaler=None,
                epoch=epoch + 1,
                global_step=None,
                extra=_build_extra_payload(),
            )

    del loss_perceptual
    torch.cuda.empty_cache()


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file with argument defaults.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="DDP wrapper for MONAI 3D VAE training from CSV.",
        parents=[pre_ap],
    )
    ap.add_argument("--csv", default="", help="Path to CSV with columns: image[/path], sex, age, vol, group")
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True)
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument(
        "--data_split_json_path",
        type=str,
        default="data/patient_splits_image_ids_75_10_15.json",
        help="Path to JSON file containing train/val/test splits of image IDs.",
    )
    ap.add_argument("--seed", type=int, default=1017)

    ap.add_argument("--ae_epochs", type=int, default=50)
    ap.add_argument("--ae_lr", type=float, default=1e-4)
    ap.add_argument("--ae_latent_ch", type=int, default=8)
    ap.add_argument("--ae_kl", type=float, default=1e-6)
    ap.add_argument("--ae_adv_weight", type=float, default=0.01)
    ap.add_argument("--ae_perceptual_weight", type=float, default=0.001)
    ap.add_argument("--ae_kl_weight", type=float, default=1e-6)
    ap.add_argument("--ae_num_channels", default="64,128,256")
    ap.add_argument("--ae_attention_levels", default="0,0,0")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt (optional)")
    ap.add_argument(
        "--resume_ae_ckpt",
        default="",
        help="Path to AE checkpoint (.pt) or run directory to resume full AE training state.",
    )

    ap.add_argument("--outdir", default="ckpts")
    ap.add_argument("--out_prefix", default="")
    ap.add_argument("--out_postfix", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            cfg_defaults = json.load(f)
        if not isinstance(cfg_defaults, dict):
            raise ValueError(f"Config must be a JSON object: {pre_args.config}")
        known_keys = {a.dest for a in ap._actions}
        unknown_keys = sorted(set(cfg_defaults.keys()) - known_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config {pre_args.config}: {unknown_keys}"
            )
        ap.set_defaults(**cfg_defaults)

    args = ap.parse_args()
    if not args.csv:
        ap.error("--csv is required (pass on CLI or in --config).")
    torch_autocast = parse_bool(args.torch_autocast)
    rank, world_size, local_rank = setup_ddp()

    if rank == 0:
        run_name = f"{args.out_prefix}_{args.out_postfix}"
    else:
        run_name = ""
    run_name = broadcast_str(run_name, src=0)
    experiment_dir = os.path.join(args.outdir, run_name)
    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
    dist.barrier(device_ids=[local_rank])
    args.outdir = experiment_dir

    resume_ae_ckpt = args.resume_ae_ckpt
    if resume_ae_ckpt and os.path.isdir(resume_ae_ckpt):
        resume_ae_ckpt = os.path.join(resume_ae_ckpt, "AE_last.pt")
    if resume_ae_ckpt and not os.path.exists(resume_ae_ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_ae_ckpt}")

    log_path = os.path.join(args.outdir, f"run_rank{rank}.log")
    log_mode = "a" if resume_ae_ckpt else "w"
    log = open(log_path, log_mode, buffering=1)
    sys.stdout = log
    sys.stderr = log

    if rank == 0:
        cfg = vars(args).copy()
        cfg["resolved_resume_ae_ckpt"] = resume_ae_ckpt
        for k, v in list(cfg.items()):
            if isinstance(v, tuple):
                cfg[k] = list(v)
        with open(os.path.join(args.outdir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"\nOutput dir: {args.outdir}\n")

    spacing = tuple(float(x) for x in args.spacing.split(","))
    _ = tuple(int(x) for x in args.size.split(","))
    keys = ["image"]
    train_transforms = build_transforms(keys, spacing)

    device = torch.device(f"cuda:{local_rank}")
    base.seed_all(args.seed + rank)
    random.seed(args.seed + rank)

    try:
        n_samples = int(args.n_samples)
    except Exception:
        n_samples = args.n_samples

    train_loader, val_loader, test_loader, train_sampler, val_sampler, _ = make_dataloaders_ddp(
        args.csv,
        conditions=["age", "sex", "vol", "group"],
        train_transforms=train_transforms,
        n_samples=n_samples,
        data_split_json_path=args.data_split_json_path,
        batch_size=args.batch,
        num_workers=args.workers,
        seed=args.seed,
    )

    if rank == 0:
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        print(f"Number of test samples: {len(test_loader.dataset)}")

    ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
    ae_latent_ch = int(args.ae_latent_ch)
    ae_attention_levels = tuple(bool(int(x)) for x in args.ae_attention_levels.split(","))

    autoencoder = base.AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=ae_num_channels,
        latent_channels=ae_latent_ch,
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=ae_attention_levels,
    ).to(device)

    if args.ae_ckpt:
        base._load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)

    discriminator = base.PatchDiscriminator(
        spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1
    ).to(device)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    ddp_autoencoder = DDP(autoencoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    ddp_discriminator = DDP(
        discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )

    try:
        train_ae_ddp(
            ddp_autoencoder=ddp_autoencoder,
            ddp_discriminator=ddp_discriminator,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            val_sampler=val_sampler,
            val_interval=1,
            ae_epochs=args.ae_epochs,
            adv_weight=args.ae_adv_weight,
            perceptual_weight=args.ae_perceptual_weight,
            kl_weight=args.ae_kl_weight,
            lr=args.ae_lr,
            torch_autocast=torch_autocast,
            device=device,
            outdir=args.outdir,
            rank=rank,
            resume_ae_ckpt=resume_ae_ckpt,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
