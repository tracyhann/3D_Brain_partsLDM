import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
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

from eval_utils import psnr, ssim_3d, fid_from_features, plot_recon_loss, plot_adversarial_loss, plot_reconstructions, sample_ldm, plot_unet_loss

from glob import glob
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import nibabel as nib
import numpy as np
import random
import argparse

# Avoid shared-memory exhaustion in DataLoader workers (e.g., limited /dev/shm)
mp.set_sharing_strategy("file_system")

def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # determinism (good for debugging; may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(1017)


def make_dataloaders_from_csv(csv_path, parts = ['lhemi', 'rhemi', 'cerebellum'], 
                              conditions = ['age', 'sex', 'vol', 'part'], train_transforms = None,
                              n_samples = None, train_val_split = 0.1, batch_size = 1, 
                              num_workers=8, seed = 1017):
    #Make a list of dicts. Keys must match your transforms.
    #images = sorted(glob("data/ADNI_turboprepout_whole_brain/*.nii.gz")) 
    #labels = sorted(glob("/data/ct/labelsTr/*.nii.gz"))
    #conds = pd.read_csv('data/whole_brain_data.csv').to_dict()

    df = pd.read_csv(csv_path)
    data = []
    for i, row in df.iterrows():
        if i == n_samples:
            break
        sample = {}
        sample['image'] = row['image']
        sample['template_head_mask'] = 'data/template/mask_dilated_r3.nii.gz'
        for part in parts:
            sample[part] = row[part]
            sample[part+'_mask'] = row[part+'_mask']
        for c in conditions:
            sample[c] = row[c]
        data.append(sample)
    random.seed(seed)
    random.shuffle(data)
    random.seed()
    split = int(len(data)*train_val_split)
    train_data, val_data = data[:-split], data[-split:]
    train_ds = Dataset(data=train_data, transform=train_transforms)
    print(f'Transformed data shape: {train_ds[0]["image"].shape}')
    print(f"Number of training samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ds = Dataset(data=val_data, transform=train_transforms)
    print(f"Number of validation samples: {len(val_ds)}")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader




def check_train_data(train_loader, idx = 0, title = 'Train data example', save_path = None):
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
    if not save_path == None:
        plt.savefig(save_path)
    plt.close('all')





def _load_ckpt_into_ae(ae, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    missing, unexpected = ae.load_state_dict(state, strict=False)


def _save_ckpt(path: str, unet: torch.nn.Module, opt, scaler, epoch: int, global_step: int,
               ema: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None):
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


def _load_ckpt_into_unet(unet: torch.nn.Module, ckpt_path: str, device) -> Tuple[int, int]:
    """Loads either a raw state_dict or a packaged checkpoint. Returns (epoch, global_step) if present."""
    sd = torch.load(ckpt_path, map_location=device)
    if "state_dict" in sd:
        missing, unexpected = unet.load_state_dict(sd["state_dict"], strict=False)
        print(f"[LDM] resumed UNet (packaged): missing={len(missing)} unexpected={len(unexpected)}")
        return int(sd.get("epoch", 0)), 0 #int(sd.get("global_step", 0))
    else:
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        print(f"[LDM] resumed UNet (raw): missing={len(missing)} unexpected={len(unexpected)}")
        return 0, 0


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def train_ae(autoencoder, train_loader, val_loader = None, val_interval = 1, ae_epochs = 100, 
             adv_weight = 0.01, perceptual_weight = 0.001, kl_weight = 1e-6, 
             lr=1e-4, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), outdir = 'ckpts'):

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)


    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)



    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), lr=lr)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=lr)



    n_epochs = ae_epochs
    autoencoder_warm_up_n_epochs = int(0.2*n_epochs) # default: 20% of training epochs
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_ssim_train_list, epoch_psnr_train_list = [], []
    val_recon_epoch_loss_list = []
    epoch_ssim_val_list, epoch_psnr_val_list = [], []
    intermediary_images = []
    n_example_images = 4

    ae_best = None
    ae_best_loss = float('inf')

    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        ssim_train, psnr_train = 0, 0
        ssim_val, psnr_val = 0, 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)  # choose only one of Brats channels

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            #reconstruction = torch.clamp(reconstruction, -1.0, 1.0)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss
            psnr_train += psnr(reconstruction.float(), images.float())
            ssim_train += ssim_3d(reconstruction.float(), images.float())

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

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

        #print(epoch_gen_loss_list)

        plot_recon_loss(epoch_recon_loss_list, epoch_ssim_train_list, epoch_psnr_train_list, title = f'Train Reconstruction Loss Curve_ep{epoch+1}', outdir = outdir, filename = 'AE_train_recon_loss.png')
        plot_adversarial_loss(epoch_gen_loss_list, epoch_disc_loss_list, title = f'Adversarial Training Curves_ep{epoch+1}', outdir = outdir, filename = 'AE_disc_loss.png')
        plot_reconstructions(batch, reconstruction, idx = 0, channel=0, title = f'Train Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_train_recons.png')

        if val_loader is not None and (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            val_epoch_loss = 0
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=70)
            val_progress_bar.set_description(f"Val Epoch {epoch}")
            with torch.no_grad():
                for val_step, val_batch in val_progress_bar:
                    val_images = val_batch["image"].to(device)
                    val_reconstruction, _, _ = autoencoder(val_images)
                    val_recons_loss = l1_loss(val_reconstruction.float(), val_images.float())
                    val_epoch_loss += val_recons_loss.item()
                    psnr_val += psnr(val_reconstruction.float(), val_images.float())
                    ssim_val += ssim_3d(val_reconstruction.float(), val_images.float())
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
            plot_recon_loss(val_recon_epoch_loss_list, epoch_ssim_val_list, epoch_psnr_val_list, title = f'Val Reconstruction Loss Curve_ep{epoch+1}', outdir = outdir, filename = 'AE_val_recon_loss.png')

            if val_epoch_loss / (val_step + 1) < ae_best_loss:
                print('Updating best AE checkpoint...')
                ae_best_loss = val_epoch_loss / (val_step + 1)
                ae_best = copy.deepcopy(autoencoder)
                _save_ckpt(os.path.join(outdir, "AE_best.pt"), autoencoder, opt=optimizer_g, scaler=None, epoch=epoch+1, global_step=None, 
                            extra={"opt_d": optimizer_d, "train_rec_loss": epoch_recon_loss_list, "train_disc_loss": epoch_disc_loss_list, 
                                   "train_gen_loss": epoch_gen_loss_list, "train_ssim": epoch_ssim_train_list, "train_psnr": epoch_psnr_train_list,
                                   "val_rec_loss": val_recon_epoch_loss_list, "val_ssim": epoch_ssim_val_list, "val_psnr": epoch_psnr_val_list})
                plot_reconstructions(val_batch, val_reconstruction, idx = 0, channel=0, title = f'Val Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_val_recons.png')
            if epoch + 1 == n_epochs:
                print('Final epoch, saving last AE checkpoint and val reconstructions...')
                plot_reconstructions(val_batch, val_reconstruction, idx = 0, channel=0, title = f'Val Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_last_val_recons.png')
        _save_ckpt(os.path.join(outdir, "AE_last.pt"), autoencoder, opt=optimizer_g, scaler=None, epoch=epoch+1, global_step=None, 
                        extra={"opt_d": optimizer_d, "train_rec_loss": epoch_recon_loss_list, "train_disc_loss": epoch_disc_loss_list, 
                               "train_gen_loss": epoch_gen_loss_list, "train_ssim": epoch_ssim_train_list, "train_psnr": epoch_psnr_train_list,
                               "val_rec_loss": val_recon_epoch_loss_list, "val_ssim": epoch_ssim_val_list, "val_psnr": epoch_psnr_val_list})
    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()
    print('Reutrning the last and best AEs.')
    return autoencoder, ae_best


'''
# 5) Iterate
for batch in loader:
    imgs, segs = batch["image"], batch["label"]
    # ... forward pass ...
'''


# ---------------------------
# Morphology on binary masks
# ---------------------------
def dilate3d(mask: torch.Tensor, r: int) -> torch.Tensor:
    """
    mask: [B,1,H,W,D] float {0,1}
    """
    if r <= 0:
        return mask
    k = 2 * r + 1
    return F.max_pool3d(mask, kernel_size=k, stride=1, padding=r)

def erode3d(mask: torch.Tensor, r: int) -> torch.Tensor:
    """
    Erosion via complement + dilation.
    """
    if r <= 0:
        return mask
    return 1.0 - dilate3d(1.0 - mask, r)

def seam_band(mask: torch.Tensor, r: int) -> torch.Tensor:
    """
    seam = dilate(mask,r) - erode(mask,r)
    """
    if r <= 0:
        return torch.zeros_like(mask)
    return (dilate3d(mask, r) - erode3d(mask, r)).clamp(0, 1)

# ---------------------------
# Downsample masks to latent grid (96->12 etc.)
# ---------------------------
def down8_max(mask_img: torch.Tensor) -> torch.Tensor:
    """
    mask_img: [B,1,96,128,96] -> [B,1,12,16,12]
    Uses max-pool so if any voxel in the 8x8x8 block is "on", latent mask is on.
    """
    return F.max_pool3d(mask_img, kernel_size=8, stride=8)

def expand_to_latent_channels(m_lat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    m_lat: [B,1,h,w,d], z: [B,C,h,w,d] -> [B,C,h,w,d]
    """
    return m_lat.expand(z.shape[0], z.shape[1], z.shape[2], z.shape[3], z.shape[4])



def known_and_overlap_from_list(masks, device):
    # masks: list of [B,1,H,W,D]
    m_bool = [(m > 0.5) for m in masks]
    m_sum = torch.zeros_like(m_bool[0], dtype=torch.int16, device=device)
    for m in m_bool:
        m_sum += m.to(torch.int16)

    known_mask   = (m_sum >= 1).float()
    overlap_mask = (m_sum >= 2).float()
    return known_mask, overlap_mask

import os
import numpy as np
import nibabel as nib
import torch

def save_masks_nifti(
    masks: dict,
    outdir: str,
    prefix: str = "masks",
    idx: int = 0,
    affine: np.ndarray | None = None,
    make_uint8: bool = True,
):
    """
    Save masks (B,1,H,W,D) or (B,H,W,D) tensors to NIfTI files.
    Saves one subject: batch index `idx`.
    """
    os.makedirs(outdir, exist_ok=True)
    if affine is None:
        affine = np.eye(4)

    for name, t in masks.items():
        if t is None:
            continue

        # Move to CPU tensor
        if isinstance(t, torch.Tensor):
            x = t.detach().cpu()
        else:
            raise TypeError(f"Mask '{name}' is not a torch.Tensor: {type(t)}")

        # Accept shapes: [B,1,H,W,D], [B,H,W,D], [1,H,W,D], [H,W,D]
        if x.ndim == 5:          # [B,C,H,W,D]
            x = x[idx]
            if x.shape[0] == 1:  # [1,H,W,D] -> [H,W,D]
                x = x[0]
            else:
                # If multi-channel, save each channel separately
                for c in range(x.shape[0]):
                    vol = x[c].numpy()
                    if make_uint8:
                        vol = (vol > 0.5).astype(np.uint8)
                    fname = os.path.join(outdir, f"{prefix}_{name}_c{c}.nii.gz")
                    nib.save(nib.Nifti1Image(vol, affine), fname)
                continue

        elif x.ndim == 4:        # [B,H,W,D] or [C,H,W,D]
            x = x[idx].numpy()
        elif x.ndim == 3:        # [H,W,D]
            x = x.numpy()
        else:
            raise ValueError(f"Mask '{name}' has unsupported shape {tuple(x.shape)}")

        if make_uint8:
            x = (x > 0.5).astype(np.uint8)
        else:
            x = x.astype(np.float32)

        fname = os.path.join(outdir, f"{prefix}_{name}.nii.gz")
        nib.save(nib.Nifti1Image(x, affine), fname)

    print(f"Saved masks to: {outdir} (prefix={prefix}, idx={idx})")


def build_proto1_from_batch(batch, bg_val: float = -1.0,
                            r_int: int = 2, r_seam: int = 4, save_example = True):
    """
    Returns:
      cond_img: [B, Ccond, 96,128,96]  (image-space cond)
      x_known_merged: [B,1,96,128,96] (merged known canvas, for clamping)
      masks dict: {mL,mR,mC,O,known,I_safe,B,U} all [B,1,96,128,96]
    """
    pL = batch["lhemi"].to(dtype=torch.float32)
    pR = batch["rhemi"].to(dtype=torch.float32)
    pC = batch["cerebellum"].to(dtype=torch.float32)

    mL = batch["lhemi_mask"].to(dtype=torch.float32)
    mR = batch["rhemi_mask"].to(dtype=torch.float32)
    mC = batch["cerebellum_mask"].to(dtype=torch.float32)
    mHead = batch['template_head_mask'].to(dtype=torch.float32)
    mL = (mL > 0).float()
    mR = (mR > 0).float()
    mC = (mC > 0).float()
    mHead = (mHead > 0).float()

    # Per-part masked intensity channels (keeps conflicts explicit)
    xL = pL * mL + bg_val * (1 - mL)
    xR = pR * mR + bg_val * (1 - mR)
    xC = pC * mC + bg_val * (1 - mC)

    # Interior + seam per part
    IL = erode3d(mL, random.randint(0, r_int))
    IR = erode3d(mR, random.randint(0, r_int))
    IC = erode3d(mC, random.randint(0, r_int))

    I = (IL + IR + IC).clamp(0, 1)
    B = mHead.clamp(0, 1)


    masks = [mL, mR, mC]
    device = batch["lhemi_mask"].device
    known, O = known_and_overlap_from_list(masks, device)

    # Preserve only non-overlapping interior
    I_safe = (I * (1 - O)).clamp(0, 1)

    # Unknown-ish region (editable) excluding seam band and safe interior
    U = ((1 - I_safe)).clamp(0, 1)

    # A merged canvas used ONLY for clamping interiors (priority order is arbitrary; adjust if needed)
    x_known = torch.full_like(pL, bg_val)
    x_known = x_known * (1 - mL) + pL * mL
    x_known = x_known * (1 - mR) + pR * mR
    x_known = x_known * (1 - mC) + pC * mC
    coarse = x_known * (1 - O) + bg_val * O

    # Conditioning channels (image-space)
    # You can reorder; keep it consistent between train/infer.
    # cond_img = torch.cat([coarse, mL, mR, mC, O], dim=1)  # [B,8,H,W,D]
    cond_img = torch.cat([coarse, U], dim=1) # [B,2,H,W,D]
    masks = dict(mL=mL, mR=mR, mC=mC, O=O, coarse=coarse, I_safe=I_safe, B=B, U=U)
    if save_example == True:
        save_masks_nifti(masks, outdir="ckpts/masks_debug", prefix="proto1", idx=0,make_uint8=False)
    return cond_img, coarse, masks


import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_ldm_proto1(
    data_loader, autoencoder, unet, scheduler, device,
    r_int=2, r_seam=5, bg_val=-1.0,
    idx=0, channel=0,
    outdir="ckpts", filename="synthetic",
    num_inference_steps=1000,
    clamp_every=1,   # 1 = clamp each step; 5 = clamp every 5 steps
):
    autoencoder.eval()
    unet.eval()

    batch = first(data_loader)

    # Build cond + known canvas from batch
    cond_img, coarse, masks = build_proto1_from_batch(batch, bg_val=bg_val, r_int=r_int, r_seam=r_seam)

    x_known = coarse.to(device)                 # [B,1,96,128,96]
    I_safe  = masks["I_safe"].float().to(device) # [B,1,96,128,96]
    B  = masks["B"].float().to(device) # [B,1,96,128,96]

    # Encode z_known (clamp target)
    z_known = autoencoder.encode_stage_2_inputs(x_known).to(device)  # [B,3,12,16,12]

    # Build I_safe_lat3
    I_safe_lat = F.interpolate(I_safe, size=z_known.shape[-3:], mode="nearest")     # [B,1,12,16,12]
    I_safe_lat3 = I_safe_lat.expand(-1, z_known.shape[1], -1, -1, -1)               # [B,3,12,16,12]

    # Build I_safe_lat3
    B_lat = F.interpolate(B, size=z_known.shape[-3:], mode="nearest")     # [B,1,12,16,12]
    B_lat3 = B_lat.expand(-1, z_known.shape[1], -1, -1, -1)               # [B,3,12,16,12]


    # Build cond_lat: intensities trilinear, masks nearest
    cond_img = cond_img.to(device)  # [B,2,96,128,96]
    x_int = cond_img[:, :1]
    x_msk = cond_img[:, 1:]

    x_int_lat = F.interpolate(x_int, size=z_known.shape[-3:], mode="trilinear", align_corners=False)
    x_msk_lat = F.interpolate(x_msk, size=z_known.shape[-3:], mode="nearest")
    cond_lat = torch.cat([x_int_lat, x_msk_lat], dim=1)  # [B,2,12,16,12]

    # Sample latent noise
    z = torch.randn_like(z_known)

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

    # Sampling loop with clamp
    timesteps = scheduler.timesteps
    for i, t in enumerate(timesteps):
        if clamp_every and (i % clamp_every == 0):
            outside = (1.0 - B_lat3)

            keep = torch.clamp(I_safe_lat3 + outside, 0, 1)  # 1 = clamp
            z = z * (1 - keep) + z_known * keep

        # UNet predicts eps (concat conditioning)
        B = z.shape[0]

        # t from scheduler is scalar; make it [B]
        t_b = torch.full((B,), int(t), device=z.device, dtype=torch.long)

        model_in = torch.cat([z, cond_lat], dim=1) # [B, 3+2, 12,16,12]
        eps_hat = unet(model_in, t_b)

        # scheduler step
        # Most MONAI schedulers return an object or tuple; handle both.
        out = scheduler.step(eps_hat, t, z)
        z = out.prev_sample if hasattr(out, "prev_sample") else out[0]

    # final clamp
    outside = (1.0 - B_lat3)
    keep = torch.clamp(I_safe_lat3 + outside, 0, 1)  # 1 = clamp
    z = z * (1 - keep) + z_known * keep

    # Decode to image space
    x = autoencoder.decode_stage_2_outputs(z)  # [B,1,96,128,96] (or similar)
    synthetic_images = torch.clamp(x, -1.0, 1.0)

    # ---- save one sample ----
    img = synthetic_images[idx, channel].cpu().numpy()
    print("IMG values check:", img.min(), img.max(), img.mean(), img.std())

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
    plt.savefig(os.path.join(outdir, filename + '.png'))
    nib.save(nib.Nifti1Image(img, np.eye(4)), os.path.join(outdir, filename + '.nii.gz'))
    plt.close('all')



def train_ldm(unet, train_loader, autoencoder, r_int = 2, r_seam = 5, bg_val = -1.0, ldm_epochs = 150,
              lr=1e-4, scale_factor = None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
              outdir = 'ckpts', sample_every = 25):

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

    with torch.no_grad():
        with autocast(device_type = 'cuda', enabled=(device.type == "cuda")):
                check_data = first(train_loader)
                z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
    if scale_factor == None:
        scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")


    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=lr)


    n_epochs = ldm_epochs
    epoch_loss_list = []
    autoencoder.eval()
    scaler = GradScaler()

    unet_best = None
    unet_best_loss = float('inf')

    for p in autoencoder.parameters(): p.requires_grad = False
    for epoch in range(n_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            x_gt = batch["image"].to(device)  # [B,1,96,128,96]
                    
            # Build cond + known canvas from batch
            cond_img, coarse, masks = build_proto1_from_batch(batch, bg_val=bg_val, r_int=r_int, r_seam=r_seam)

            x_known = coarse.to(device)                 # [B,1,96,128,96]
            I_safe  = masks["I_safe"].float().to(device) # [B,1,96,128,96]
            B  = masks["B"].float().to(device) # [B,1,96,128,96]

            # precompute these once per batch (no grad)
            with torch.no_grad():

                x_known = x_known.to(device=device, dtype=torch.float32, non_blocking=True)
                x_known = x_known.contiguous(memory_format=torch.contiguous_format)
                z_gt    = autoencoder.encode_stage_2_inputs(x_gt)      # [B,3,12,16,12]
                z_known = autoencoder.encode_stage_2_inputs(x_known)   # [B,3,12,16,12]
            
            # Build cond_lat: intensities trilinear, masks nearest
            cond_img = cond_img.to(device)  # [B,2,96,128,96]
            x_int = cond_img[:, :1]
            x_msk = cond_img[:, 1:]

            with autocast(device_type = 'cuda', enabled=(device.type == "cuda")):
                # Build I_safe_lat3
                I_safe_lat = F.interpolate(I_safe, size=z_known.shape[-3:], mode="nearest")     # [B,1,12,16,12]
                I_safe_lat3 = I_safe_lat.expand(-1, z_known.shape[1], -1, -1, -1)               # [B,3,12,16,12]

                # Build I_safe_lat3
                B_lat = F.interpolate(B, size=z_known.shape[-3:], mode="nearest")     # [B,1,12,16,12]
                B_lat3 = B_lat.expand(-1, z_known.shape[1], -1, -1, -1)               # [B,3,12,16,12]

                # Sample timestep + noise
                x_int_lat = F.interpolate(x_int, size=(z_gt.shape[-3], z_gt.shape[-2], z_gt.shape[-1]),
                                    mode="trilinear", align_corners=False).to(device)
                x_msk_lat = F.interpolate(x_msk, size=(z_gt.shape[-3], z_gt.shape[-2], z_gt.shape[-1]),
                                        mode="nearest").to(device)
                eps = torch.randn_like(z_gt).to(device)
                t   = torch.randint(0, scheduler.num_train_timesteps, (z_gt.shape[0],), device=z_gt.device).long().to(device)

                # make z_t, but anchor I_safe with *noised* known latent
                z_t_gt    = scheduler.add_noise(z_gt,    eps, t).to(device)
                z_t_known = scheduler.add_noise(z_known, eps, t).to(device)
                outside = (1.0 - B_lat3)
                keep = torch.clamp(I_safe_lat3 + outside, 0, 1)  # 1 = clamp
                z_t = z_t_gt*(1 - keep) + z_t_known*keep
                z_t = z_t.to(device)

                # build cond_lat correctly (you already do this part)
                cond_lat = torch.cat([x_int_lat, x_msk_lat], dim=1).to(device)  # [B,5,12,16,12]

                optimizer_diff.zero_grad(set_to_none=True)

                # IMPORTANT: for concat mode, feed UNet the concatenated latent
                eps_hat = unet(torch.cat([z_t, cond_lat], dim=1), t).to(device)   # or unet(x=..., timesteps=t) depending on your wrapper

                loss = F.mse_loss(eps_hat.float(), eps.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if epoch_loss / (step + 1) < unet_best_loss:
            print('Updating best UNet + Conditioner checkpoint...')
            unet_best_loss = epoch_loss / (step + 1)
            unet_best = copy.deepcopy(unet)
            _save_ckpt(os.path.join(outdir, "UNET_best.pt"), unet, opt=optimizer_diff, scaler=scaler, epoch=epoch+1, global_step=None, 
                extra={"train_loss": epoch_loss_list, "scale_factor": float(scale_factor)})
            sample_ldm_proto1(train_loader, autoencoder, unet, scheduler, device,
                              r_int = r_int, r_seam=r_seam, bg_val = bg_val, idx = 0, channel = 0, 
                       outdir = outdir, filename = f'synthetic_ep{epoch+1}_BEST')

        if (epoch + 1) % sample_every == 0:
            sample_ldm_proto1(train_loader, autoencoder, unet, scheduler, device,
                              r_int = r_int, r_seam=r_seam, bg_val = bg_val, idx = 0, channel = 0, 
                       outdir = outdir, filename = f'synthetic_ep{epoch+1}')
            
        _save_ckpt(os.path.join(outdir, "UNET_last.pt"), unet, opt=optimizer_diff, scaler=scaler, epoch=epoch+1, global_step=None, 
                    extra={"train_loss": epoch_loss_list, "scale_factor": float(scale_factor)})
        plot_unet_loss(epoch_loss_list, title = f'UNET Loss Curves_ep{epoch+1}', outdir = outdir, filename = 'UNET_loss.png')





# ------------
# CLI wrapper
# ------------
def main():
    ap = argparse.ArgumentParser(description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.")
    ap.add_argument("--csv", required=True, help="Path to CSV with columns: image[/path], sex, age, vol, [target_label]")
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train_val_split", type=float, default=0.1)

    ap.add_argument("--n_samples", default="ALL")
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
    #ap.add_argument("--ae_factors", default="1,2,2,2")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt (optional)")
    #ap.add_argument("--ae_decoder_only", action="store_true", help="Fine-tune decoder only")


    # LDM config
    ap.add_argument("--ldm_epochs", type=int, default=150)
    ap.add_argument("--ldm_lr", type=float, default=1e-4)
    ap.add_argument("--ldm_use_cond", default="False", help="Use [part_vol_norm,sex,age] conditioning")
    ap.add_argument("--ldm_num_channels", default="128,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_r_int", default=2, type=int, help='Inpainting mask radius')
    ap.add_argument("--ldm_r_seam", default=5, type=int, help='Generative field radius')
    ap.add_argument("--ldm_ckpt", default="", help="Resume UNet weights (optional)")
    ap.add_argument("--ldm_sample_every", type=int, default=25, help="Synthesize samples every N epochs")


    ap.add_argument("--outdir", default="ckpts")
    ap.add_argument("--out_prefix", default="")


    args = ap.parse_args()

    from datetime import datetime
    import json
    experiment_dir = os.path.join(args.outdir, f"run_{args.out_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    args.outdir = experiment_dir
    print(f"\n✅ Output dir: {args.outdir}\n")

    # argparse Namespace -> dict
    cfg = vars(args).copy()

    # Make non-JSON types serializable (e.g., tuples)
    for k,v in list(cfg.items()):
        if isinstance(v, tuple): cfg[k] = list(v)
    with open(os.path.join(args.outdir, 'args.json'), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size    = tuple(int(x)   for x in args.size.split(","))

    channel = 0  # 0 = Flair
    keys = ['image']
    parts = ['lhemi', 'rhemi', 'cerebellum']
    for part in parts:
        keys.append(part)
        keys.append(part+'_mask')
    keys.append('template_head_mask')
    
    affine_per_part = [
        transforms.RandAffined(
            keys=[k],
            prob=0.9,
            rotate_range=(0.10, 0.10, 0.10),      # radians (~5.7°)
            translate_range=(8, 8, 8),            # voxels
            scale_range=(0.05, 0.05, 0.05),       # +/- 5%
            shear_range=(0.02, 0.02, 0.02),
            mode="bilinear",
            padding_mode="constant",
        ) for k in keys[1:]
    ]
    train_transforms = transforms.Compose(
        [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        transforms.EnsureTyped(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
        #transforms.CropForegroundd(keys="image", source_key="image"),
        #transforms.CenterSpatialCropd(keys=["image"], roi_size=size), 
        ##*affine_per_part,
        transforms.SpatialPadd(keys=keys, spatial_size=size, mode='constant', constant_values=-1.0), # MRI volumes are [-1,1] normalized
        #transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        ]
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️ Using {device}\n")
    
    seed = 1017
    g = torch.Generator(device=device).manual_seed(seed)
    random.seed(1017)
    seed_all(1017)

    try:
        n_samples = int(args.n_samples)
    except:
        n_samples = args.n_samples

    train_loader, val_loader = make_dataloaders_from_csv(args.csv, conditions = ['age', 'sex', 'vol'], train_transforms = train_transforms,
                                n_samples = n_samples, train_val_split = args.train_val_split, batch_size = args.batch, num_workers=args.workers, seed = 1017)

    ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
    ae_latent_ch = int(args.ae_latent_ch)

    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1, 
        num_channels=ae_num_channels, # default: (64, 128, 256, 512)
        latent_channels=ae_latent_ch, # default: 3
        num_res_blocks=2, # default: 2
        norm_num_groups=32, # default: 32
        norm_eps=1e-06,
        attention_levels=(False, False, True, True),
    )
    autoencoder.to(device)

    if args.ae_ckpt != "":
        _load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)

    if args.stage in ["ae", "both"]:
        last_ae, best_ae = train_ae(autoencoder, train_loader, val_loader, val_interval = 1, ae_epochs = args.ae_epochs, 
                                    adv_weight = args.ae_adv_weight, perceptual_weight = args.ae_perceptual_weight, kl_weight = args.ae_kl_weight, 
                                    lr=args.ae_lr, device = device, outdir = args.outdir)
        if best_ae:
            autoencoder = best_ae
    

    ldm_num_channels = tuple(int(x) for x in args.ldm_num_channels.split(","))
    ldm_num_head_channels = tuple(int(x) for x in args.ldm_num_head_channels.split(","))
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=5, # default: 3
        out_channels=3, # default: 3
        num_res_blocks=2,
        num_channels=ldm_num_channels, # default: (128, 256, 512)
        attention_levels=(False, True, True),
        num_head_channels=ldm_num_head_channels, # default: (0, 64, 64)
        norm_num_groups=32, norm_eps=1e-6,
        resblock_updown=True,
        upcast_attention=True,
        use_flash_attention=False
    )
    unet.to(device)

    scale_factor = None
    if args.ldm_ckpt != "":
        _load_ckpt_into_unet(unet, args.ldm_ckpt, device)
        ckpt = torch.load(args.ldm_ckpt, map_location=device)
        try:
            ckpt = torch.load(args.ldm_ckpt, map_location=device)
            scale_factor = float(ckpt['extra']['scale_factor'])
            print('Loading scale factor = ', scale_factor)
        except:
            scale_factor = None

    if args.stage in ["ldm", "both"]:
        train_ldm(unet, train_loader, autoencoder, r_int = args.ldm_r_int, r_seam = args.ldm_r_seam, ldm_epochs = args.ldm_epochs,
              lr=args.ldm_lr, scale_factor=scale_factor, device = device, outdir = args.outdir, sample_every = args.ldm_sample_every)


if __name__ == "__main__":
   main()
