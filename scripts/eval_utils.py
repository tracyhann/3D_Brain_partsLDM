import torch
from typing import Tuple
import math
import matplotlib.pyplot as plt
import os
from monai.utils import first, set_determinism
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
import torch.nn.functional as F
import pandas as pd
import random
'''
def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    # assume normalized to [0,1]
    mse = torch.mean((x - y) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)
'''


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 2.0, # [-1,1]
    eps: float = 1e-8,
    cap: float = 99.0,
) -> float:
    """
    x,y: [B,C,D,H,W]
    mask: same shape or [B,1,D,H,W] (1 for valid voxels)
    data_range: 1.0 for [0,1], 2.0 for [-1,1]
    returns: batch-mean PSNR (float)
    """
    if mask is None:
        mask = (y > -1.0).float()
    else:
        mask = mask.float()

    B = x.shape[0]
    diff2 = (x - y) ** 2

    diff2 = diff2.view(B, -1)
    m = mask.view(B, -1)

    denom = m.sum(dim=1).clamp_min(1.0)
    mse = (diff2 * m).sum(dim=1) / denom  # [B]

    # per-sample psnr then mean
    psnr_vals = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    psnr_vals = torch.clamp(psnr_vals, max=cap)
    return psnr_vals.mean().item()


'''
def ssim_3d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    # simple global SSIM surrogate (not windowed); fine for quick sanity checks
    brain_mask = (x > -0.9).float()  # or your segmentation / Bmask
    x = x*brain_mask
    brain_mask = (y > -0.9).float()  # or your segmentation / Bmask
    y = y*brain_mask
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(unbiased=False), y.var(unbiased=False)
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return (num / (den + eps)).item()

'''
def ssim_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 2.0, # [-1,1]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Global (non-windowed) SSIM-like score per sample.
    x, y: [B, C, D, H, W] (or broadcastable)
    mask: same shape as x/y or [B,1,D,H,W] boolean/0-1; if None, uses intersection x>-0.9 & y>-0.9
    returns: [B] tensor
    """
    if mask is None:
        mask = y > -1.0  # default mask: where x is not background
    mask = mask.to(dtype=x.dtype)

    # flatten per sample
    B = x.shape[0]
    x = x.view(B, -1)
    y = y.view(B, -1)
    m = mask.view(B, -1)

    # avoid empty masks
    denom = m.sum(dim=1).clamp_min(1.0)

    mu_x = (x * m).sum(dim=1) / denom
    mu_y = (y * m).sum(dim=1) / denom

    x0 = x - mu_x[:, None]
    y0 = y - mu_y[:, None]

    var_x = (m * x0 * x0).sum(dim=1) / denom
    var_y = (m * y0 * y0).sum(dim=1) / denom
    cov  = (m * x0 * y0).sum(dim=1) / denom

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2)

    return (num / (den + eps)).mean().item()



def _gaussian_1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-12)
    return g


def _gaussian_kernel_3d(window_size: int, sigma: float, device, dtype):
    """
    Returns [1,1,ws,ws,ws] Gaussian kernel suitable for conv3d.
    """
    g1 = _gaussian_1d(window_size, sigma, device, dtype)  # [ws]
    g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]  # [ws,ws,ws]
    k = g3.unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws,ws]
    return k


@torch.no_grad()
def ssim_3d_heatmap(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
    mask: torch.Tensor | None = None,
    padding: str = "same",
):
    """
    Compute *local* SSIM map for 3D volumes.

    Args:
        x, y: tensors shaped [D,H,W] or [B,1,D,H,W] (float32/float16 ok)
        data_range: dynamic range of input. If your images are in [-1,1], use 2.0.
                    If in [0,1], use 1.0.
        window_size: odd int recommended (e.g., 7, 9, 11)
        sigma: Gaussian std for window
        mask: optional [B,1,D,H,W] mask (1 where to compute mean; 0 ignore)
        padding: "same" (recommended) or "valid"

    Returns:
        ssim_map: [B,1,D,H,W] (or smaller if padding="valid")
        ssim_mean: float (mean over all voxels or masked voxels)
    """
    # ---- shape normalize ----
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    elif x.dim() == 4:
        x = x.unsqueeze(1)  # [B,1,D,H,W]
    if y.dim() == 3:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.dim() == 4:
        y = y.unsqueeze(1)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape. Got {tuple(x.shape)} vs {tuple(y.shape)}")

    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError(f"Expected single-channel volumes [B,1,D,H,W]. Got x: {tuple(x.shape)}")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    device = x.device
    dtype = x.dtype

    # ---- constants ----
    # Standard SSIM constants: K1=0.01, K2=0.03
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # ---- kernel ----
    k = _gaussian_kernel_3d(window_size, sigma, device=device, dtype=dtype)

    # padding
    if padding == "same":
        pad = window_size // 2
        pad_tuple = (pad, pad, pad, pad, pad, pad)  # W, W, H, H, D, D order for F.pad
        # reflect padding tends to behave nicer than zeros at edges
        x_pad = F.pad(x, pad_tuple, mode="reflect")
        y_pad = F.pad(y, pad_tuple, mode="reflect")
        conv_pad = 0
    elif padding == "valid":
        x_pad, y_pad = x, y
        conv_pad = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # ---- local moments ----
    mu_x = F.conv3d(x_pad, k, padding=conv_pad)
    mu_y = F.conv3d(y_pad, k, padding=conv_pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv3d(x_pad * x_pad, k, padding=conv_pad) - mu_x2
    sigma_y2 = F.conv3d(y_pad * y_pad, k, padding=conv_pad) - mu_y2
    sigma_xy = F.conv3d(x_pad * y_pad, k, padding=conv_pad) - mu_xy

    # numeric safety
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    # ---- SSIM map ----
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + eps)

    # ---- mean (optionally masked) ----
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 4:
            mask = mask.unsqueeze(1)
        if mask.shape != ssim_map.shape:
            raise ValueError(f"mask shape must match ssim_map. Got {tuple(mask.shape)} vs {tuple(ssim_map.shape)}")

        m = mask.to(device=device, dtype=ssim_map.dtype)
        ssim_mean = (ssim_map * m).sum() / (m.sum() + eps)
    else:
        ssim_mean = ssim_map.mean()

    return ssim_map, float(ssim_mean.item())


@torch.no_grad()
def _mean_and_cov(feats: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feats: [N, D] float tensor
    returns: (mean[D], cov[D, D]) in float64 for stability
    """
    if feats.ndim != 2:
        raise ValueError(f"feats must be [N, D], got {feats.shape}")
    x = feats.to(dtype=torch.float64)
    mu = x.mean(dim=0)
    # rowvar=False equivalent: we want cov over columns (features)
    xm = x - mu
    # unbiased=False (ML estimate) is common for FID
    cov = (xm.T @ xm) / (x.shape[0] - 1)
    # jitter for numerical stability
    cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * eps
    return mu, cov

@torch.no_grad()
def _matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Matrix square root for symmetric (PSD) matrices using eigen-decomposition.
    Returns a real, symmetric result with tiny negatives clipped.
    """
    # Ensure float64 for numerical stability
    A = mat.to(dtype=torch.float64)
    # Enforce symmetry (guards tiny asymmetries)
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = torch.linalg.eigh(A)  # guaranteed real for symmetric
    eigvals = torch.clamp(eigvals, min=eps)
    sqrt_vals = torch.sqrt(eigvals)
    S = eigvecs @ torch.diag(sqrt_vals) @ eigvecs.T
    # Symmetrize again to kill numerical drift
    return 0.5 * (S + S.T)

@torch.no_grad()
def frechet_distance(
    mu1: torch.Tensor,
    cov1: torch.Tensor,
    mu2: torch.Tensor,
    cov2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Core Fréchet distance between two Gaussians N(mu1, cov1) and N(mu2, cov2).
    All inputs can be on any device; computations are done in float64 for stability.

    Returns: scalar float (FID).
    """
    mu1 = mu1.to(dtype=torch.float64)
    mu2 = mu2.to(dtype=torch.float64)
    cov1 = cov1.to(dtype=torch.float64)
    cov2 = cov2.to(dtype=torch.float64)

    # Add small jitter to covariances to avoid singularities
    I = torch.eye(cov1.shape[0], dtype=torch.float64, device=cov1.device)
    cov1 = cov1 + I * eps
    cov2 = cov2 + I * eps

    diff = mu1 - mu2
    # Compute sqrt(cov1 * cov2) via the stable sandwich: sqrt(cov1) @ cov2 @ sqrt(cov1)
    sqrt_cov1 = _matrix_sqrt_psd(cov1, eps=eps)
    cov_prod = sqrt_cov1 @ cov2 @ sqrt_cov1
    cov_prod_sqrt = _matrix_sqrt_psd(cov_prod, eps=eps)

    # Traces and quadratic term
    trace_term = torch.trace(cov1 + cov2 - 2.0 * cov_prod_sqrt)
    fid = diff.dot(diff) + trace_term

    # Guard against tiny negative due to numeric error
    return float(torch.clamp(fid, min=0.0).item())

@torch.no_grad()
def fid_from_features(
    feats_fake: torch.Tensor,
    feats_real: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Convenience wrapper: compute FID directly from feature matrices.

    feats_*: [N, D] float tensors of activations (e.g., pooled Inception features).
    Returns: scalar float FID.
    """
    if feats_real.shape[1] != feats_fake.shape[1]:
        raise ValueError(f"Feature dims must match, got {feats_real.shape} vs {feats_fake.shape}")

    mu1, cov1 = _mean_and_cov(feats_real, eps=eps)
    mu2, cov2 = _mean_and_cov(feats_fake, eps=eps)
    return frechet_distance(mu1, cov1, mu2, cov2, eps=eps)


def plot_recon_loss(epoch_recon_loss_list, epoch_ssim_loss_list, epoch_psnr_loss_list, title = 'Reconstruction Loss Curve', outdir = 'ckpts', filename = 'AE_recon_loss.png'):
    plt.style.use("ggplot")
    plt.title(title, fontsize=20)
    plt.plot(epoch_recon_loss_list, color="blue", linewidth=2.0, label="Recon L1-Loss")
    plt.plot(epoch_ssim_loss_list, color="green", linewidth=2.0, label="SSIM")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(outdir, filename))
    plt.close('all')

    plt.style.use("ggplot")
    plt.title(title, fontsize=20)
    plt.plot(epoch_ssim_loss_list, color="green", linewidth=2.0, label="SSIM")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(outdir, filename.replace('recon_loss', 'SSIM')))
    plt.close('all')

    plt.style.use("ggplot")
    plt.title(title, fontsize=20)
    plt.plot(epoch_psnr_loss_list, color="orange", linewidth=2.0, label="PSNR")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(outdir, filename.replace('recon_loss', 'PSNR')))
    plt.close('all')

def plot_adversarial_loss(epoch_gen_loss_list, epoch_disc_loss_list, title = 'Adversarial Training Curves', outdir = 'ckpts', filename = 'AE_disc_loss.png'):
    plt.title(title, fontsize=20)
    plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
    plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(outdir, filename))
    plt.close('all')

def plot_reconstructions(data_batch, reconstruction, idx = 0, channel=0, title = 'Reconstruction', outdir = 'ckpts', filename = 'AE_recons.png'):
    fig, axs = plt.subplots(nrows=2, ncols=3)
    # Plot axial, coronal and sagittal slices of a training sample
    img = data_batch["image"][idx, channel].detach().cpu().numpy()
    for ax in axs[0]:
        ax.axis("off")
    ax = axs[0, 0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[0, 1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[0, 2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
    print('Real IMG values check: ', img.min(), img.max(), img.mean(), img.std())
    # Plot axial, coronal and sagittal slices of its recon sample
    img = reconstruction[idx, channel].detach().cpu().numpy()
    for ax in axs[1]:
        ax.axis("off")
    ax = axs[1, 0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1, 1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[1, 2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
    axs[0, 1].set_title(f"{title}\nOriginal", fontsize=14)
    axs[1, 1].set_title("Reconstruction", fontsize=14)
    print('Recon IMG values check: ', img.min(), img.max(), img.mean(), img.std())
    plt.savefig(os.path.join(outdir, filename))
    plt.close('all')



def sample_ldm(data_loader, autoencoder, unet, scheduler, inferer, device, idx = 0, channel = 0, outdir = 'ckpts', filename = 'synthetic'):
    autoencoder.eval()
    unet.eval()

    with torch.no_grad():
        check_data = first(data_loader)
        z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
    latent_shape = z.shape  # (B, C, D, H, W)

    noise = torch.randn(latent_shape)
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)

    with torch.no_grad():
        synthetic_images = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
        )
        #print('GENERATED IMAGE\n', synthetic_images)
        #synthetic_images = torch.clamp(synthetic_images, -1.0, 1.0)

    img = synthetic_images[idx, channel].cpu().numpy()
    print('IMG values check: ', img.min(), img.max(), img.mean(), img.std())
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



import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.amp import autocast
from monai.utils import first

@torch.no_grad()
def sample_ldm_dit(
    data_loader,
    autoencoder,
    dit,
    scheduler,
    device,
    idx=0,
    channel=0,
    outdir="ckpts",
    filename="synthetic",
    num_inference_steps=1000,
    scale_factor=None,   # pass the SAME scale_factor used in training (recommended)
):
    """
    DDPM sampling in AE latent space using a DiT denoiser.

    autoencoder: frozen AE; must support encode_stage_2_inputs + decode_stage_2_outputs
    dit: predicts noise eps_hat in latent space: eps_hat = dit(z_t, t)
    scheduler: DDPMScheduler
    """
    autoencoder.eval()
    dit.eval()
    os.makedirs(outdir, exist_ok=True)

    # 1) Get latent shape (and optionally scale_factor if not provided)
    batch = first(data_loader)
    images = batch["image"].to(device)  # [B,1,96,128,96]

    with autocast(device_type="cuda", enabled=(device.type == "cuda")):
        z0 = autoencoder.encode_stage_2_inputs(images)  # [B,3,12,16,12]

    latent_shape = z0.shape  # [B,Cz,d,h,w]
    B = latent_shape[0]

    if scale_factor is None:
        # Not ideal (should match training), but works for quick sanity checks.
        sf = (1.0 / torch.std(z0)).detach()
        scale_factor = sf
        print(f"[sample_ldm_dit] WARNING: scale_factor not provided, using 1/std(z0)={scale_factor.item():.4f}")
    else:
        if not torch.is_tensor(scale_factor):
            scale_factor = torch.tensor(scale_factor, device=device)
        scale_factor = scale_factor.to(device)

    # 2) Start from Gaussian noise in SCALED latent space (match training)
    z = torch.randn(latent_shape, device=device)  # z_T (scaled space)

    # 3) Reverse diffusion
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.to(device)

    for t in timesteps:
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            eps_hat = dit(z, t_batch)  # [B,Cz,d,h,w], predicts noise

        # scheduler.step returns an object or tuple depending on version; handle both
        out = scheduler.step(model_output=eps_hat, timestep=t, sample=z)
        z = out.prev_sample if hasattr(out, "prev_sample") else out[0]

    # 4) Unscale + decode
    z_final = z / scale_factor

    with autocast(device_type="cuda", enabled=(device.type == "cuda")):
        synthetic_images = autoencoder.decode_stage_2_outputs(z_final)  # [B,1,96,128,96] typically

    # 5) Save preview + nifti
    img = synthetic_images[idx, channel].float().cpu().numpy()  # [96,128,96]
    print("IMG values check:", img.min(), img.max(), img.mean(), img.std())

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(img[..., img.shape[2] // 2], cmap="gray")
    axs[1].imshow(img[:, img.shape[1] // 2, :], cmap="gray")
    axs[2].imshow(img[img.shape[0] // 2, :, :], cmap="gray")
    plt.savefig(os.path.join(outdir, filename + ".png"), bbox_inches="tight")
    plt.close(fig)

    nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)),
             os.path.join(outdir, filename + ".nii.gz"))



import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import matplotlib.pyplot as plt
import nibabel as nib
from monai.utils import first

@torch.no_grad()
def sample_ldm_dit_parts(
    data_loader,
    autoencoder,
    dit,
    scheduler,
    device,
    part_keys=("lhemi", "rhemi", "cerebellum"),
    num_parts=3,
    cfg_scale=1.0,            # 1.0 = no CFG; try 2-5
    idx=0,
    channel=0,
    outdir="ckpts",
    filename="synthetic",
    num_inference_steps=250,
    scale_factor=None,        # MUST match training
):
    """
    DDPM sampling in AE latent space using a Part-conditioned DiT (local+global attention).

    Assumes dit forward supports:
      eps_hat = dit(z_t, t_batch, part_latents=z_parts, num_parts=num_parts)

    z_parts is shaped (B, P, Cz, d, h, w).
    """
    autoencoder.eval()
    dit.eval()
    os.makedirs(outdir, exist_ok=True)

    # 1) Grab a batch for shape + conditioning parts
    batch = first(data_loader)
    images = batch["image"].to(device)  # [B,1,96,128,96]
    parts = [batch[k].to(device) for k in part_keys]  # list of [B,1,96,128,96]
    assert len(parts) == num_parts, f"len(part_keys)={len(parts)} must equal num_parts={num_parts}"

    # 2) Encode whole brain just to get latent shape (and optional scale_factor)
    with autocast(device_type="cuda", enabled=(device.type == "cuda")):
        z0 = autoencoder.encode_stage_2_inputs(images)  # [B,Cz,d,h,w]

    latent_shape = z0.shape
    B, Cz, d, h, w = latent_shape

    # scale_factor handling
    if scale_factor is None:
        scale_factor = (1.0 / torch.std(z0)).detach()
        print(f"[sample_ldm_dit_parts] WARNING: scale_factor not provided. Using 1/std(z0)={scale_factor.item():.6f}")
    else:
        if not torch.is_tensor(scale_factor):
            scale_factor = torch.tensor(scale_factor, device=device)
        scale_factor = scale_factor.to(device)

    # 3) Encode parts for conditioning (and SCALE them like training!)
    with autocast(device_type="cuda", enabled=(device.type == "cuda")):
        z_parts_list = [autoencoder.encode_stage_2_inputs(p) for p in parts]  # each [B,Cz,d,h,w]

    z_parts_list = [zp * scale_factor for zp in z_parts_list]
    z_parts = torch.stack(z_parts_list, dim=1)  # [B,P,Cz,d,h,w]

    # 4) Start from noise in SCALED latent space
    z = torch.randn(latent_shape, device=device)

    # 5) Reverse diffusion
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.to(device)

    for t in timesteps:
        t_int = int(t)
        t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            if cfg_scale is None or cfg_scale == 1.0:
                eps_hat = dit(z, t_batch, part_latents=z_parts, num_parts=num_parts)
            else:
                # unconditional pass: zero conditioning
                z_parts_null = torch.zeros_like(z_parts)
                eps_uncond = dit(z, t_batch, part_latents=z_parts_null, num_parts=num_parts)
                eps_cond   = dit(z, t_batch, part_latents=z_parts,      num_parts=num_parts)
                eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        out = scheduler.step(model_output=eps_hat, timestep=t, sample=z)
        z = out.prev_sample if hasattr(out, "prev_sample") else out[0]

    # 6) Unscale + decode
    z_final = z / scale_factor
    with autocast(device_type="cuda", enabled=(device.type == "cuda")):
        synthetic_images = autoencoder.decode_stage_2_outputs(z_final)  # [B,1,96,128,96] typically

    # 7) Save preview + nifti
    img = synthetic_images[idx, channel].float().cpu().numpy()
    print("IMG values check:", img.min(), img.max(), img.mean(), img.std())

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(img[..., img.shape[2] // 2], cmap="gray")
    axs[1].imshow(img[:, img.shape[1] // 2, :], cmap="gray")
    axs[2].imshow(img[img.shape[0] // 2, :, :], cmap="gray")
    plt.savefig(os.path.join(outdir, filename + ".png"), bbox_inches="tight")
    plt.close(fig)

    nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)),
             os.path.join(outdir, filename + ".nii.gz"))



def sample_ldm_from_fused(data_loader, autoencoder, unet, scheduler, inferer, device, use_mask = True, idx = 0, channel = 0, outdir = 'ckpts', filename = 'synthetic'):
    autoencoder.eval()
    unet.eval()

    with torch.no_grad():
        check_data = first(data_loader)
        z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
        scale_factor = 1.0 / torch.std(z)
        #coarse = check_data['coarse'].to(device)           # conditioning
        df = pd.read_csv('data/whole_brain_data_fused_synth_1114.csv')
        coarse_list = list(df['coarse'])
        coarse_path = random.choice(coarse_list)

        # load -> (D,H,W)
        coarse_np = nib.load(coarse_path).get_fdata()

        # torch -> (1, D, H, W)
        coarse = torch.from_numpy(coarse_np).float().unsqueeze(0).to(device)

        # -> (1, 1, D, H, W)
        coarse = coarse.unsqueeze(0)
        coarse = coarse.repeat(2, 1, 1, 1, 1)  # (2, 1, D, H, W)
        mask = check_data.get('mask', None)
        if use_mask:
            if mask is None:
                raise KeyError(f"use_mask=True but missing '{'mask'}' in batch.")
            mask = mask.to(device)

        # encode target + coarse (no grad)
        with torch.no_grad():
            zc = autoencoder.encode_stage_2_inputs(coarse)     # (B,Cz,d,h,w)

        # build cond latent (concat coarse latent + mask latent)
        if use_mask:
            mask_lat = F.interpolate(mask, size=z.shape[2:], mode="nearest")  # (B,1,d,h,w)
            cond_lat = torch.cat([zc * scale_factor, mask_lat], dim=1)         # (B,Cz+1,d,h,w)
        else:
            mask_lat = None
            cond_lat = zc * scale_factor  # (B,Cz,d,h,w)
    latent_shape = z.shape  # (B, C, D, H, W)

    noise = torch.randn(latent_shape)
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)

    with torch.no_grad():
        synthetic_images = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
            conditioning=cond_lat,
            mode='concat'
        )
        #synthetic_images = torch.clamp(synthetic_images, -1.0, 1.0)

    img = synthetic_images[idx, channel].cpu().numpy()
    print('IMG values check: ', img.min(), img.max(), img.mean(), img.std())
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






def sample_ldm_cond(
    data_loader,
    autoencoder,
    unet,
    scheduler,
    device,
    idx=0,
    channel=0,
    outdir="ckpts",
    filename="synthetic",
    num_inference_steps=250,
    part_id=0,          # which part to generate
):
    """
    Sample from a part-conditioned latent diffusion model.

    Assumes:
      - UNet was trained with in_channels = latent_channels + num_parts
      - Conditioning is done by concatenating a one-hot part_map in latent space.
    """
    autoencoder.eval()
    unet.eval()

    # 1) Get latent shape and scale_factor the same way as in training
    with torch.no_grad():
        batch = next(iter(data_loader))
        images = batch["image"].to(device)                # [B, C, D, H, W]
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            z = autoencoder.encode_stage_2_inputs(images) # [B, Cz, Dz, Hz, Wz]

    # latent shape and channels
    latent_shape = z.shape
    B, Cz, Dz, Hz, Wz = latent_shape

    # scale_factor computed like in train_ldm
    scale_factor = 1.0 / torch.std(z)
    print(f"[sample_ldm] Using scale_factor = {scale_factor.item():.4f}")

    # infer num_parts from UNet in_channels
    num_parts = unet.in_channels - Cz
    if num_parts <= 0:
        raise ValueError(
            f"UNet.in_channels={unet.in_channels}, latent_channels={Cz}, "
            f"so inferred num_parts={num_parts} (must be > 0)."
        )

    # 2) Sample initial noise in latent space (scaled)
    # You can use B>1, but idx only makes sense if B>idx
    noise = torch.randn(latent_shape, device=device)  # [B, Cz, Dz, Hz, Wz]

    # 3) Set up DDPM timesteps
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = scheduler.timesteps.to(device)        # e.g. [1000, ..., 0]

    x = noise

    with torch.no_grad():
        for t in timesteps:
            # t: scalar tensor
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # 3a) Build part_map in LATENT space
            #     same part_id for all samples here; you can generalize later
            part_ids = torch.full((B,), int(part_id), device=device, dtype=torch.long)  # [B]
            part_onehot = F.one_hot(part_ids, num_classes=num_parts).float()           # [B, num_parts]

            part_map = part_onehot.view(B, num_parts, 1, 1, 1)                         # [B, P, 1, 1, 1]
            part_map = part_map.expand(-1, -1, Dz, Hz, Wz)                              # [B, P, Dz, Hz, Wz]

            # 3b) Concatenate conditioning channels to current latent sample
            x_cond = torch.cat([x, part_map], dim=1)                                    # [B, Cz+P, Dz, Hz, Wz]

            # 3c) UNet predicts noise
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                model_output = unet(x_cond, t_batch)                                    # [B, Cz, Dz, Hz, Wz]

            # 3d) DDPM update
            x = scheduler.step(model_output=model_output, timestep=t, sample=x)[0]

        # x is final latent (scaled); unscale before decode
        latents_final = x / scale_factor

        # 4) Decode latents back to image space
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            synthetic_images = autoencoder.decode_stage_2_outputs(latents_final)        # [B, C, D, H, W]

    # 5) Visualization / saving (same as before)
    img = synthetic_images[idx, channel].cpu().numpy()   # [D, H, W]
    print("IMG values check: ", img.min(), img.max(), img.mean(), img.std())

    os.makedirs(outdir, exist_ok=True)

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(img[..., img.shape[2] // 2], cmap="gray")
    axs[1].imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    axs[2].imshow(img[img.shape[0] // 2, ...], cmap="gray")

    plt.savefig(os.path.join(outdir, filename + ".png"), bbox_inches="tight")
    plt.close(fig)

    # Save as nifti (float32)
    img_nii = img.astype(np.float32)
    nib.save(
        nib.Nifti1Image(img_nii, np.eye(4)),
        os.path.join(outdir, filename + ".nii.gz"),
    )




def sample_ldm_cond(
    data_loader,
    autoencoder,
    unet,
    scheduler,
    device,
    idx=0,
    channel=0,
    outdir="ckpts",
    filename="synthetic",
    num_inference_steps=250,
    part_id=0,          # which part to generate
):
    """
    Sample from a part-conditioned latent diffusion model.

    Assumes:
      - UNet was trained with in_channels = latent_channels + num_parts
      - Conditioning is done by concatenating a one-hot part_map in latent space.
    """
    autoencoder.eval()
    unet.eval()

    # 1) Get latent shape and scale_factor the same way as in training
    with torch.no_grad():
        batch = next(iter(data_loader))
        images = batch["image"].to(device)                # [B, C, D, H, W]
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            z = autoencoder.encode_stage_2_inputs(images) # [B, Cz, Dz, Hz, Wz]

    # latent shape and channels
    latent_shape = z.shape
    B, Cz, Dz, Hz, Wz = latent_shape

    # scale_factor computed like in train_ldm
    scale_factor = 1.0 / torch.std(z)
    print(f"[sample_ldm] Using scale_factor = {scale_factor.item():.4f}")

    # infer num_parts from UNet in_channels
    num_parts = unet.in_channels - Cz
    if num_parts <= 0:
        raise ValueError(
            f"UNet.in_channels={unet.in_channels}, latent_channels={Cz}, "
            f"so inferred num_parts={num_parts} (must be > 0)."
        )

    # 2) Sample initial noise in latent space (scaled)
    # You can use B>1, but idx only makes sense if B>idx
    noise = torch.randn(latent_shape, device=device)  # [B, Cz, Dz, Hz, Wz]

    # 3) Set up DDPM timesteps
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = scheduler.timesteps.to(device)        # e.g. [1000, ..., 0]

    x = noise

    with torch.no_grad():
        for t in timesteps:
            # t: scalar tensor
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # 3a) Build part_map in LATENT space
            #     same part_id for all samples here; you can generalize later
            part_ids = torch.full((B,), int(part_id), device=device, dtype=torch.long)  # [B]
            part_onehot = F.one_hot(part_ids, num_classes=num_parts).float()           # [B, num_parts]

            part_map = part_onehot.view(B, num_parts, 1, 1, 1)                         # [B, P, 1, 1, 1]
            part_map = part_map.expand(-1, -1, Dz, Hz, Wz)                              # [B, P, Dz, Hz, Wz]

            # 3b) Concatenate conditioning channels to current latent sample
            x_cond = torch.cat([x, part_map], dim=1)                                    # [B, Cz+P, Dz, Hz, Wz]

            # 3c) UNet predicts noise
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                model_output = unet(x_cond, t_batch)                                    # [B, Cz, Dz, Hz, Wz]

            # 3d) DDPM update
            x = scheduler.step(model_output=model_output, timestep=t, sample=x)[0]

        # x is final latent (scaled); unscale before decode
        latents_final = x / scale_factor

        # 4) Decode latents back to image space
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            synthetic_images = autoencoder.decode_stage_2_outputs(latents_final)        # [B, C, D, H, W]

    # 5) Visualization / saving (same as before)
    img = synthetic_images[idx, channel].cpu().numpy()   # [D, H, W]
    print("IMG values check: ", img.min(), img.max(), img.mean(), img.std())

    os.makedirs(outdir, exist_ok=True)

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(img[..., img.shape[2] // 2], cmap="gray")
    axs[1].imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    axs[2].imshow(img[img.shape[0] // 2, ...], cmap="gray")

    plt.savefig(os.path.join(outdir, filename + ".png"), bbox_inches="tight")
    plt.close(fig)

    # Save as nifti (float32)
    img_nii = img.astype(np.float32)
    nib.save(
        nib.Nifti1Image(img_nii, np.eye(4)),
        os.path.join(outdir, filename + ".nii.gz"),
    )





def _get_prev_sample(step_out):
    # MONAI schedulers often return an object with .prev_sample
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    # sometimes dict-like
    if isinstance(step_out, dict):
        return step_out.get("prev_sample", step_out.get("sample", None))
    # sometimes tuple
    if isinstance(step_out, (tuple, list)) and len(step_out) > 0:
        return step_out[0]
    return step_out


def scheduler_to(sched, device):
    # Move common MONAI/DDPM scheduler buffers to device (best-effort).
    for name in [
        "betas", "alphas", "alphas_cumprod",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "timesteps", "sigmas",
    ]:
        if hasattr(sched, name):
            t = getattr(sched, name)
            if isinstance(t, torch.Tensor):
                setattr(sched, name, t.to(device))
    return sched


def sample_ldm_refine(
    data_loader,
    autoencoder,
    unet,
    scheduler,
    device,
    scale_factor: float,
    idx: int = 0,
    channel: int = 0,
    outdir: str = "ckpts",
    filename: str = "refined",
    # refinement knobs
    strength: float = 0.35,          # 0..1, higher => more rewrite
    num_inference_steps: int = 1000,  # 100-250 is a good starting range
    coarse_key: str = "coarse",
    mask_key: str = "mask",
    use_mask: bool = True,
    # constraint knobs
    hard_keep: bool = True,          # keep known voxels in image space at the end
):
    autoencoder.eval()
    unet.eval()

    batch = first(data_loader)
    coarse = batch[coarse_key].to(device)  # (B,1,D,H,W)
    mask = batch.get(mask_key, None)
    if use_mask:
        if mask is None:
            raise KeyError(f"use_mask=True but batch missing '{mask_key}'")
        mask = mask.to(device)  # (B,1,D,H,W)

    # --- encode coarse to latent (scaled) ---
    zc = autoencoder.encode_stage_2_inputs(coarse)      # (B,Cz,d,h,w) unscaled
    zc = zc * scale_factor                               # scaled

    if use_mask:
        mask_lat = F.interpolate(mask, size=zc.shape[2:], mode="nearest")  # (B,1,d,h,w)
        cond = torch.cat([zc, mask_lat], dim=1)                             # (B,Cz+1,d,h,w)
    else:
        mask_lat = None
        cond = zc  # (B,Cz,d,h,w)

    # --- set inference timesteps ---
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    scheduler_to(scheduler, device)   # ✅ important

    # strength => where to start in the reverse chain (SDEdit-style)
    strength = float(max(0.0, min(1.0, strength)))
    start_index = int(strength * (len(scheduler.timesteps) - 1))
    start_t = scheduler.timesteps[start_index].to(device)

    # initialize from partially noised coarse latent
    noise = torch.randn_like(zc)
    tt0 = torch.full((zc.shape[0],), int(start_t.item()), device=device, dtype=torch.long)
    zt = scheduler.add_noise(original_samples=zc, noise=noise, timesteps=tt0)

    # --- reverse diffusion loop from start_t -> 0 ---
    # scheduler.timesteps is typically descending, so we slice from start_index to end.
    for t in scheduler.timesteps[start_index:]:
        t_int = int(t.item())
        tt = torch.full((zt.shape[0],), t_int, device=device, dtype=torch.long)

        # concat conditioning into UNet input channels
        model_in = torch.cat([zt, cond], dim=1)  # (B, Cz + (Cz+1), d,h,w) if use_mask

        eps_hat = unet(model_in, tt)  # predict noise, shape (B,Cz,d,h,w)

        step_out = scheduler.step(model_output=eps_hat, timestep=tt, sample=zt)
        zt = _get_prev_sample(step_out)

    # --- decode refined latent back to image space ---
    z_final = zt / scale_factor
    refined = autoencoder.decode_stage_2_outputs(z_final)  # (B,1,D,H,W)

    # Optional: hard keep known region in IMAGE space at the end
    if hard_keep and use_mask:
        refined = refined * (1.0 - mask) + coarse * mask

    img = refined[idx, channel].detach().cpu().numpy()
    print("IMG values check:", img.min(), img.max(), img.mean(), img.std())

    os.makedirs(outdir, exist_ok=True)

    # Save 3-slice preview
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(img[..., img.shape[2] // 2], cmap="gray")
    axs[1].imshow(img[:, img.shape[1] // 2, :], cmap="gray")
    axs[2].imshow(img[img.shape[0] // 2, :, :], cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{filename}_s{strength:.2f}.png"), dpi=150)
    plt.close(fig)

    # Save NIfTI (identity affine like your original)
    nib.save(nib.Nifti1Image(img, np.eye(4)), os.path.join(outdir, f"{filename}_s{strength:.2f}.nii.gz"))




def plot_unet_loss(epoch_loss_list, title = 'UNET Loss Curves', outdir = 'ckpts', filename = 'UNET_loss.png'):
    plt.plot(epoch_loss_list)
    plt.title(title, fontsize=20)
    plt.plot(epoch_loss_list)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(outdir, filename))
    plt.close('all')
