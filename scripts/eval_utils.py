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

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    # assume normalized to [0,1]
    mse = torch.mean((x - y) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def ssim_3d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    # simple global SSIM surrogate (not windowed); fine for quick sanity checks
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(unbiased=False), y.var(unbiased=False)
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return (num / (den + eps)).item()

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