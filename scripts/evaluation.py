import os

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
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn import L1Loss
from tqdm import tqdm
import copy
import torch.nn as nn

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
import json

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


def make_dataloaders_from_csv(csv_path, conditions = ['age', 'sex', 'vol'], train_transforms = None,
                              n_samples = None, train_val_split = 0.1, batch_size = 1, num_workers=8, seed = 1017):
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
        for c in conditions:
            sample[c] = row[c]
        data.append(sample)
    random.seed(seed)
    random.shuffle(data)
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



def make_dataloaders_for_eval(
    csv_path,
    samples_dir,
    N=100,
    conditions=('age', 'sex', 'vol'),
    real_transforms=None,          # transformation applied to real data
    gen_transforms = None, 
    n_samples=None,                # optional: cap rows read from CSV (like your train fn)
    batch_size=1,
    num_workers=8,
    seed=1017,
    shuffle=False,                 # eval default
):
    df = pd.read_csv(csv_path)

    # ----- build "real" data list (same style as make_dataloaders_from_csv) -----
    real_all = []
    for i, row in df.iterrows():
        if n_samples is not None and i == n_samples:
            break
        sample = {'image': row['image']}
        for c in conditions:
            sample[c] = row[c]
        real_all.append(sample)

    random.seed(seed)
    random.shuffle(real_all)

    # decide how many examples to use
    n_real = min(len(real_all), N)

    real_data = real_all[:n_real]
    real_ds = Dataset(data=real_data, transform=real_transforms)
    print(f'Transformed data shape (real data): {real_ds[0]["image"].shape}')
    print(f"Number of real samples: {len(real_ds)}")
    real_loader = DataLoader(
        real_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ----- build "gen" data list -----
    # grab synthetic nii.gz files
    nii_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".nii.gz")])
    n_gen = min(len(nii_files), n_real)  # keep counts aligned with real
    nii_files = nii_files[:n_gen]

    # attach same conditions as corresponding real samples (so batch has same keys)
    gen_data = []
    for k, fname in enumerate(nii_files):
        sample = {'image': os.path.join(samples_dir, fname)}
        # copy conditions from the matched real sample index (after shuffle)
        for c in conditions:
            sample[c] = real_data[k][c]
        gen_data.append(sample)

    gen_ds = Dataset(data=gen_data, transform=gen_transforms)
    print(f'Transformed data shape (synthetic data): {gen_ds[0]["image"].shape}')
    print(f"Number of synthetic samples: {len(gen_ds)}")
    gen_loader = DataLoader(
        gen_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return real_loader, gen_loader



'''
def make_dataloaders_for_eval(csv_path, samples_dir, N = 100, conditions = ['age', 'sex', 'vol'], train_transforms = None,
                              train_val_split = 0.1, batch_size = 1, num_workers=8, seed = 1017):
    df = pd.read_csv(csv_path)
    data = []
    for i, row in df.iterrows():
        sample = {}
        sample['image'] = row['image']
        for c in conditions:
            sample[c] = row[c]
        data.append(sample)
    random.seed(seed)
    random.shuffle(data)
    random.seed()
    n = min(len(os.listdir(samples_dir))//2, N)
    n = N
    real_data = data[:int(n)]
    real_ds = Dataset(data=real_data, transform=train_transforms)
    print(f'Transformed data shape (real data): {real_ds[0]["image"].shape}')
    print(f"Number of real samples: {len(real_ds)}")
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    data = []
    for nii in os.listdir(samples_dir):
        if '.nii.gz' in nii:
            sample = {}
            sample['image'] = os.path.join(samples_dir, nii)
            #for c in conditions:
                #sample[c] = row[c]
            data.append(sample)
    random.seed(seed)
    random.shuffle(data)
    random.seed()
    n = min(len(os.listdir(samples_dir))//2, N)
    n = N
    gen_data = data[:int(n)]
    gen_ds = Dataset(data=gen_data, transform=train_transforms)
    print(f'Transformed data shape (synthetic data): {gen_ds[0]["image"].shape}')
    print(f"Number of synthetic samples: {len(gen_ds)}")
    gen_loader = DataLoader(gen_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return real_loader, gen_loader

'''
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



class MedicalNetEmbedding(nn.Module):
    """
    Component to perform the perceptual evaluation with the networks pretrained by Chen, et al. "Med3D: Transfer
    Learning for 3D Medical Image Analysis". This class uses torch Hub to download the networks from
    "Warvito/MedicalNet-models".

    Args:
        net: {``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``}
            Specifies the network architecture to use. Defaults to ``"medicalnet_resnet10_23datasets"``.
        verbose: if false, mute messages from torch Hub load function.
        channel_wise: if True, the loss is returned per channel. Otherwise the loss is averaged over the channels.
                Defaults to ``False``.
    """

    def __init__(
        self, net: str = "medicalnet_resnet10_23datasets", verbose: bool = False, channel_wise: bool = False
    ) -> None:
        super().__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load("warvito/MedicalNet-models", model=net, verbose=verbose, trust_repo=True)
        self.eval()

        self.channel_wise = channel_wise

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using MedicalNet 3D networks. The input and target tensors are inputted in the
        pre-trained MedicalNet that is used for feature extraction. Then, these extracted features are normalised across
        the channels. Finally, we compute the difference between the input and target features and calculate the mean
        value from the spatial dimensions to obtain the perceptual loss.

        Args:
            input: 3D input tensor with shape BCDHW.
            target: 3D target tensor with shape BCDHW.

        """
        input = medicalnet_intensity_normalisation(input)
        target = medicalnet_intensity_normalisation(target)

        # Get model outputs
        feats_per_ch = 0
        for ch_idx in range(input.shape[1]):
            input_channel = input[:, ch_idx, ...].unsqueeze(1)
            target_channel = target[:, ch_idx, ...].unsqueeze(1)

            if ch_idx == 0:
                outs_input = self.model.forward(input_channel)
                outs_target = self.model.forward(target_channel)
                feats_per_ch = outs_input.shape[1]
            else:
                outs_input = torch.cat([outs_input, self.model.forward(input_channel)], dim=1)
                outs_target = torch.cat([outs_target, self.model.forward(target_channel)], dim=1)

        # Normalise through the channels
        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        return feats_input, feats_target


def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def medicalnet_intensity_normalisation(volume):
    """Based on https://github.com/Tencent/MedicalNet/blob/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b/datasets/brains18.py#L133"""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std



@torch.no_grad()
def _mean_and_cov(feats: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feats: [N, D]
    returns: mean[D], cov[D, D] (float64)
    Note: using (N-1) denominator (sample covariance), consistent with np.cov default.
    """
    if feats.ndim != 2:
        raise ValueError(f"feats must be [N, D], got {feats.shape}")
    n = feats.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples to compute covariance, got N={n}")

    x = feats.to(dtype=torch.float64)
    mu = x.mean(dim=0)
    xm = x - mu
    cov = (xm.T @ xm) / (n - 1)
    cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * eps
    return mu, cov

@torch.no_grad()
def _matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = mat.to(dtype=torch.float64)
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=eps)
    sqrt_vals = torch.sqrt(eigvals)
    # faster than diag: V * s then @ V^T
    S = (eigvecs * sqrt_vals.unsqueeze(0)) @ eigvecs.T
    return 0.5 * (S + S.T)

@torch.no_grad()
def frechet_distance(mu1, cov1, mu2, cov2, eps: float = 1e-6) -> float:
    mu1 = mu1.to(dtype=torch.float64)
    mu2 = mu2.to(dtype=torch.float64)
    cov1 = cov1.to(dtype=torch.float64)
    cov2 = cov2.to(dtype=torch.float64)

    I = torch.eye(cov1.shape[0], dtype=torch.float64, device=cov1.device)
    cov1 = cov1 + I * eps
    cov2 = cov2 + I * eps

    diff = mu1 - mu2
    sqrt_cov1 = _matrix_sqrt_psd(cov1, eps=eps)
    cov_prod = sqrt_cov1 @ cov2 @ sqrt_cov1
    cov_prod_sqrt = _matrix_sqrt_psd(cov_prod, eps=eps)

    fid = diff.dot(diff) + torch.trace(cov1 + cov2 - 2.0 * cov_prod_sqrt)
    fid = torch.clamp(fid, min=0.0)
    return float(fid.item())


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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class Inception2DFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        w = Inception_V3_Weights.DEFAULT
        m = inception_v3(weights=w, aux_logits=True)  # <-- must be True with weights
        m.fc = nn.Identity()                          # output 2048-d
        # optional: remove aux classifier to save a bit of memory
        m.AuxLogits = None
        m.eval()
        self.model = m

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.model(x)
        # Inception sometimes returns (logits, aux) depending on mode/version; guard:
        if isinstance(out, tuple):
            out = out[0]
        return out

def volume_to_slice_rgb01(vol: torch.Tensor, slice_idx: int, axis: int = 2,
                          clamp_z: float = 3.0) -> torch.Tensor:
    """
    vol: [B,1,D,H,W] (recommended) OR [B,D,H,W]
    returns: [B,3,H,W] float32 in ~[0,1]
    axis: which dim is the slice dim in [B,1,D,H,W] indexing (default D=2)
    """
    if vol.ndim == 4:
        vol = vol.unsqueeze(1)  # [B,1,D,H,W]
    if vol.ndim != 5:
        raise ValueError(f"Expected [B,1,D,H,W] or [B,D,H,W], got {vol.shape}")

    # pick slice (assumes axial if axis=2)
    if axis == 2:
        x = vol[:, :, slice_idx, :, :]  # [B,1,H,W]
    elif axis == 3:
        x = vol[:, :, :, slice_idx, :]  # [B,1,D,W] (coronal-like)
    elif axis == 4:
        x = vol[:, :, :, :, slice_idx]  # [B,1,D,H] (sagittal-like)
    else:
        raise ValueError("axis must be 2, 3, or 4 for [B,1,D,H,W]")

    # z-score per sample (per-slice is ok too, but per-sample is more stable)
    mean = x.mean(dim=(2,3), keepdim=True)
    std  = x.std(dim=(2,3), keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    # clamp + map to [0,1]
    x = x.clamp(-clamp_z, clamp_z)
    x = (x + clamp_z) / (2 * clamp_z)

    # to 3 channels
    x = x.repeat(1, 3, 1, 1).contiguous()
    return x.float()


class RunningGaussian:
    def __init__(self, feat_dim: int, device: torch.device):
        self.n = 0
        self.sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
        self.sum_xxt = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, feats: torch.Tensor):
        # feats: [B,D] float32/float16 OK
        x = feats.to(dtype=torch.float64)
        self.n += x.shape[0]
        self.sum += x.sum(dim=0)
        self.sum_xxt += x.T @ x

    @torch.no_grad()
    def finalize(self, eps: float = 1e-6):
        if self.n < 2:
            raise ValueError(f"Need at least 2 samples, got n={self.n}")
        mu = self.sum / self.n
        # sample covariance (N-1)
        cov = (self.sum_xxt - self.n * torch.outer(mu, mu)) / (self.n - 1)
        cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * eps
        return mu, cov


from typing import Dict, List

def _get_tensor_from_batch(batch):
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        if "image" in batch and torch.is_tensor(batch["image"]):
            return batch["image"]
        for v in batch.values():
            if torch.is_tensor(v):
                return v
        raise ValueError(f"Dict batch has no tensor values: keys={list(batch.keys())}")
    if isinstance(batch, (list, tuple)):
        for v in batch:
            if torch.is_tensor(v):
                return v
        raise ValueError("List/tuple batch has no tensor elements.")
    raise TypeError(f"Unsupported batch type: {type(batch)}")

@torch.no_grad()
def compute_slice_fids(
    real_loader,
    fake_loader,
    device: torch.device,
    axis: int = 2,                 # axis in [B,1,D,H,W] coordinates
    # D: sag, H: cor, W: axial
    max_slices: Optional[int] = 32,
    margin: int = 16,               # skip extreme ends
    eps: float = 1e-6,
) -> Dict[int, float]:
    feat_net = Inception2DFeatures().to(device).eval()

    # Peek a batch to infer depth (normalize to [B,1,D,H,W])
    real0 = _get_tensor_from_batch(next(iter(real_loader)))
    if real0.ndim == 4:
        real0 = real0.unsqueeze(1)   # [B,1,D,H,W] if originally [B,D,H,W]
    if real0.ndim != 5:
        raise ValueError(f"Expected [B,D,H,W] or [B,1,D,H,W], got {real0.shape}")

    depth = real0.shape[axis]
    print('Depth: ', depth)
    if depth <= 2 * margin:
        raise ValueError(f"depth={depth} too small for margin={margin}")

    # choose slice indices
    if max_slices is None or max_slices >= (depth - 2 * margin):
        slice_ids = list(range(margin, depth - margin))
    else:
        # uniformly spaced within [margin, depth-1-margin]
        slice_ids = (
            torch.linspace(margin, depth - 1 - margin, steps=max_slices)
            .round()
            .long()
            .unique()
            .tolist()
        )
    print('Computing FID from ', len(slice_ids), ' slices.')

    # running stats per chosen slice index

    # keep RunningGaussian on CPU
    stats_real = {s: RunningGaussian(2048, torch.device("cpu")) for s in slice_ids}
    stats_fake = {s: RunningGaussian(2048, torch.device("cpu")) for s in slice_ids}

    # pass 1: real
    for batch in real_loader:
        vol = _get_tensor_from_batch(batch).to(device, non_blocking=True)
        for s in slice_ids:
            img2d = volume_to_slice_rgb01(vol, s, axis=axis)   # [B,3,H,W] in ~[0,1]
            # in the loop
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
                feats = feat_net(img2d)                            # [B,2048]
                stats_real[s].update(feats.detach().cpu())

    # pass 2: fake
    for batch in fake_loader:
        vol = _get_tensor_from_batch(batch).to(device, non_blocking=True)
        for s in slice_ids:
            img2d = volume_to_slice_rgb01(vol, s, axis=axis)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
                feats = feat_net(img2d)                            # [B,2048]
                stats_fake[s].update(feats.detach().cpu())
    # FID per slice
    fids: Dict[int, float] = {}
    for s in slice_ids:
        mu_r, cov_r = stats_real[s].finalize(eps=eps)
        mu_f, cov_f = stats_fake[s].finalize(eps=eps)
        fids[s] = frechet_distance(mu_r, cov_r, mu_f, cov_f, eps=eps)

    return fids

def plot_slice_fids(fids: dict, title: str = "Per-slice FID", savepath: str | None = None):
    # Sort by slice index
    xs = sorted(fids.keys())
    ys = [fids[x] for x in xs]

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Slice index")
    plt.ylabel("FID")
    plt.title(title)
    plt.grid(True)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {savepath}")
    plt.close('all')


def compute_ssim_3d(real_loader, gen_loader, device):
    ssims = []
    for real_batch, gen_batch in zip(real_loader, gen_loader):
        real_vols = real_batch['image'].to(device)
        gen_vols = gen_batch['image'].to(device)
        ssims.append(ssim_3d(gen_vols.float(), real_vols.float()))
    return ssims


from metrics.regression import compute_ms_ssim, KernelType

@torch.no_grad()
def compute_ms_ssim_real_gen(
    real_loader,
    gen_loader,
    device,
    *,
    spatial_dims=3,
    data_range=1.0,
    kernel_type="gaussian",
    kernel_size=11,
    kernel_sigma=1.5,
    k1=0.01,
    k2=0.03,
    weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
):
    vals = []
    for real_batch, gen_batch in zip(real_loader, gen_loader):
        real_vols = real_batch["image"].to(device)  # [B,C,H,W,D]
        gen_vols  = gen_batch["image"].to(device)   # [B,C,H,W,D]

        if real_vols.shape != gen_vols.shape:
            raise ValueError(f"Shape mismatch: real {real_vols.shape} vs gen {gen_vols.shape}")

        ms = compute_ms_ssim(
            y_pred=gen_vols,
            y=real_vols,
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            weights=weights,
        )

        # be robust to [B] vs [B,1] vs [B, ...]
        vals.append(ms.reshape(ms.shape[0], -1).mean(dim=1).detach().cpu())

    all_vals = torch.cat(vals, dim=0) if vals else torch.empty(0)
    return {"mean": all_vals.mean().item(), "std": all_vals.std(unbiased=False).item(), "all": all_vals}



@torch.no_grad()
def renorm_weights(w):
    s = sum(w)
    return tuple([x / s for x in w])

def compute_ms_ssim_diversity_gen(
    gen_loader,
    device,
    *,
    spatial_dims=3,
    data_range=1.0,
    kernel_type="gaussian",
    kernel_size=11,
    kernel_sigma=1.5,
    k1=0.01,
    k2=0.03,
    weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    max_samples=32,          # how many generated volumes to use total
    max_pairs=512,           # cap pair count for speed
):
    # 1) collect up to max_samples generated volumes
    samples = []
    for batch in gen_loader:
        x = batch["image"].to(device)          # [B,C,H,W,D]
        for j in range(x.shape[0]):
            samples.append(x[j:j+1])           # keep as [1,C,H,W,D]
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    if len(samples) < 2:
        return {
            "pairwise_mean_ms_ssim": float("nan"),
            "pairwise_std_ms_ssim": float("nan"),
            "pairwise_mean_diversity": float("nan"),
            "all_pairwise_ms_ssim": [],   # JSON-safe
            "n_samples": len(samples),
            "n_pairs": 0,
        }

    x = torch.cat(samples, dim=0)  # [N,C,H,W,D]
    N = x.shape[0]

    # 2) choose pairs
    pairs = torch.combinations(torch.arange(N, device=device), r=2)  # [P,2]
    if pairs.shape[0] > max_pairs:
        idx = torch.randperm(pairs.shape[0], device=device)[:max_pairs]
        pairs = pairs[idx]

    a = x[pairs[:, 0]]
    b = x[pairs[:, 1]]

    # take first 3 default weights and renormalize
    weights3 = renorm_weights((0.0448, 0.2856, 0.3001))

    ms = compute_ms_ssim(
        y_pred=a, y=b,
        spatial_dims=3,
        data_range=1.0,
        kernel_type="gaussian",
        kernel_size=11,
        kernel_sigma=1.5,
        weights=weights3,   # <-- smaller volumes, renorm weights
    )

    vals = ms.reshape(ms.shape[0], -1).mean(dim=1).detach().cpu()  # [P]
    mean = vals.mean().item()
    std = vals.std(unbiased=False).item()

    return {
        "pairwise_mean_ms_ssim": mean,
        "pairwise_std_ms_ssim": std,
        "pairwise_mean_diversity": 1.0 - mean,
        "all_pairwise_ms_ssim": vals.tolist(),   # JSON-safe
        "n_samples": int(N),
        "n_pairs": int(vals.numel()),
    }


def save_fullview_like(arr, t1_img, out_path, dtype=None):
    if dtype is not None:
        arr = arr.astype(dtype)
    print('Saving to ', out_path)
    nib.save(nib.Nifti1Image(arr, t1_img.affine, t1_img.header), str(out_path))
    print(f'MIN: {arr.min():.4f}, MAX: {arr.max():.4f}, STD:{arr.std():.4f}')


# ------------
# CLI wrapper
# ------------
def main():

    ap = argparse.ArgumentParser(description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.")
    ap.add_argument("--csv", required=True, help="Path to CSV of real volumes; columns: image[/path], sex, age, vol, [target_label]")
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--generate_n_samples", type=int, default=100)

    ap.add_argument("--outdir", default="samples", help = 'Directory of generative volumes.')


    args = ap.parse_args()

    from datetime import datetime
    import json
    
    import sys
    log_path = os.path.join(args.outdir, 'run.log')
    log = open(log_path, "w", buffering=1)
    sys.stdout = log
    sys.stderr = log  # tracebacks too
    
    print(f"\n✅ Output dir: {args.outdir}\n")
    os.makedirs(args.outdir, exist_ok=True)

    # argparse Namespace -> dict
    cfg = vars(args).copy()

    # Make non-JSON types serializable (e.g., tuples)
    for k,v in list(cfg.items()):
        if isinstance(v, tuple): cfg[k] = list(v)
    with open(os.path.join(args.outdir, 'args.json'), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size    = tuple(int(x)   for x in args.size.split(","))
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️ Using {device}\n")


    seed = 1017
    g = torch.Generator(device=device).manual_seed(seed)
    random.seed(1017)
    seed_all(1017)

    #medicalnet = MedicalNetEmbedding()

    channel = 0  # 0 = Flair
    keys = ['image']
    real_transforms = transforms.Compose(
        [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        transforms.EnsureTyped(keys=keys),
        transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
        transforms.CropForegroundd(keys=keys, source_key='image'),
        transforms.DivisiblePadd(keys=keys, k=32, mode="constant",constant_values=-1.0),
        ]
    )

    gen_transforms = transforms.Compose(
        [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        transforms.EnsureTyped(keys=keys),
        transforms.Spacingd(keys=keys, pixdim=(1,1,1), mode=("bilinear")),
        transforms.CropForegroundd(keys=keys, source_key='image'),
        transforms.DivisiblePadd(keys=keys, k=32, mode="constant",constant_values=-1.0),
        ]
    )

    real_loader, gen_loader = make_dataloaders_for_eval(args.csv, samples_dir=f'{args.outdir}', N=300, conditions = ['age', 'sex', 'vol'], 
                                                        real_transforms = real_transforms, gen_transforms = gen_transforms, 
                                                        batch_size = args.batch, num_workers=args.workers, seed = 1017)
    
    ''' 
    t1img = nib.load('data/ADNI_turboprepout_whole_brain_0107/whole_brainADNI_011_S_0003_MR_MPR__GradWarp_Br_20061208110842803_S9128_I32234_normalized.nii.gz')
    n = 0
    os.makedirs('samples/read_whole_brain_0107', exist_ok=True)
    for batch in real_loader:
        if n == 20:
            break
        image = np.array(batch['image'].squeeze().cpu()).astype(np.float32)
        save_fullview_like(image, t1img, out_path=f'samples/read_whole_brain_0107/real_sample_{n}.nii.gz')
        n += 1

    '''    
    diversity = compute_ms_ssim_diversity_gen(real_loader, device)
    print('Diversity scores MS-SSIM REAL: \n', diversity)

    diversity = compute_ms_ssim_diversity_gen(gen_loader, device)
    print('Diversity scores MS-SSIM GENERATIVE: \n', diversity)
    with open(f"{args.outdir}/diversity.json", "w") as json_file:
        json.dump(diversity, json_file, indent=2)
    views = {'sag':2, 'cor':3, 'ax':4}
    for view in views.keys():
        axis = views[view]
        fids = compute_slice_fids(real_loader, gen_loader,device, axis = axis, max_slices=64, eps = 1e-6)
        with open(f"{args.outdir}/{view}_fids.json", "w") as json_file:
            json.dump(fids, json_file, indent=2)
        plot_slice_fids(fids, title = f"Per-slice FID ({view})", savepath = os.path.join(args.outdir, f'slice_fid_{view}.png'))

        fid_scores = np.array(list(fids.values()))
        fid_scores = fid_scores[fid_scores != 0] # remove 0 FIDs if any

        mean = np.mean(fid_scores)
        std = np.std(fid_scores)
        min_ = np.min(fid_scores)
        max_ = np.max(fid_scores)
        median = np.median(fid_scores)

        lines = [
            f"Mean FID ({view}): {mean}",
            f"Std FID ({view}): {std}", 
            f"Median FID ({view}): {median}",
            f"Min FID ({view}): {min_}",
            f"Max FID ({view}): {max_}",]
        
        for line in lines:
            print(line)

        out_path = os.path.join(args.outdir, 'eval_summary.txt')
        with open(out_path, "a") as f:
            for line in lines:
                print(line, file=f)

    ssim_scores = compute_ssim_3d(real_loader, gen_loader, device)
    print(ssim_scores)

    ssim_scores = np.array(ssim_scores)

    lines = [
        f"Mean SSIM: {np.mean(ssim_scores)}",
        f"Std SSIM: {np.std(ssim_scores)}"
        f"Median SSIM: {np.median(ssim_scores)}",
        f"Min SSIM: {np.min(ssim_scores)}",
        f"Max SSIM: {np.max(ssim_scores)}"
    ]

    for line in lines:
        print(line)

    out_path = os.path.join(args.outdir, 'eval_summary.txt')
    with open(out_path, "a") as f:
        for line in lines:
            print(line, file=f)


if __name__ == "__main__":
   main()
