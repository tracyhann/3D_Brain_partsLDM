#!/usr/bin/env python3
# ae_recon_infer.py
# Evaluate a trained MONAI AutoencoderKL on the CSV-defined dataset (val split).
# Saves reconstructions (NIfTI + mid-slice PNGs) and prints L1 / PSNR / SSIM.

import os, math, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import nibabel as nib
import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.data import Dataset, set_track_meta
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, SpatialPadd, CenterSpatialCropd, EnsureTyped, MapTransform
)
from generative.networks.nets import AutoencoderKL

set_track_meta(False)

# ----------------------------
# Utilities
# ----------------------------
def save_nii(vol: torch.Tensor, path: Path):
    """vol: [1,1,D,H,W] tensor in [0,1] (or similar)."""
    arr = vol.squeeze(0).squeeze(0).detach().cpu().numpy()
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))

def save_mid_slices(vol: torch.Tensor, stem: str, outdir: Path):
    v = vol.squeeze(0).squeeze(0).detach().cpu().numpy()
    zc, yc, xc = [d // 2 for d in v.shape]
    for name, sl in [("ax", v[zc, :, :]), ("cor", v[:, yc, :]), ("sag", v[:, :, xc])]:
        sln = (sl - sl.min()) / (sl.max() - sl.min() + 1e-7)
        imageio.imwrite(str(outdir / f"{stem}_{name}.png"), (sln * 255).astype(np.uint8))

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

# ----------------------------
# Optional center finder (matches your training)
# ----------------------------
class FindNonBGCenter(MapTransform):
    def __init__(self, key="image", bg=None, tol=0.0):
        super().__init__(keys=(key,))
        self.key, self.bg, self.tol = key, bg, tol
    def __call__(self, d):
        a = d[self.key][0].cpu().numpy()
        bg = self.bg
        if bg is None:
            u, c = np.unique(a, return_counts=True)
            bg = float(u[c.argmax()])
        mask = np.abs(a - bg) > self.tol
        idx = np.argwhere(mask)
        if idx.size == 0:
            raise RuntimeError("No foreground found.")
        d["roi_center"] = idx.mean(axis=0).astype(int)
        return d

# ----------------------------
# Data
# ----------------------------

def make_loaders_from_csv(
   csv_path: str,
   spacing=(1.0, 1.0, 1.0),
   size=(160, 224, 160),
   batch=1,
   workers=4,
   val_frac=0.1,
) -> Tuple[DataLoader, DataLoader]:
   import pandas as pd
   from pathlib import Path as P
   df = pd.read_csv(csv_path)


   # Normalize column names we accept
   rename_map = {}
   if "path" in df.columns and "image" not in df.columns:
       rename_map["path"] = "image"
   if "seg_path" in df.columns and "seg" not in df.columns:
       rename_map["seg_path"] = "seg"
   if rename_map:
       df = df.rename(columns=rename_map)


   # Required minimal set
   for col in ("image", "sex", "age"):
       if col not in df.columns:
           raise ValueError(f"CSV missing required column '{col}'")


   has_seg = "seg" in df.columns


   # Build items list
   items: List[Dict[str, Any]] = []
   for _, r in df.iterrows():
       it: Dict[str, Any] = {
           "image": str(Path(r["image"]).expanduser()),
           "sex": float(r["sex"]),
           "age": float(r["age"]),
           "vol": float(r['vol'])
       }
       if has_seg and not pd.isna(r["seg"]):
           it["seg"] = str(Path(r["seg"]).expanduser())
       if "target_label" in df.columns and not pd.isna(r["target_label"]):
           it["target_label"] = int(r["target_label"])
       items.append(it)


   # Robust age bounds for normalization
   ages = np.array([it["age"] for it in items], float)
   age_min = float(np.percentile(ages, 1))
   age_max = float(np.percentile(ages, 99))

   vols = np.array([it["vol"] for it in items], float)
   vol_min = float(np.percentile(vols, 1))
   vol_max = float(np.percentile(vols, 99))


   tx = Compose([
       LoadImaged(keys=["image"]),
       EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
       Orientationd(keys=["image"], axcodes="RAS"),
       Spacingd(keys=["image"], pixdim=spacing, mode="nearest"),
       SpatialPadd(keys=["image"], spatial_size=size),
       FindNonBGCenter(key="image", bg=None, tol=0.0),
       CenterSpatialCropd(keys=["image"], roi_size=size),
       EnsureTyped(keys=["image"]),
   ])




   # Split train/val (last val_frac for val)
   n = len(items)
   v = max(1, int(round(n * val_frac))) if n > 1 else 1
   train_items = items[:max(n - v, 1)]
   val_items   = items[max(n - v, 0):] if n > 1 else items


   train_ds = Dataset(train_items, transform=tx)
   val_ds   = Dataset(val_items,   transform=tx)


   tl = DataLoader(train_ds, batch_size=batch, shuffle=False,  num_workers=workers)
   vl = DataLoader(val_ds,   batch_size=1,     shuffle=False, num_workers=workers)


   print(f"[data] {len(items)} items | train {len(train_ds)} | val {len(val_ds)}")
   print(f"[data] age_norm bounds: [{age_min:.1f}, {age_max:.1f}] → scaled to [0,1]")
   print(f"[data] vol_norm bounds: [{vol_min:.1f}, {vol_max:.1f}] → scaled to [0,1]")
   return tl, vl





def make_val_loader_from_csv(
    csv_path: str,
    spacing=(1.0, 1.0, 1.0),
    size=(160, 224, 160),
    batch=1,
    workers=4,
    val_frac=0.1,
) -> DataLoader:
    import pandas as pd
    from pathlib import Path as P

    df = pd.read_csv(csv_path)
    # normalize accepted column names
    if "path" in df.columns and "image" not in df.columns:
        df = df.rename(columns={"path": "image"})
    if "seg_path" in df.columns and "seg" not in df.columns:
        df = df.rename(columns={"seg_path": "seg"})

    for col in ("image",):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")

    items: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        it = {"image": str(P(r["image"]).expanduser())}
        items.append(it)

    # split train/val exactly like your training (last chunk = val)
    n = len(items)
    v = max(1, int(round(n * val_frac))) if n > 1 else 1
    val_items = items[max(n - v, 0):] if n > 1 else items

    tx = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Use the same interpolation you trained with; if you trained with nearest, keep it.
        # For MRI intensities, bilinear is usually better. Change if needed to mirror training.
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        SpatialPadd(keys=["image"], spatial_size=size),
        FindNonBGCenter(key="image", bg=None, tol=0.0),
        CenterSpatialCropd(keys=["image"], roi_size=size),
        EnsureTyped(keys=["image"]),
    ])

    val_ds = Dataset(val_items, transform=tx)
    vl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)
    print(f"[data] total {n} | val {len(val_ds)}")
    return vl

# ----------------------------
# AE forward robustly (covers API variants)
# ----------------------------
def ae_forward_recon(ae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    """
    Returns recon [B,1,D,H,W]. Handles (recon, mu, logvar) tuple or direct tensor.
    """
    out = ae(x)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="AE reconstruction inference (val split).")
    ap.add_argument("--csv", required=True, help="CSV with at least an 'image' column (same one used in training).")
    ap.add_argument("--ae_ckpt", required=True, help="Path to AE weights (state_dict). e.g., ckpts/ae_best.pt")
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm, e.g., 1,1,1")
    ap.add_argument("--size", default="160,224,160", help="Canvas D,H,W, e.g., 160,224,160")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--latent_ch", type=int, default=3)
    ap.add_argument("--ae_channels", default="64,128,128,128")
    ap.add_argument("--outdir", default="ckpts/ae_recons")
    ap.add_argument("--max_save", type=int, default=8)
    args = ap.parse_args()

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size    = tuple(int(x)   for x in args.size.split(","))
    ae_channels = tuple(int(x) for x in args.ae_channels.split(","))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # data (val only)
    train_loader, val_loader = make_loaders_from_csv(
        csv_path=args.csv, spacing=spacing, size=size,
        batch=args.batch, workers=args.workers, val_frac=args.val_frac
    )

    # model
    ae = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1,
        latent_channels=args.latent_ch,
        num_channels=ae_channels,
        num_res_blocks=2, norm_num_groups=32, norm_eps=1e-6,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device).eval()

    state = torch.load(args.ae_ckpt, map_location="cpu")
    state = state.get("state_dict", state)
    missing, unexpected = ae.load_state_dict(state, strict=False)
    print(f"[AE] loaded: missing={len(missing)} unexpected={len(unexpected)}")

    # metrics
    l1 = torch.nn.L1Loss(reduction="mean")
    n, l1_sum, psnr_sum, ssim_sum, saved = 0, 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="[AE-RECON]")):
            if i >= 50:
                x = batch["image"].to(device)      # [B,1,D,H,W] in [0,1] (given your pipeline)
                recon = ae_forward_recon(ae, x)    # [B,1,D,H,W]

                # clamp for fair PSNR/SSIM if you trained in [0,1]
                xr = recon.clamp(0, 1)
                xx = x.clamp(0, 1)

                l1_sum   += l1(xr, xx).item()
                psnr_sum += psnr(xr, xx)
                ssim_sum += ssim_3d(xr, xx)
                n += 1

                if saved < args.max_save:
                    save_nii(xx,  outdir / f"case{i:03d}_gt.nii.gz")
                    save_nii(xr,  outdir / f"case{i:03d}_recon.nii.gz")
                    save_mid_slices(xx, f"case{i:03d}_gt", outdir)
                    save_mid_slices(xr, f"case{i:03d}_recon", outdir)
                    saved += 1

    print(f"[AE-RECON] L1: {l1_sum/max(n,1):.4f} | PSNR: {psnr_sum/max(n,1):.2f} dB | SSIM: {ssim_sum/max(n,1):.4f}")
    print(f"[AE-RECON] wrote {min(saved, args.max_save)} example pairs to: {outdir}")

if __name__ == "__main__":
    main()
