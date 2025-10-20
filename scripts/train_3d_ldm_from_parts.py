#!/usr/bin/env python3
# train_3d_ldm_from_csv.py
# End-to-end training for MONAI 3D LDM (AE -> LDM) from a CSV listing file paths.


import os, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from monai.data import Dataset, set_track_meta
from monai.transforms import (
   Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
   SpatialPadd, CenterSpatialCropd, EnsureTyped, MapTransform
)
from monai.transforms import MapTransform, Compose, LoadImaged, EnsureChannelFirstd, Orientationd, SpatialCropd, EnsureTyped
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.losses import PerceptualLoss


set_track_meta(False)




# ----------------------------
# CSV → items + MONAI pipeline
# ----------------------------
def _sex01(x) -> int:
   s = str(x).strip().lower()
   if s in ("0", "f", "female"): return 0
   if s in ("1", "m", "male"):   return 1
   raise ValueError(f"Unrecognized sex: {x}")
  


class FindNonBGCenter(MapTransform):
   def __init__(self, key="image", bg=None, tol=0.0):
       super().__init__(keys=(key,))
       self.key, self.bg, self.tol = key, bg, tol
   def __call__(self, d):
       a = d[self.key][0].cpu().numpy()
       bg = self.bg
       if bg is None:
           u,c = np.unique(a, return_counts=True)
           bg = float(u[c.argmax()])
       mask = np.abs(a - bg) > self.tol
       idx = np.argwhere(mask)
       if idx.size == 0:
           raise RuntimeError("No foreground found.")
       d["roi_center"] = idx.mean(axis=0).astype(int)  # (z,y,x)
       return d




class BuildCondVec(MapTransform):
   """
   Adds 'cond' = [part_vol_norm, sex, age_norm]
   - If 'seg' exists, computes part_vol_norm using 'target_label' (if provided) else seg>0 (brain).
   - If no 'seg', sets part_vol_norm=0.0 (still returns a 3-dim cond vector).
   """
   def __init__(self, age_min: float, age_max: float, vol_min: float, vol_max: float):
       super().__init__(keys=("image",))
       self.age_min = float(age_min)
       self.age_max = float(age_max)
       self.age_den = max(self.age_max - self.age_min, 1e-6)
       self.pv_min = float(vol_min)
       self.pv_max = float(vol_max)
       self.pv_den = max(self.pv_max - self.pv_min, 1e-6)


   def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
       # Compute part_vol_norm
       # sex, age_norm
       sex = float(d["sex"])
       age = float(d["age"])
       pv = float(d['vol'])
       age_norm = float(np.clip((age - self.age_min) / self.age_den, 0.0, 1.0))
       pv_norm = float(np.clip((pv - self.pv_min) / self.pv_den, 0.0, 1.0))
       d["cond"] = torch.tensor([pv_norm, sex, age_norm], dtype=torch.float32)
       return d




def make_loaders_from_csv(
   csv_path: str,
   spacing=(1.0, 1.0, 1.0),
   size=(160, 224, 160),
   batch=1,
   workers=4,
   val_frac=0.1,
) -> Tuple[DataLoader, DataLoader]:
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
           "sex": _sex01(r["sex"]),
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
       BuildCondVec(age_min=age_min, age_max=age_max, vol_min=vol_min, vol_max=vol_max)
   ])




   # Split train/val (last val_frac for val)
   n = len(items)
   v = max(1, int(round(n * val_frac))) if n > 1 else 1
   train_items = items[:max(n - v, 1)]
   val_items   = items[max(n - v, 0):] if n > 1 else items


   train_ds = Dataset(train_items, transform=tx)
   val_ds   = Dataset(val_items,   transform=tx)


   tl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers)
   vl = DataLoader(val_ds,   batch_size=1,     shuffle=False, num_workers=workers)


   print(f"[data] {len(items)} items | train {len(train_ds)} | val {len(val_ds)}")
   print(f"[data] age_norm bounds: [{age_min:.1f}, {age_max:.1f}] → scaled to [0,1]")
   print(f"[data] vol_norm bounds: [{vol_min:.1f}, {vol_max:.1f}] → scaled to [0,1]")
   return tl, vl




# ------------------------
# Stage A: AutoencoderKL
# ------------------------
def train_autoencoder(
   train_loader: DataLoader,
   val_loader: DataLoader,
   outdir: str = "ckpts",
   num_epochs: int = 50,
   lr: float = 2e-4,
   weight_decay: float = 1e-5,
   latent_channels: int = 3,
   num_channels=(64, 128, 128, 128),
   kl_weight: float = 1e-6,
   lpips_weight: float = 0.1,
   finetune_ckpt: str = "",
   finetune_decoder_only: bool = False,
):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ae = AutoencoderKL(
       spatial_dims=3,
       in_channels=1, out_channels=1,
       latent_channels=latent_channels,
       num_channels=tuple(num_channels),
       num_res_blocks=2, norm_num_groups=32, norm_eps=1e-06,
       attention_levels=(False, False, False, False),
       with_encoder_nonlocal_attn=False,
       with_decoder_nonlocal_attn=False,
   ).to(device)


   # Load pretrained if provided
   if finetune_ckpt:
       state = torch.load(finetune_ckpt, map_location="cpu")
       state = state.get("state_dict", state)
       missing, unexpected = ae.load_state_dict(state, strict=False)
       print(f"[AE] loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}")


   # Freeze if decoder-only FT
   if finetune_decoder_only:
       for p in ae.parameters(): p.requires_grad = False
       for p in ae.decoder.parameters(): p.requires_grad = True


   opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, ae.parameters()),
                           lr=lr, weight_decay=weight_decay)
   l1 = torch.nn.L1Loss()
   lpips = PerceptualLoss(spatial_dims=3, network_type="alex").to(device)  # optional but strong


   os.makedirs(outdir, exist_ok=True)
   best = 1e9
   scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


   for epoch in range(num_epochs):
       ae.train(); tr = 0.0
       pbar = tqdm(train_loader, desc=f"[AE] epoch {epoch+1}/{num_epochs}")
       for b in pbar:
           x = b["image"].to(device)  # [B,1,D,H,W] in [0,1]


           with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
               recon, mu, logvar = ae(x)
               loss_rec = l1(recon, x) + lpips_weight * lpips(recon, x)
               loss_kl = kl_weight * torch.mean(mu**2 + torch.exp(logvar) - logvar - 1)
               loss = loss_rec + loss_kl


           opt.zero_grad(set_to_none=True)
           scaler.scale(loss).backward()
           scaler.step(opt)
           scaler.update()


           tr += loss.item()
           pbar.set_postfix(loss=f"{loss.item():.4f}")


       # quick val (L1)
       ae.eval(); vl = 0.0; n=0
       with torch.no_grad():
           for b in val_loader:
               x = b["image"].to(device)
               recon, _, _ = ae(x)
               vl += l1(recon, x).item(); n += 1
       vl /= max(n, 1)


       print(f"[AE] train {(tr/len(train_loader)):.4f} | val {vl:.4f}")
       torch.save(ae.state_dict(), os.path.join(outdir, "ae_last.pt"))
       if vl < best:
           best = vl
           torch.save(ae.state_dict(), os.path.join(outdir, "ae_best.pt"))
   return os.path.join(outdir, "ae_best.pt")




# ------------------------
# Stage B: Latent Diffusion
# ------------------------
def train_ldm(
   train_loader: DataLoader,
   val_loader: DataLoader,
   ae_ckpt: str,
   outdir: str = "ckpts",
   num_epochs: int = 100,
   lr: float = 1e-4,
   weight_decay: float = 1e-5,
   latent_channels: int = 3,
   unet_channels=(128, 256, 512),
   use_cond: bool = True,     # if True, concat [pv, sex, age] (tiled) to the latent channels
   resume_unet: str = "",
):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


   # Frozen Autoencoder (encode/decode)
   ae = AutoencoderKL(
       spatial_dims=3,
       in_channels=1, out_channels=1,
       latent_channels=latent_channels,
       num_channels=tuple((64, 128, 128, 128)),
       num_res_blocks=2, norm_num_groups=32, norm_eps=1e-06,
       attention_levels=(False, False, False, False),
       with_encoder_nonlocal_attn=False,
       with_decoder_nonlocal_attn=False,
   ).to(device) .eval()
   ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
   for p in ae.parameters(): p.requires_grad = False


   cond_dim = 3 if use_cond else 0
   in_ch = latent_channels + cond_dim


   # Latent Diffusion UNet (epsilon prediction)


   unet = DiffusionModelUNet(
       spatial_dims=3,
       in_channels=in_ch,
       out_channels=latent_channels,
       num_res_blocks=2,
       num_channels=tuple(unet_channels),
       attention_levels=(False, True, True),
       num_head_channels=64,
   ).to(device)


   if resume_unet:
       state = torch.load(resume_unet, map_location="cpu")
       state = state.get("state_dict", state)
       missing, unexpected = unet.load_state_dict(state, strict=False)
       print(f"[LDM] resumed: missing={len(missing)} unexpected={len(unexpected)}")


   opt = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay)
   mse = torch.nn.MSELoss()


   # Scheduler
   sched = DDIMScheduler(num_train_timesteps=1000, schedule="cosine", clip_sample=True)


   os.makedirs(outdir, exist_ok=True)


   def tile_cond(cond_vec: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
       # cond_vec: [B,3] → [B,3,d,h,w] to concat with latents
       B, C, d, h, w = z.shape
       c = cond_vec[:, :, None, None, None]
       return c.expand(B, C, d, h, w)


   for epoch in range(num_epochs):
       unet.train(); tr = 0.0
       pbar = tqdm(train_loader, desc=f"[LDM] epoch {epoch+1}/{num_epochs}")
       for b in pbar:
           x = b["image"].to(device)           # [B,1,D,H,W]
           with torch.no_grad():
               z = ae.encode_stage_2_inputs(x)  # [B,3,d,h,w]


           B = z.size(0)
           t = torch.randint(0, sched.num_train_timesteps, (B,), device=device, dtype=torch.long)
           noise = torch.randn_like(z)
           z_t = sched.add_noise(z, noise, t)   # q_sample


           if use_cond:
               cond = b["cond"].to(device).float()  # [B,3] = [pv,sex,age]
               cond_lat = tile_cond(cond, z)
               z_in = torch.cat([z_t, cond_lat], dim=1)
           else:
               z_in = z_t


           eps_hat = unet(z_in, t)
           loss = mse(eps_hat, noise)


           opt.zero_grad(set_to_none=True)
           loss.backward()
           opt.step()


           tr += loss.item()
           pbar.set_postfix(loss=f"{loss.item():.4f}")


       print(f"[LDM] train {(tr/len(train_loader)):.4f}")
       torch.save(unet.state_dict(), os.path.join(outdir, "ldm_last.pt"))
       if (epoch + 1) % 10 == 0:
           torch.save(unet.state_dict(), os.path.join(outdir, f"ldm_ep{epoch+1}.pt"))


   return os.path.join(outdir, "ldm_last.pt")




# ------------
# CLI wrapper
# ------------
def main():
   ap = argparse.ArgumentParser(description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.")
   ap.add_argument("--csv", required=True, help="Path to CSV with columns: image[/path], sex, age, [seg/seg_path], [target_label]")
   ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
   ap.add_argument("--size", default="160,224,160", help="Canvas D,H,W (e.g., 160,224,160)")
   ap.add_argument("--batch", type=int, default=1)
   ap.add_argument("--workers", type=int, default=4)
   ap.add_argument("--val_frac", type=float, default=0.1)


   ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="both")


   # AE config
   ap.add_argument("--ae_epochs", type=int, default=50)
   ap.add_argument("--ae_lr", type=float, default=2e-4)
   ap.add_argument("--ae_latent_ch", type=int, default=3)
   ap.add_argument("--ae_kl", type=float, default=1e-6)
   ap.add_argument("--ae_lpips_w", type=float, default=0.1)
   ap.add_argument("--ae_channels", default="64,128,128,128")
   ap.add_argument("--ae_factors", default="1,2,2,2")
   ap.add_argument("--ae_finetune_ckpt", default="", help="Path to pretrained AE .pt (optional)")
   ap.add_argument("--ae_decoder_only", action="store_true", help="Fine-tune decoder only")


   # LDM config
   ap.add_argument("--ldm_epochs", type=int, default=100)
   ap.add_argument("--ldm_lr", type=float, default=1e-4)
   ap.add_argument("--ldm_use_cond", action="store_true", help="Use [part_vol_norm,sex,age] conditioning")
   ap.add_argument("--ldm_channels", default="128,256,512")
   ap.add_argument("--ldm_resume", default="", help="Resume UNet weights (optional)")


   ap.add_argument("--outdir", default="ckpts")


   args = ap.parse_args()


   spacing = tuple(float(x) for x in args.spacing.split(","))
   size    = tuple(int(x)   for x in args.size.split(","))
   ae_channels = tuple(int(x) for x in args.ae_channels.split(","))
   ae_factors  = tuple(int(x) for x in args.ae_factors.split(","))
   ldm_channels = tuple(int(x) for x in args.ldm_channels.split(","))


   # Build loaders
   train_loader, val_loader = make_loaders_from_csv(
       csv_path=args.csv,
       spacing=spacing, size=size,
       batch=args.batch, workers=args.workers, val_frac=args.val_frac
   )


   # Stage(s)
   ae_ckpt_path = args.ae_finetune_ckpt
   if args.stage in ("ae", "both"):
       ae_ckpt_path = train_autoencoder(
           train_loader, val_loader,
           outdir=args.outdir, num_epochs=args.ae_epochs, lr=args.ae_lr,
           latent_channels=args.ae_latent_ch, num_channels=ae_channels,
           kl_weight=args.ae_kl, lpips_weight=args.ae_lpips_w,
           finetune_ckpt=args.ae_finetune_ckpt, finetune_decoder_only=args.ae_decoder_only,
       )


   if args.stage in ("ldm", "both"):
       if not ae_ckpt_path or not Path(ae_ckpt_path).exists():
           raise FileNotFoundError("Autoencoder checkpoint not found. Provide --ae_finetune_ckpt or run --stage ae/both.")
       train_ldm(
           train_loader, val_loader,
           ae_ckpt=ae_ckpt_path,
           outdir=args.outdir, num_epochs=args.ldm_epochs, lr=args.ldm_lr,
           latent_channels=args.ae_latent_ch, unet_channels=ldm_channels,
           use_cond=args.ldm_use_cond, resume_unet=args.ldm_resume,
       )




if __name__ == "__main__":
   main()


