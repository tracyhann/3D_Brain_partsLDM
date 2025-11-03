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
import math


from monai.data import Dataset, set_track_meta
from monai.transforms import (
   Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
   SpatialPadd, CenterSpatialCropd, EnsureTyped, MapTransform
)
from monai.transforms import MapTransform, Compose, LoadImaged, EnsureChannelFirstd, Orientationd, SpatialCropd, EnsureTyped
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.losses import PerceptualLoss
from typing import Optional


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
   logfile: str = "",
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

   from torch.profiler import profile, ProfilerActivity, record_function
   profiled = False
   for epoch in range(num_epochs):
       ae.train(); tr = 0.0; tr_rec = 0.0; tr_kl = 0.0
       pbar = tqdm(train_loader, desc=f"[AE] epoch {epoch+1}/{num_epochs}")
       for b in pbar:
            x = b["image"].to(device)  # [B,1,D,H,W] in [0,1]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                recon, mu, logvar = ae(x)
                loss_l1 = l1(recon, x)
                loss_lpips = lpips_weight * lpips(recon, x)
                loss_rec = loss_l1 + loss_lpips
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
       with open(logfile, "a", encoding="utf-8") as f:
           f.write(f"epoch {epoch+1}/{num_epochs} | train batches {len(train_loader)} | train {(tr/len(train_loader)):.4f} | val batches {len(val_loader)} | val {vl:.4f} | ")
       torch.save(ae.state_dict(), os.path.join(outdir, "ae_last.pt"))
       if vl < best:
           best = vl
           torch.save(ae.state_dict(), os.path.join(outdir, "ae_best.pt"))
   return os.path.join(outdir, "ae_best.pt")


# ------------------------
# Robust Stage B: Latent Diffusion (finetune-ready)
# ------------------------


class EMA:
    """Exponential Moving Average for model parameters (optional but useful for diffusion)."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd and sd[k].dtype.is_floating_point:
                sd[k].copy_(v)


def _save_ckpt(path: str, unet: torch.nn.Module, opt, scaler, epoch: int, global_step: int,
               ema: Optional[EMA] = None, extra: Optional[Dict[str, Any]] = None):
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
        return int(sd.get("epoch", 0)), int(sd.get("global_step", 0))
    else:
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        print(f"[LDM] resumed UNet (raw): missing={len(missing)} unexpected={len(unexpected)}")
        return 0, 0


def _maybe_restore_optim(ckpt_path: str, opt, scaler, ema: Optional[EMA] = None) -> Tuple[int, int]:
    """If ckpt is packaged, restore optimizer, scaler, and EMA. Returns (epoch, global_step)."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict) or "state_dict" not in sd:
        return 0, 0
    if opt is not None and sd.get("optimizer") is not None:
        opt.load_state_dict(sd["optimizer"])
    if scaler is not None and sd.get("scaler") is not None:
        scaler.load_state_dict(sd["scaler"])
    if ema is not None and sd.get("ema") is not None:
        ema.shadow = {k: v.to(next(unet.parameters()).device) for k, v in sd["ema"].items()}
    return int(sd.get("epoch", 0)), int(sd.get("global_step", 0))


def train_ldm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    ae_ckpt: str,
    outdir: str = "ckpts",
    num_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    latent_channels: int = 3,
    unet_channels=(256, 512, 768),
    use_cond: bool = False,      # if True, concat conditioning channels and use cross-attn
    resume_unet: str = "",
    logfile: str = "",
    grad_clip: float = 1.0,
    ema_decay: float = 0.0,     # set to e.g. 0.9999 to enable EMA
    sample_every: int = 50,      # every N epochs, decode a tiny sample grid (0=skip)
    save_every: int = 100,       # save checkpoint every N epochs
    best_key: str = "val_loss", # select best by this metric
    cond_dim: int = 4           # *** aligns with your bundle: [gender, age, ventricular_vol, brain_vol]
):
    """
    Trains/finetunes the latent diffusion UNet using a frozen AutoencoderKL.
    - Matches inference bundle: concatenated 4-D conditioning + cross-attention dim=4.
    - Robust: AMP, grad clip, EMA (optional), best+last checkpointing, resume.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # -----------------------
    # Frozen Autoencoder (encode-only path)
    # -----------------------
    use_cond = True
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

    # -----------------------
    # Diffusion UNet (ε-prediction), aligned to inference bundle
    # in_channels = latent_channels + cond_dim (concat)
    # cross_attention_dim = cond_dim (tokens)
    # -----------------------
    in_ch = latent_channels + (cond_dim if use_cond else 0)
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=latent_channels,
        num_channels=tuple(unet_channels),              # e.g., (256, 512, 768)
        num_res_blocks=2,
        attention_levels=(False, True, True),
        norm_num_groups=32, norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=64,                        # <-- was [0,512,768] or 0 somewhere; set to 64
        with_conditioning=use_cond,
        transformer_num_layers=1 if use_cond else 0,
        cross_attention_dim=(cond_dim if use_cond else None),
        upcast_attention=True,
        use_flash_attention=False,
    ).to(device)


    # -----------------------
    # Finetune from existing UNet weights if provided (raw or packaged)
    # -----------------------
    start_epoch, global_step = 0, 0
    if resume_unet and os.path.exists(resume_unet):
        # Load weights into UNet
        e_ep, e_step = _load_ckpt_into_unet(unet, resume_unet)
        start_epoch, global_step = max(start_epoch, e_ep), max(global_step, e_step)

    # -----------------------
    # Optimizer, AMP, EMA
    # -----------------------
    opt = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(unet, decay=ema_decay) if ema_decay and ema_decay < 1.0 else None

    # If resume provided a packaged checkpoint, restore optimizer/scaler/EMA too
    if resume_unet and os.path.exists(resume_unet):
        try:
            r_ep, r_step = _maybe_restore_optim(resume_unet, opt, scaler, ema)
            start_epoch, global_step = max(start_epoch, r_ep), max(global_step, r_step)
        except Exception as e:
            print(f"[LDM] optimizer/EMA restore skipped: {e}")

    mse = torch.nn.MSELoss()

    # -----------------------
    # Schedule: match your inference bundle defaults
    # -----------------------
    sched = DDIMScheduler(
        beta_start=0.0015,
        beta_end=0.0205,
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        clip_sample=False
    )

    os.makedirs(outdir, exist_ok=True)
    log_fp = open(logfile, "a", encoding="utf-8") if logfile else None

    def tile_cond(cond_vec: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # cond_vec: [B,cond_dim] -> [B,cond_dim,d,h,w] for channel concat
        B, _, d, h, w = z.shape
        c = cond_vec[:, :, None, None, None]
        return c.expand(B, cond_dim, d, h, w)

    def add_noise(z0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor) -> torch.Tensor:
        # q(z_t | z_0, t)
        a_cum = sched.alphas_cumprod.to(z0.device)  # [T]
        a_t = a_cum[t].view(-1, 1, 1, 1, 1)         # [B,1,1,1,1]
        return torch.sqrt(a_t) * z0 + torch.sqrt(1.0 - a_t) * noise

    def safe_mean(xs):
        return float(sum(xs) / max(len(xs), 1))

    best_metric = math.inf

    # -----------------------
    # Training loop
    # -----------------------
    print('start epoch', start_epoch)
    for epoch in range(start_epoch, num_epochs): # num_epochs
        unet.train()
        tr_losses = []
        pbar = tqdm(train_loader, desc=f"[LDM] epoch {epoch+1}/{num_epochs}")
        for b in pbar:
            x = b["image"].to(device)  # [B,1,D,H,W]

            # encode → latent
            with torch.no_grad():
                # Compat across MONAI versions:
                if hasattr(ae, "encode_stage_2_inputs"):
                    z = ae.encode_stage_2_inputs(x)        # [B,latent_ch,d,h,w]
                else:
                    # fallback: forward returns recon, mu, logvar; use mu as z
                    _recon, z_mu, _z_logvar = ae(x)
                    z = z_mu

            B = z.size(0)
            t = torch.randint(0, sched.num_train_timesteps, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(z)
            z_t = add_noise(z, t, noise)

            xin = z_t
            context = None
            if use_cond:
                # Accept several possible field names; map into [B,4]: [gender, age, ventricular_vol, brain_vol]
                if "cond" in b:
                    c = b["cond"].to(device).float()
                    c = torch.zeros((B, 4), device=device, dtype=torch.float32)
                    c_img = c[:, :, None, None, None].expand(B, 4, z_t.size(2), z_t.size(3), z_t.size(4))
                    #print('SHAPE', c_img.shape, xin.shape)
                    xin = torch.cat([z_t, c_img], dim=1)
                    #print('SHAPE', xin.shape)# [B,7,d,h,w]
                    # 2) cross-attn context: MUST be [B, L, 4], use L=1
                    context = c.unsqueeze(1)
                else:
                    # heuristic fallback if your loader provides separate fields
                    gender = b.get("gender", None)
                    age = b.get("age", None)
                    vent = b.get("ventricular_vol", b.get("vent", None))
                    brain = b.get("brain_vol", None)
                    fields = [gender, age, vent, brain]
                    if any(v is None for v in fields):
                        raise KeyError("Conditioning missing: expected 'cond' or fields gender/age/ventricular_vol/brain_vol.")
                    c = torch.stack([gender, age, vent, brain], dim=1).to(device).float()

                if c.ndim == 1:  # [4] -> [1,4]
                    c = c.unsqueeze(0)
                if c.shape[1] != cond_dim:
                    raise ValueError(f"cond_dim mismatch: got {c.shape[1]} but expected {cond_dim}")

                #c_img = tile_cond(c, z)      # [B,cond_dim,d,h,w]
                #xin = torch.cat([xin, c_img], dim=1)
                #context = c                  # [B,cond_dim]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # UNet forward; pass timesteps and (optionally) cross-attention context
                if use_cond:
                    eps_hat = unet(xin, t, context=context)
                else:
                    eps_hat = unet(xin, t)
                print(f"[stats] noise mean/std/min/max: {noise.mean().item():.4f} / {noise.std().item():.4f} / {noise.min().item():.4f} / {noise.max().item():.4f}| "
                      f"xin min/max/mean: {xin.min().item():.4f} / {xin.max().item():.4f} / {xin.mean().item():.4f}")
                loss = mse(eps_hat, noise)

            # basic NaN/inf guard
            if not torch.isfinite(loss):
                print("[LDM] WARNING: non-finite loss, skipping step.")
                continue

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            if ema is not None:
                ema.update(unet)

            tr_losses.append(loss.item())
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = safe_mean(tr_losses)
        print(f"[LDM] train {train_loss:.5f}")

        # -----------------------
        # Validation (ε MSE)
        # -----------------------
        unet.eval()
        val_losses = []
        with torch.no_grad():
            for b in val_loader:
                x = b["image"].to(device)
                if hasattr(ae, "encode_stage_2_inputs"):
                    z = ae.encode_stage_2_inputs(x)
                else:
                    _recon, z_mu, _z_logvar = ae(x)
                    z = z_mu
                B = z.size(0)
                t = torch.randint(0, sched.num_train_timesteps, (B,), device=device, dtype=torch.long)
                noise = torch.randn_like(z)
                z_t = add_noise(z, t, noise)
                xin = z_t
                context = None
                if use_cond:
                    c = torch.zeros((B, 4), device=device, dtype=torch.float32)
                    c_img = c[:, :, None, None, None].expand(B, 4, z_t.size(2), z_t.size(3), z_t.size(4))
                    #print('SHAPE', c_img.shape, xin.shape)
                    xin = torch.cat([z_t, c_img], dim=1)
                    #print('SHAPE', xin.shape)# [B,7,d,h,w]
                    # 2) cross-attn context: MUST be [B, L, 4], use L=1
                    context = c.unsqueeze(1)
                eps_hat = unet(xin, t, context=context) if use_cond else unet(xin, t)
                val_losses.append(mse(eps_hat, noise).item())

        val_loss = safe_mean(val_losses)
        print(f"[LDM] val  {val_loss:.5f}")

        # -----------------------
        # Logging
        # -----------------------
        if log_fp:
            log_fp.write(f"epoch {epoch+1}/{num_epochs} | step {global_step} | train {train_loss:.6f} | val {val_loss:.6f}\n")
            log_fp.flush()

        # -----------------------
        # Checkpointing
        # -----------------------
        # Save "last"
        _save_ckpt(os.path.join(outdir, "ldm_last.pt"), unet, opt, scaler, epoch+1, global_step, ema,
                   extra={"train_loss": train_loss, "val_loss": val_loss})

        # Save "best" on chosen metric (lower is better)
        metric = val_loss if best_key == "val_loss" else train_loss
        if metric < best_metric:
            best_metric = metric
            # if EMA present, store an EMA-averaged copy for best
            if ema is not None:
                # copy weights -> temp, save, then restore
                tmp = DiffusionModelUNet(**{**unet._get_name_kwargs()}) if hasattr(unet, "_get_name_kwargs") else None
                if tmp is None:
                    # safer: apply EMA to a copy of state_dict
                    _save_ckpt(os.path.join(outdir, "ldm_best.pt"), unet, opt, scaler, epoch+1, global_step, ema,
                               extra={"train_loss": train_loss, "val_loss": val_loss})
                else:
                    tmp.load_state_dict(unet.state_dict())
                    ema.copy_to(unet)
                    _save_ckpt(os.path.join(outdir, "ldm_best.pt"), unet, opt, scaler, epoch+1, global_step, ema,
                               extra={"train_loss": train_loss, "val_loss": val_loss})
                    unet.load_state_dict(tmp.state_dict())
            else:
                _save_ckpt(os.path.join(outdir, "ldm_best.pt"), unet, opt, scaler, epoch+1, global_step, ema,
                           extra={"train_loss": train_loss, "val_loss": val_loss})
                
        
        # Optional sampling preview (quick sanity check)
        print('sample every ', sample_every, ' epochs.')
        if sample_every > 0 and (epoch + 1) % sample_every == 0:
            try:
                unet.eval()
                if ema is not None:
                    # temporarily apply EMA weights
                    ema.copy_to(unet)
                # take first val batch; denoise a single latent to z0 and decode to NIfTI
                for b in val_loader:
                    x = b["image"].to(device)
                    if hasattr(ae, "encode_stage_2_inputs"):
                        z = ae.encode_stage_2_inputs(x)
                    else:
                        _recon, z_mu, _z_logvar = ae(x)
                        z = z_mu
                    B = min(z.shape[0], 1)
                    z = z[:B]
                    if use_cond:
                        c = torch.zeros((B, 4), device=device, dtype=torch.float32)
                    # DDIM sampling (few steps) to visualize
                    sampler = DDIMScheduler(
                        beta_start=0.0015, beta_end=0.0205, num_train_timesteps=1000,
                        schedule="scaled_linear_beta", clip_sample=False
                    )
                    sampler.set_timesteps(num_inference_steps=25)
                    z_t = torch.randn_like(z)
                    for t_i in sampler.timesteps:
                        xin = z_t
                        if use_cond:
                            c = torch.zeros((B, 4), device=device, dtype=torch.float32)
                            c_img = c[:, :, None, None, None].expand(B, 4, z_t.size(2), z_t.size(3), z_t.size(4))
                            xin = torch.cat([z_t, c_img], dim=1)
                            context = c.unsqueeze(1)
                            eps_hat = unet(xin, t, context=context) if use_cond else unet(xin, t)
                        else:
                            eps_hat = unet(xin, t_i)
                        z_t = sampler.step(eps_hat, t_i, z_t)[0]
                    # decode
                    with torch.no_grad():
                        x_hat = ae.decode_stage_2_outputs(z_t) if hasattr(ae, "decode_stage_2_outputs") else ae.decode(z_t)
                        arr = x_hat.squeeze(0).squeeze(0).detach().cpu().numpy()  # [D,H,W]
                        nib.save(nib.Nifti1Image(arr, np.eye(4)), os.path.join(outdir, f"preview_ep{epoch+1}_t{t}.nii.gz"))
                        print(f'preview_ep{epoch+1}.pt saved to {os.path.join(outdir, f"preview_ep{epoch+1}_t{t}.nii.gz")}')
                    break
            except Exception as e:
                print(f"[LDM] preview sampling failed: {e}")


        # Periodic saves
        if save_every and (epoch + 1) % save_every == 0:
            _save_ckpt(os.path.join(outdir, f"ldm_ep{epoch+1}.pt"), unet, opt, scaler, epoch+1, global_step, ema,
                       extra={"train_loss": train_loss, "val_loss": val_loss})
    if log_fp:
        log_fp.close()

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
   ap.add_argument("--ldm_use_cond", default="False", help="Use [part_vol_norm,sex,age] conditioning")
   ap.add_argument("--ldm_channels", default="128,256,512")
   ap.add_argument("--ldm_resume", default="", help="Resume UNet weights (optional)")


   ap.add_argument("--outdir", default="ckpts")


   args = ap.parse_args()

   from datetime import datetime
   import json
   experiment_dir = os.path.join(args.outdir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
   os.makedirs(experiment_dir, exist_ok=True)
   args.outdir = experiment_dir

   # argparse Namespace -> dict
   cfg = vars(args).copy()

   # Make non-JSON types serializable (e.g., tuples)
   for k,v in list(cfg.items()):
       if isinstance(v, tuple): cfg[k] = list(v)
   with open(os.path.join(args.outdir, 'args.json'), "w", encoding="utf-8") as f:
       json.dump(cfg, f, indent=2, sort_keys=True)

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
           logfile=os.path.join(args.outdir, "ae_training_log.txt")
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
           logfile=os.path.join(args.outdir, "ldm_training_log.txt")
       )




if __name__ == "__main__":
   main()


