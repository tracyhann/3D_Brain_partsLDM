#!/usr/bin/env python3
# ldm_infer.py — 3D LDM inference (DDIM) with robust IO, stats, and latent scaling

import argparse, json
from pathlib import Path
from typing import Optional
import numpy as np
import nibabel as nib
import imageio
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
import os

# ----------------------------
# IO helpers
# ----------------------------
def save_nii(vol: torch.Tensor, path: Path):
    arr = vol.squeeze(0).squeeze(0).detach().cpu().numpy()  # [D,H,W]
    #arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))

def save_mid_slices(vol: torch.Tensor, stem: str, outdir: Path):
    v = vol.squeeze(0).squeeze(0).detach().cpu().numpy()
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
    zc, yc, xc = [d // 2 for d in v.shape]

    def _norm(sl):
        sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(np.min(sl)), float(np.max(sl))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(sl, dtype=np.uint8)
        sl = (sl - lo) / (hi - lo)
        return (np.clip(sl, 0, 1) * 255.0).astype(np.uint8)

    for name, sl in [("ax", v[zc, :, :]), ("cor", v[:, yc, :]), ("sag", v[:, :, xc])]:
        imageio.imwrite(str(outdir / f"{stem}_{name}.png"), _norm(sl))

# ----------------------------
# AE encode/decode (version-robust)
# ----------------------------
@torch.no_grad()
def ae_encode_latents(ae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    out = ae.encode(x) if hasattr(ae, "encode") else None
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "sample"):
        return out.sample()
    if isinstance(out, (tuple, list)):
        if len(out) >= 1 and isinstance(out[0], torch.Tensor) and out[0].dim() == 5:
            return out[0]
        if len(out) >= 2 and all(isinstance(t, torch.Tensor) for t in out[:2]):
            mu, logvar = out[0], out[1]
            std = (0.5 * logvar).exp()
            return mu + torch.randn_like(std) * std
    if hasattr(ae, "encode_stage_2_inputs"):
        return ae.encode_stage_2_inputs(x)
    raise RuntimeError("Unrecognized AutoencoderKL.encode() return type")

@torch.no_grad()
def ae_decode_latents(ae: AutoencoderKL, z: torch.Tensor) -> torch.Tensor:
    try:
        return ae.decode(z)
    except Exception:
        return ae.decode_stage_2_outputs(z)

# ----------------------------
# Conditioning utils
# ----------------------------
def parse_cond(s: Optional[str]) -> Optional[torch.Tensor]:
    if not s:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError("--cond must be 'vol_norm,sex,age_norm' (3 numbers)")
    return torch.tensor(vals, dtype=torch.float32)[None, :]  # [1,3]

def tile_cond(cond_vec: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # cond_vec [B,3] -> [B,3,d,h,w]
    B, _, d, h, w = z.shape
    c = cond_vec[:, :, None, None, None]
    return c.expand(B, 4, d, h, w)

# ----------------------------
# Timesteps / DDIM helpers
# ----------------------------
def batch_timesteps(t, B, device):
    if not torch.is_tensor(t):
        t = torch.tensor(int(t), dtype=torch.long, device=device)
    t = t.to(device).long()
    if t.dim() == 0:
        t = t.expand(B)  # [B]
    return t

def step_prev_sample(step_out):
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, tuple):
        return step_out[0]
    if isinstance(step_out, dict) and "prev_sample" in step_out:
        return step_out["prev_sample"]
    raise RuntimeError(f"Unknown DDIM step return type: {type(step_out)}")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="LDM 3D inference (DDIM).")
    # Weights + shapes (match training!)
    ap.add_argument("--ae_ckpt",   required=True)
    ap.add_argument("--unet_ckpt", required=True)
    ap.add_argument("--latent_ch", type=int, default=3)
    ap.add_argument("--ae_channels",   default="64,128,128,128")
    ap.add_argument("--unet_channels", default="128,256,512")
    ap.add_argument("--size", default="160,224,160", help="Training crop D,H,W")

    # Sampling
    ap.add_argument("--steps", type=int, default=250, help="DDIM steps")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0=deterministic)")
    ap.add_argument("--num", type=int, default=4, help="#samples to generate")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    ap.add_argument("--outdir", default="samples_ldm")

    # Latent noise scaling (helps if AE latents not ~N(0,1) due to tiny KL)
    ap.add_argument("--z_scale", type=float, default=1.0, help="Scale of initial latent noise")

    # Conditioning + CFG
    ap.add_argument("--use_cond", action="store_true", help="UNet in_ch = latent_ch + 3")
    ap.add_argument("--cond", default="0.0,0.0,0.0,0.0", help="vol_norm,sex,age_norm in [0,1], e.g. 0.4,1,0.35")
    ap.add_argument("--cfg_scale", type=float, default=1.0, help=">1 enables classifier-free guidance")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    size = tuple(int(x) for x in args.size.split(","))
    ae_channels = tuple(int(x) for x in args.ae_channels.split(","))
    unet_channels = tuple(int(x) for x in args.unet_channels.split(","))

    # --- Build AE & UNet (match training) ---
    ae = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1,
        latent_channels=args.latent_ch, num_channels=ae_channels,
        num_res_blocks=2, norm_num_groups=32, norm_eps=1e-6,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    ).to(device).eval()
    state = torch.load(args.ae_ckpt, map_location="cpu")
    state = state.get("state_dict", state)
    ae.load_state_dict(state, strict=False)

    cond_dim = 4 if args.use_cond else 0
    in_ch = args.latent_ch + cond_dim
    attn_lvls = (False, True, True) if len(unet_channels) == 3 else (False, True, True, True)

    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=in_ch, out_channels=args.latent_ch,
        num_res_blocks=2, num_channels=unet_channels,
        attention_levels=attn_lvls, num_head_channels=64,
    ).to(device).eval()
    state = torch.load(args.unet_ckpt, map_location="cpu")
    state = state.get("state_dict", state)
    unet.load_state_dict(state, strict=False)

    # --- Latent shape probe ---
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *size, device=device)
        z_shape = ae_encode_latents(ae, dummy).shape  # [1,C,zd,zh,zw]
    _, C, zd, zh, zw = z_shape
    if C != args.latent_ch:
        print(f"[warn] latent_ch mismatch: AE gives C={C}, but --latent_ch={args.latent_ch}")
    print(f"[shape] latent: {tuple(z_shape)} for image size {size}")

    # --- Scheduler ---
    #sched = DDIMScheduler(num_train_timesteps=1000, schedule="cosine", clip_sample=True)
    sched = DDIMScheduler(
        beta_start=0.0015,
        beta_end=0.0205,
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        clip_sample=False
    )
    sched.set_timesteps(args.steps)

    # --- Conditioning ---
    cond = parse_cond(args.cond) if args.use_cond else None
    if args.use_cond and cond is None:
        raise ValueError("Conditioning enabled but --cond is empty. Provide 'vol_norm,sex,age_norm'.")
    if cond is not None:
        cond = cond.to(device)

    # --- Sampling ---
    torch.manual_seed(args.seed)
    for i in range(args.num):
        g = torch.Generator(device=device).manual_seed(args.seed + i)
        z = args.z_scale * torch.randn(1, C, zd, zh, zw, device=device, generator=g)
        z = torch.randn((1, 3, 20, 28, 20)).to(device)

        with torch.no_grad():
            for t in sched.timesteps:
                t_b = batch_timesteps(t, z.shape[0], device)

                if args.use_cond:
                    c_lat = tile_cond(cond, z)  # [B,3,zd,zh,zw]
                    if args.cfg_scale > 1.0:
                        z_in_u = torch.cat([z, torch.zeros_like(c_lat)], dim=1)
                        z_in_c = torch.cat([z, c_lat], dim=1)
                        eps_u = unet(z_in_u, t_b)
                        eps_c = unet(z_in_c, t_b)
                        eps   = eps_u + args.cfg_scale * (eps_c - eps_u)
                    else:
                        z_in = torch.cat([z, c_lat], dim=1)
                        eps  = unet(z_in, t_b)
                else:
                    eps = unet(z, t_b)

                if not torch.isfinite(eps).all():
                    print(f"[warn] non-finite eps at t={int(t)}; clamping")
                    eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)

                step_out = sched.step(model_output=eps, timestep=t, sample=z, eta=args.eta)
                
                z_step = step_prev_sample(step_out)
                if z_step.mean().item() > 0.0:
                    z = z_step

                

            x = ae_decode_latents(ae, z)
            #x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

        # Stats to diagnose black outputs
        print(f"[stats] z mean/std/min/max: {z.mean().item():.4f} / {z.std().item():.4f} / {z.min().item():.4f} / {z.max().item():.4f}| "
              f"x min/max/mean: {x.min().item():.4f} / {x.max().item():.4f} / {x.mean().item():.4f}")

        stem = f"sample_{i:03d}"
        os.makedirs(f'{outdir}/step{args.steps}', exist_ok = True)
        save_nii(x, outdir / f'step{args.steps}' / f"{stem}.nii.gz")
        save_mid_slices(x, stem, outdir)
        print(f"[save] {stem} -> {outdir}")

    (outdir / "manifest.json").write_text(json.dumps({
        "ae_ckpt": str(args.ae_ckpt),
        "unet_ckpt": str(args.unet_ckpt),
        "size": size,
        "latent_ch": args.latent_ch,
        "ae_channels": ae_channels,
        "unet_channels": unet_channels,
        "steps": args.steps,
        "eta": args.eta,
        "num": args.num,
        "seed": args.seed,
        "use_cond": args.use_cond,
        "cond": args.cond,
        "cfg_scale": args.cfg_scale,
        "z_scale": args.z_scale,
    }, indent=2))
    print(f"[done] wrote manifest.json to {outdir}")

if __name__ == "__main__":
    main()