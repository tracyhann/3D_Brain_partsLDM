from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai import transforms
from torch.amp import autocast
from tqdm import tqdm

from generative.networks.schedulers import DDIMScheduler

# Reuse architecture/checkpoint/data-loader utilities from the training script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_3d_brain_ldm_steps as base  # noqa: E402


def _resolve_ckpt_path(path: str, default_name: str = "UNET_last.pt") -> str:
    if os.path.isdir(path):
        resolved = os.path.join(path, default_name)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Checkpoint directory provided, but {default_name} not found: {path}"
            )
        return resolved
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _safe_component(x: Any) -> str:
    s = str(x)
    s = re.sub(r"[^\w.\-]+", "-", s)
    s = s.strip("-")
    return s if s else "unknown"


def _extract_image_id(batch: dict, fallback: str) -> str:
    if "imageID" not in batch:
        return fallback
    v = batch["imageID"]
    if isinstance(v, (list, tuple)):
        return str(v[0]) if len(v) > 0 else fallback
    if torch.is_tensor(v):
        if v.numel() == 0:
            return fallback
        if v.ndim == 0:
            return str(v.item())
        return str(v.reshape(-1)[0].item())
    return str(v)


def _extract_affine(batch: dict) -> np.ndarray:
    affine = np.eye(4, dtype=np.float32)
    img = batch.get("image", None)
    if hasattr(img, "affine"):
        aff = img.affine
        if isinstance(aff, torch.Tensor):
            aff = aff.detach().cpu().numpy()
        aff = np.asarray(aff)
        if aff.ndim == 3:
            aff = aff[0]
        if aff.shape == (4, 4):
            return aff.astype(np.float32)
    meta = batch.get("image_meta_dict", None)
    if isinstance(meta, dict) and "affine" in meta:
        aff = meta["affine"]
        if isinstance(aff, torch.Tensor):
            aff = aff.detach().cpu().numpy()
        aff = np.asarray(aff)
        if aff.ndim == 3:
            aff = aff[0]
        if aff.shape == (4, 4):
            return aff.astype(np.float32)
    return affine


def _build_condition_tensor(batch: dict, cond_keys: list[str], device: torch.device) -> Optional[torch.Tensor]:
    vals = []
    for k in cond_keys:
        if k not in batch:
            return None
        v = batch[k]
        if torch.is_tensor(v):
            t = v
        else:
            t = torch.as_tensor(v)
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.reshape(t.shape[0], 1)
        else:
            t = t.reshape(t.shape[0], -1)
        vals.append(t)
    if not vals:
        return None
    return torch.cat(vals, dim=1).unsqueeze(1)


@torch.no_grad()
def _ddim_sample_one(
    *,
    unet: torch.nn.Module,
    autoencoder: torch.nn.Module,
    scheduler: DDIMScheduler,
    latent_shape: tuple[int, ...],
    scale_factor: float,
    device: torch.device,
    amp_enabled: bool,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    z = torch.randn((1,) + latent_shape, device=device)
    for t in scheduler.timesteps:
        t_int = int(t.item()) if hasattr(t, "item") else int(t)
        t_batch = torch.full((1,), t_int, device=device, dtype=torch.long)
        with autocast(device_type="cuda", enabled=amp_enabled):
            if condition is None:
                eps = unet(z, timesteps=t_batch)
            else:
                eps = unet(z, timesteps=t_batch, context=condition)
        out = scheduler.step(model_output=eps, timestep=t_int, sample=z)
        if hasattr(out, "prev_sample"):
            z = out.prev_sample
        elif isinstance(out, (tuple, list)):
            z = out[0]
        else:
            z = out
    with autocast(device_type="cuda", enabled=amp_enabled):
        x_gen = autoencoder.decode_stage_2_outputs(z / scale_factor)
    return x_gen


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config with argument defaults.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Inference for step-based MONAI 3D LDM using test split matching.",
        parents=[pre_ap],
    )
    ap.add_argument("--csv", default="", help="Path to CSV with columns: image, imageID, and optional condition columns.")
    ap.add_argument(
        "--data_split_json_path",
        default="data/patient_splits_image_ids_75_10_15.json",
        help="JSON file with train/val/test imageID splits.",
    )
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1, help="Training loader batch used by shared loader util.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True, help="Use autocast on CUDA.")
    ap.add_argument("--n_samples", default="ALL", help="Passed to shared loader; keep ALL for full split consistency.")
    ap.add_argument("--seed", type=int, default=1017)

    ap.add_argument("--ae_latent_ch", type=int, default=8)
    ap.add_argument("--ae_num_channels", default="64,128,256")
    ap.add_argument("--ae_attention_levels", default="0,0,0")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE checkpoint (.pt).")

    ap.add_argument("--ldm_num_channels", default="256,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_use_cond", default="False", help="Set True only for with_conditioning UNet checkpoints.")
    ap.add_argument("--cond_dim", type=int, default=64, help="Cross-attention conditioning dim when ldm_use_cond=True.")
    ap.add_argument("--cond_keys", default="vol,sex,age", help="Condition keys from loader batch when ldm_use_cond=True.")
    ap.add_argument("--ldm_ckpt", default="", help="Path to UNet checkpoint (.pt) or experiment dir containing UNET_last.pt.")
    ap.add_argument("--scale_factor", type=float, default=-1.0, help="Override latent scale factor. Use <=0 to auto-load/estimate.")

    ap.add_argument("--ddim_steps", type=int, default=50, help="DDIM denoising steps.")
    ap.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of volumes to generate. Default (-1) uses the full test loader length.",
    )
    ap.add_argument(
        "--conditions",
        default="age,sex,vol,group",
        help="CSV columns to include in the shared loader (imageID is auto-added for naming).",
    )

    ap.add_argument("--outdir", default="samples")
    ap.add_argument("--out_prefix", default="ldm_steps_infer")
    ap.add_argument("--out_postfix", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            cfg_defaults = json.load(f)
        if not isinstance(cfg_defaults, dict):
            raise ValueError(f"Config must be a JSON object: {pre_args.config}")
        known_keys = {a.dest for a in ap._actions}
        unknown_keys = sorted(set(cfg_defaults.keys()) - known_keys)
        if unknown_keys:
            raise ValueError(f"Unknown keys in config {pre_args.config}: {unknown_keys}")
        ap.set_defaults(**cfg_defaults)

    args = ap.parse_args()
    if not args.csv:
        ap.error("--csv is required (pass on CLI or in --config).")
    if not args.ae_ckpt:
        ap.error("--ae_ckpt is required (pass on CLI or in --config).")
    if not args.ldm_ckpt:
        ap.error("--ldm_ckpt is required (pass on CLI or in --config).")

    resolved_ldm_ckpt = _resolve_ckpt_path(args.ldm_ckpt, default_name="UNET_last.pt")
    run_name = f"{args.out_prefix}_{args.out_postfix}" if args.out_prefix else args.out_postfix
    args.outdir = os.path.join(args.outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nOutput dir: {args.outdir}\n")
    cfg = vars(args).copy()
    cfg["resolved_ldm_ckpt"] = resolved_ldm_ckpt
    for k, v in list(cfg.items()):
        if isinstance(v, tuple):
            cfg[k] = list(v)
    with open(os.path.join(args.outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size = tuple(int(x) for x in args.size.split(","))
    torch_autocast = base.str2bool(args.torch_autocast)
    use_cond = base.str2bool(args.ldm_use_cond)
    cond_keys = [x.strip() for x in args.cond_keys.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(torch_autocast and device.type == "cuda")
    print(f"Using device: {device}")

    base.seed_all(args.seed)

    cond_fields = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if "imageID" not in cond_fields:
        cond_fields.append("imageID")
    csv_columns = pd.read_csv(args.csv, nrows=1).columns.tolist()
    missing_cols = [c for c in cond_fields if c not in csv_columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in CSV required by --conditions/imageID: {missing_cols}. "
            f"Available columns: {csv_columns}"
        )

    channel = 0
    keys = ["image"]
    shared_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys),
            transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
            transforms.DivisiblePadd(keys=keys, k=32, mode="constant", constant_values=-1.0),
            transforms.CenterSpatialCropd(keys=keys, roi_size=size),
        ]
    )

    try:
        n_samples = int(args.n_samples)
    except Exception:
        n_samples = args.n_samples

    _, _, test_loader = base.make_dataloaders_from_csv(
        args.csv,
        conditions=cond_fields,
        train_transforms=shared_transforms,
        n_samples=n_samples,
        data_split_json_path=args.data_split_json_path,
        batch_size=args.batch,
        num_workers=args.workers,
        seed=args.seed,
    )
    test_count = len(test_loader.dataset)
    if test_count == 0:
        raise RuntimeError("Test split is empty; cannot run inference.")
    if test_count != 291:
        print(f"Warning: expected 291 test samples, but loader contains {test_count}.")

    num_generate = test_count if args.num_samples <= 0 else int(args.num_samples)
    if num_generate <= 0:
        raise ValueError("--num_samples must be > 0 or -1 for default.")
    print(f"Generating {num_generate} samples (test loader size = {test_count}).")

    ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
    ae_attention_levels = tuple(bool(int(x)) for x in args.ae_attention_levels.split(","))
    autoencoder = base.AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=ae_num_channels,
        latent_channels=args.ae_latent_ch,
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=ae_attention_levels,
    ).to(device)
    base._load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    ldm_num_channels = tuple(int(x) for x in args.ldm_num_channels.split(","))
    ldm_num_head_channels = tuple(int(x) for x in args.ldm_num_head_channels.split(","))
    unet_kwargs = dict(
        spatial_dims=3,
        in_channels=args.ae_latent_ch,
        out_channels=args.ae_latent_ch,
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
    if use_cond:
        unet_kwargs["with_conditioning"] = True
        unet_kwargs["cross_attention_dim"] = args.cond_dim
    unet = base.DiffusionModelUNet(**unet_kwargs).to(device)
    base._load_ckpt_into_unet(unet, resolved_ldm_ckpt, device)
    unet.eval()

    scale_factor = None
    if args.scale_factor > 0:
        scale_factor = float(args.scale_factor)
        print(f"Using scale_factor from CLI: {scale_factor}")
    else:
        try:
            ckpt = torch.load(resolved_ldm_ckpt, map_location=device, weights_only=False)
            scale_factor = float(ckpt["extra"]["scale_factor"])
            print(f"Using scale_factor from checkpoint extra: {scale_factor}")
        except Exception:
            scale_factor = None

    first_batch = next(iter(test_loader))
    with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
        z_ref = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))
    if scale_factor is None:
        z_std = max(float(torch.std(z_ref).item()), 1e-8)
        scale_factor = float(1.0 / z_std)
        print(f"scale_factor not found in checkpoint; estimated from first test batch: {scale_factor}")
    latent_shape = tuple(z_ref.shape[1:])

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=int(args.ddim_steps))

    index_width = max(3, len(str(max(0, num_generate - 1))))
    manifest = []
    test_iter = iter(test_loader)
    for i in tqdm(range(num_generate), desc="DDIM sampling", ncols=110):
        try:
            batch = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            batch = next(test_iter)

        img_id = _extract_image_id(batch, fallback=f"test_{i % test_count:03d}")
        affine = _extract_affine(batch)
        condition = None
        if use_cond:
            condition = _build_condition_tensor(batch, cond_keys, device)
            if condition is None:
                raise ValueError(
                    f"--ldm_use_cond=True but batch is missing at least one key in --cond_keys={cond_keys}"
                )

        x_gen = _ddim_sample_one(
            unet=unet,
            autoencoder=autoencoder,
            scheduler=scheduler,
            latent_shape=latent_shape,
            scale_factor=scale_factor,
            device=device,
            amp_enabled=amp_enabled,
            condition=condition,
        )
        arr = x_gen[0, 0].detach().float().cpu().numpy()
        filename = f"sample_{i:0{index_width}d}_{_safe_component(img_id)}.nii.gz"
        out_path = os.path.join(args.outdir, filename)
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), out_path)
        manifest.append(
            {
                "sample_index": i,
                "imageID": str(img_id),
                "output_file": filename,
            }
        )

    with open(os.path.join(args.outdir, "sample_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {num_generate} generated samples to: {args.outdir}")
    print("Saved sample mapping to: sample_manifest.json")


if __name__ == "__main__":
    main()
