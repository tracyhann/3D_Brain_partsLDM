from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

# Reuse utilities from your training scripts.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_3d_brain_ldm_aux_taux as base  # noqa: E402
import train_3d_brain_ldm_aux_taux_ddp as ddp_base  # noqa: E402


def _resolve_ckpt_path(path: str, default_name: str) -> str:
    if not path:
        return ""
    if os.path.isdir(path):
        resolved = os.path.join(path, default_name)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Checkpoint directory provided, but {default_name} was not found: {path}"
            )
        return resolved
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _safe_component(x: Any) -> str:
    s = re.sub(r"[^\w.\-]+", "-", str(x)).strip("-")
    return s if s else "unknown"


def _extract_image_id(batch: dict, fallback: str) -> str:
    v = batch.get("imageID", None)
    if v is None:
        return fallback
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
    img = batch.get("image", None)
    if img is None:
        return np.eye(4, dtype=np.float32)
    try:
        return base._extract_affine_from_tensor(img)
    except Exception:
        return np.eye(4, dtype=np.float32)


def _run_name_from_ckpt_path(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if p.is_dir():
        return p.name
    if p.parent.name:
        return p.parent.name
    return p.stem


def _read_scale_factor_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    keys: Sequence[str],
) -> Optional[float]:
    if not ckpt_path:
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    extra = ckpt.get("extra", {})
    if not isinstance(extra, dict):
        return None
    for k in keys:
        if k in extra:
            try:
                return float(extra[k])
            except Exception:
                continue
    return None


def _scheduler_timesteps_as_int(scheduler) -> list[int]:
    out = []
    for t in scheduler.timesteps:
        out.append(int(t.item()) if hasattr(t, "item") else int(t))
    return out


def _resolve_taux_for_ddim(
    *,
    t_aux_arg: int,
    taux_min: int,
    taux_max: int,
    taux_units: str,
    ddim_timesteps_desc: Sequence[int],
) -> tuple[int, int, int]:
    if len(ddim_timesteps_desc) < 1:
        raise ValueError("DDIM timestep schedule is empty.")
    n = len(ddim_timesteps_desc)

    if str(taux_units) == "ddim_steps_from_end":
        if int(t_aux_arg) >= 0:
            steps_from_end = int(t_aux_arg)
        else:
            steps_from_end = random.randint(int(taux_min), int(taux_max))
        steps_from_end = max(1, min(n, int(steps_from_end)))
        start_idx = n - steps_from_end
        t_aux_train = int(ddim_timesteps_desc[start_idx])
        return t_aux_train, start_idx, steps_from_end

    if int(t_aux_arg) >= 0:
        t_aux_train = int(t_aux_arg)
    else:
        t_aux_train = random.randint(int(taux_min), int(taux_max))
    t_aux_train = max(0, min(999, int(t_aux_train)))
    start_idx = min(range(n), key=lambda i: abs(int(ddim_timesteps_desc[i]) - t_aux_train))
    steps_from_end = n - start_idx
    return t_aux_train, start_idx, steps_from_end


def _binary_erode_mask_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Binary erosion for [B,C,D,H,W] masks using all-ones 3D kernel.
    Erosion radius is in voxels; radius=0 is a no-op.
    """
    r = int(radius)
    m = (mask > 0.5).float()
    if r <= 0:
        return m
    if m.ndim != 5:
        raise ValueError(f"Expected mask [B,C,D,H,W], got {tuple(m.shape)}")

    k = int(2 * r + 1)
    c = int(m.shape[1])
    w = torch.ones((c, 1, k, k, k), device=m.device, dtype=m.dtype)
    s = F.conv3d(m, w, bias=None, stride=1, padding=r, groups=c)
    keep = s >= (float(k**3) - 1e-6)
    return keep.to(dtype=m.dtype)


def _scan_existing_outputs(outdir: str, save_prefix: str) -> dict[int, dict[str, Any]]:
    """
    Scan existing sample files and recover index + safe image id from filenames:
      <save_prefix>_<index>_<safe_id>.nii.gz
    """
    pattern = re.compile(
        rf"^{re.escape(str(save_prefix))}_(\d+?)_(.+)\.nii(?:\.gz)?$"
    )
    entries: dict[int, dict[str, Any]] = {}
    if not os.path.isdir(outdir):
        return entries
    for name in os.listdir(outdir):
        m = pattern.match(name)
        if m is None:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        safe_id = str(m.group(2))
        entries[idx] = {
            "test_index": int(idx),
            "imageID": safe_id,
            "output_file": name,
            "resume_stub": True,
        }
    return entries


def _load_existing_manifest(manifest_path: str) -> dict[int, dict[str, Any]]:
    entries: dict[int, dict[str, Any]] = {}
    if not os.path.isfile(manifest_path):
        return entries
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[resume] Warning: failed to read existing manifest at {manifest_path}: {e}")
        return entries
    if not isinstance(data, list):
        print(f"[resume] Warning: existing manifest is not a list at {manifest_path}; ignoring.")
        return entries
    for row in data:
        if not isinstance(row, dict):
            continue
        if "test_index" not in row:
            continue
        try:
            idx = int(row["test_index"])
        except Exception:
            continue
        entries[idx] = row
    return entries


def _first_missing_index(indices: set[int], total: int) -> int:
    i = 0
    while i < int(total) and i in indices:
        i += 1
    return i


def _init_single_process_group(device: torch.device) -> tuple[bool, Optional[str]]:
    if dist.is_initialized():
        return False, None
    backend = "nccl" if device.type == "cuda" else "gloo"
    f = tempfile.NamedTemporaryFile(prefix="aux_taux_infer_", suffix=".dist", delete=False)
    f.close()
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{f.name}",
        rank=0,
        world_size=1,
    )
    return True, f.name


def _cleanup_process_group(created: bool, marker_path: Optional[str]):
    if created and dist.is_initialized():
        dist.destroy_process_group()
    if marker_path and os.path.exists(marker_path):
        try:
            os.remove(marker_path)
        except OSError:
            pass


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description=(
            "Inference for aux-tAux whole-brain model. "
            "Builds test loader using train_3d_brain_ldm_aux_taux_ddp.make_dataloaders_from_csv_ddp."
        ),
        parents=[pre_ap],
    )

    # Data
    ap.add_argument("--csv", default="data/processed_parts/whole_brain+3parts+masks_0206.csv")
    ap.add_argument("--data_split_json_path", default="data/patient_splits_image_ids_75_10_15.json")
    ap.add_argument("--whole_key", default="whole_brain")
    ap.add_argument("--spacing", default="1.5,1.5,1.5")
    ap.add_argument("--size", default="128,128,128")
    ap.add_argument("--hemi_size", default="64,128,128")
    ap.add_argument("--sub_size", default="128,96,64")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dataloader_persistent_workers", default=True)
    ap.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    ap.add_argument("--torch_autocast", default=True)
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--seed", type=int, default=1017)
    ap.add_argument(
        "--conditions",
        default="age,sex,group,condition,imageID",
        help="CSV fields kept in loader samples; include imageID for output naming.",
    )

    # Template masks
    ap.add_argument("--template_dir", default="data/template")
    ap.add_argument("--lhemi_template_name", default="lhemi_mask.nii.gz")
    ap.add_argument("--rhemi_template_name", default="rhemi_mask.nii.gz")
    ap.add_argument("--sub_template_name", default="sub_mask.nii.gz")

    # Checkpoints
    ap.add_argument("--whole_ae_ckpt", default="", help="Required: whole-brain AE checkpoint.")
    ap.add_argument("--whole_unet_ckpt", default="", help="Required: whole UNet checkpoint or run dir.")
    ap.add_argument("--resume_ckpt", default="", help="Optional override for whole UNet checkpoint.")
    ap.add_argument("--hemi_ae_ckpt", default="", help="Required: hemi AE checkpoint.")
    ap.add_argument("--hemi_unet_ckpt", default="", help="Required: hemi UNet checkpoint.")
    ap.add_argument("--hemi_conditioner_ckpt", default="", help="Optional conditioner checkpoint.")
    ap.add_argument("--sub_ae_ckpt", default="", help="Required: sub AE checkpoint.")
    ap.add_argument("--sub_unet_ckpt", default="", help="Required: sub UNet checkpoint.")

    # Architecture
    ap.add_argument("--whole_ae_latent_ch", type=int, default=8)
    ap.add_argument("--whole_ae_num_channels", default="64,128,256")
    ap.add_argument("--whole_ae_attention_levels", default="0,0,0")
    ap.add_argument("--whole_ldm_num_channels", default="256,256,512")
    ap.add_argument("--whole_ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--hemi_ae_latent_ch", type=int, default=8)
    ap.add_argument("--hemi_ae_num_channels", default="64,128,256")
    ap.add_argument("--hemi_ae_attention_levels", default="0,0,0")
    ap.add_argument("--hemi_ldm_num_channels", default="256,256,512")
    ap.add_argument("--hemi_ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--hemi_cond_dim", type=int, default=64)
    ap.add_argument("--hemi_n_parts", type=int, default=2)
    ap.add_argument("--sub_ae_latent_ch", type=int, default=8)
    ap.add_argument("--sub_ae_num_channels", default="64,128,256")
    ap.add_argument("--sub_ae_attention_levels", default="0,0,0")
    ap.add_argument("--sub_ldm_num_channels", default="256,256,512")
    ap.add_argument("--sub_ldm_num_head_channels", default="0,64,64")

    # Scale factors
    ap.add_argument("--whole_scale_factor", type=float, default=0.0, help="If <=0, auto/ckpt.")
    ap.add_argument("--hemi_scale_factor", type=float, default=0.0, help="If <=0, auto/ckpt.")
    ap.add_argument("--sub_scale_factor", type=float, default=0.0, help="If <=0, auto/ckpt.")

    # TAux compose/sampling controls
    ap.add_argument("--taux_min", type=int, default=100)
    ap.add_argument("--taux_max", type=int, default=300)
    ap.add_argument("--part_taux_max", type=int, default=-1)
    ap.add_argument("--scaffold_mode", choices=["sample", "forward_noise"], default="forward_noise")
    ap.add_argument("--compose_flip_lhemi", default=False)
    ap.add_argument("--compose_flip_rhemi", default=False)
    ap.add_argument("--compose_valid_thresh", type=float, default=-0.995)
    ap.add_argument(
        "--erode_r",
        type=int,
        default=0,
        help=(
            "Erosion radius (voxels) applied to m_known before latent injection. "
            "Use >0 for a more conservative coarse/injection mask."
        ),
    )
    ap.add_argument("--bg_value", type=float, default=-1.0)

    # Output controls
    ap.add_argument("--samples_root", default="samples", help="Root folder for generated outputs.")
    ap.add_argument("--save_prefix", default="sample", help="Output filename prefix before index/imageID.")
    ap.add_argument("--max_test_samples", type=int, default=-1, help="If >0, only generate for first N test samples.")
    ap.add_argument("--t_aux", type=int, default=-1, help="If >=0, use fixed t_aux; otherwise sample per case.")
    ap.add_argument("--ddim_steps", type=int, default=50, help="DDIM inference steps for post-injection denoising.")
    ap.add_argument(
        "--taux_units",
        choices=["train_timestep", "ddim_steps_from_end"],
        default="train_timestep",
        help=(
            "Interpretation of t_aux/taux_min/taux_max. "
            "'train_timestep' uses 0..999 diffusion steps; "
            "'ddim_steps_from_end' means remaining DDIM steps after injection (e.g., t_aux=1 -> last step only)."
        ),
    )
    ap.add_argument(
        "--post_taux_steps",
        type=int,
        default=-1,
        help="If >0, truncate denoising to this many DDIM steps after injection; -1 uses all remaining steps.",
    )

    # Load config defaults, but ignore unknown keys to allow passing training configs directly.
    ignored_cfg_keys = []
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            cfg_defaults = json.load(f)
        if not isinstance(cfg_defaults, dict):
            raise ValueError(f"Config must be a JSON object: {pre_args.config}")
        known_keys = {a.dest for a in ap._actions}
        for k, v in cfg_defaults.items():
            if k in known_keys:
                ap.set_defaults(**{k: v})
            else:
                ignored_cfg_keys.append(k)

    args = ap.parse_args()
    if ignored_cfg_keys:
        print(f"[config] Ignored unknown keys: {sorted(set(ignored_cfg_keys))}")

    if not args.csv:
        ap.error("--csv is required (pass on CLI or in --config).")
    if int(args.batch) < 1:
        ap.error("Require batch >= 1.")
    if int(args.workers) < 0:
        ap.error("Require workers >= 0.")
    if int(args.dataloader_prefetch_factor) < 0:
        ap.error("Require dataloader_prefetch_factor >= 0.")
    if int(args.taux_min) < 0 or int(args.taux_max) >= 1000 or int(args.taux_min) > int(args.taux_max):
        ap.error("Require 0 <= taux_min <= taux_max < 1000.")
    if int(args.part_taux_max) >= 1000:
        ap.error("Require part_taux_max < 1000 (or -1 to disable cap).")
    if int(args.t_aux) >= 1000:
        ap.error("Require t_aux < 1000 (or -1 for random in [taux_min, taux_max]).")
    if int(args.ddim_steps) < 1 or int(args.ddim_steps) > 1000:
        ap.error("Require 1 <= ddim_steps <= 1000.")
    if int(args.post_taux_steps) == 0 or int(args.post_taux_steps) < -1:
        ap.error("Require post_taux_steps in {-1} or >= 1.")
    if int(args.erode_r) < 0:
        ap.error("Require erode_r >= 0.")
    required_ckpt_args = [
        "whole_ae_ckpt",
        "hemi_ae_ckpt",
        "hemi_unet_ckpt",
        "sub_ae_ckpt",
        "sub_unet_ckpt",
    ]
    if not (str(args.resume_ckpt).strip() or str(args.whole_unet_ckpt).strip()):
        ap.error("Either --whole_unet_ckpt or --resume_ckpt is required.")
    for k in required_ckpt_args:
        if not str(getattr(args, k)).strip():
            ap.error(f"--{k} is required.")

    # Resolve checkpoint paths.
    whole_unet_candidate = args.resume_ckpt if str(args.resume_ckpt).strip() else args.whole_unet_ckpt
    resolved_ckpts = {
        "whole_ae_ckpt": _resolve_ckpt_path(args.whole_ae_ckpt, "AE_best.pt"),
        "whole_unet_ckpt": _resolve_ckpt_path(whole_unet_candidate, "UNET_last.pt"),
        "hemi_ae_ckpt": _resolve_ckpt_path(args.hemi_ae_ckpt, "AE_best.pt"),
        "hemi_unet_ckpt": _resolve_ckpt_path(args.hemi_unet_ckpt, "UNET_last.pt"),
        "sub_ae_ckpt": _resolve_ckpt_path(args.sub_ae_ckpt, "AE_best.pt"),
        "sub_unet_ckpt": _resolve_ckpt_path(args.sub_unet_ckpt, "UNET_last.pt"),
    }
    if str(args.hemi_conditioner_ckpt).strip():
        resolved_ckpts["hemi_conditioner_ckpt"] = _resolve_ckpt_path(
            args.hemi_conditioner_ckpt, "UNET_last.pt"
        )
    else:
        resolved_ckpts["hemi_conditioner_ckpt"] = ""

    # Device + single-process distributed init (needed by DDP loader helper).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    created_pg, pg_marker = _init_single_process_group(device)

    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            raise RuntimeError("This inference script is intended for rank-0 execution.")

        torch_autocast = ddp_base.parse_bool(args.torch_autocast)
        amp_enabled = bool(torch_autocast and device.type == "cuda")
        base.seed_all(int(args.seed), deterministic=True, allow_tf32=True)
        random.seed(int(args.seed))

        conditions = [x.strip() for x in str(args.conditions).split(",") if x.strip()]
        if "imageID" not in conditions:
            conditions.append("imageID")

        spacing = base._parse_float_tuple(args.spacing)
        whole_size = base._parse_int_tuple(args.size)
        hemi_size = base._parse_int_tuple(args.hemi_size)
        sub_size = base._parse_int_tuple(args.sub_size)
        n_samples = None if str(args.n_samples).upper() == "ALL" else int(args.n_samples)

        shared_transforms = base.build_image_transforms(spacing=spacing, whole_size=whole_size)
        _, _, test_loader, _ = ddp_base.make_dataloaders_from_csv_ddp(
            csv_path=args.csv,
            image_key=args.whole_key,
            rank=rank,
            conditions=tuple(conditions),
            train_transforms=shared_transforms,
            n_samples=n_samples,
            data_split_json_path=args.data_split_json_path,
            batch_size=int(args.batch),
            num_workers=int(args.workers),
            persistent_workers=ddp_base.parse_bool(args.dataloader_persistent_workers),
            prefetch_factor=int(args.dataloader_prefetch_factor),
            seed=int(args.seed),
        )
        if test_loader is None:
            raise RuntimeError("Test loader is None on rank-0; unexpected distributed state.")
        if len(test_loader.dataset) == 0:
            raise RuntimeError("Test split is empty; cannot run inference.")

        # Build template masks.
        lhemi_mask_path = base._resolve_path(args.template_dir, args.lhemi_template_name)
        rhemi_mask_path = base._resolve_path(args.template_dir, args.rhemi_template_name)
        sub_mask_path = base._resolve_path(args.template_dir, args.sub_template_name)

        first_batch = next(iter(test_loader))
        whole_shape = tuple(int(v) for v in first_batch["image"].shape[-3:])
        mask_ctx = base.load_template_masks(
            lhemi_mask_path=lhemi_mask_path,
            rhemi_mask_path=rhemi_mask_path,
            sub_mask_path=sub_mask_path,
            spacing=spacing,
            whole_shape=whole_shape,
            device=device,
        )
        if int(args.erode_r) > 0:
            mask_ctx = dict(mask_ctx)
            old_known_vox = int((mask_ctx["m_known"] > 0.5).sum().item())
            mask_ctx["m_known"] = _binary_erode_mask_3d(mask_ctx["m_known"], radius=int(args.erode_r))
            new_known_vox = int((mask_ctx["m_known"] > 0.5).sum().item())
            print(
                f"[mask] Applied m_known erosion: erode_r={int(args.erode_r)} "
                f"(voxels {old_known_vox} -> {new_known_vox})."
            )
            if new_known_vox == 0:
                print("[mask] Warning: erode_r removed all known-mask voxels; injection mask is now empty.")

        # Build models.
        whole_ae = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=base._parse_int_tuple(args.whole_ae_num_channels),
            latent_channels=int(args.whole_ae_latent_ch),
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=tuple(bool(int(x)) for x in str(args.whole_ae_attention_levels).split(",")),
        ).to(device)

        whole_unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=int(args.whole_ae_latent_ch),
            out_channels=int(args.whole_ae_latent_ch),
            num_res_blocks=2,
            num_channels=base._parse_int_tuple(args.whole_ldm_num_channels),
            attention_levels=(False, True, True),
            num_head_channels=base._parse_int_tuple(args.whole_ldm_num_head_channels),
            norm_num_groups=32,
            norm_eps=1e-6,
            resblock_updown=True,
            upcast_attention=True,
            use_flash_attention=False,
        ).to(device)

        hemi_ae = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=base._parse_int_tuple(args.hemi_ae_num_channels),
            latent_channels=int(args.hemi_ae_latent_ch),
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=tuple(bool(int(x)) for x in str(args.hemi_ae_attention_levels).split(",")),
        ).to(device)

        hemi_unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=int(args.hemi_ae_latent_ch),
            out_channels=int(args.hemi_ae_latent_ch),
            num_res_blocks=2,
            num_channels=base._parse_int_tuple(args.hemi_ldm_num_channels),
            attention_levels=(False, True, True),
            num_head_channels=base._parse_int_tuple(args.hemi_ldm_num_head_channels),
            with_conditioning=True,
            cross_attention_dim=int(args.hemi_cond_dim),
            norm_num_groups=32,
            norm_eps=1e-6,
            resblock_updown=True,
            upcast_attention=True,
            use_flash_attention=False,
        ).to(device)

        hemi_conditioner = base.HemisphereConditioner(
            n_parts=int(args.hemi_n_parts),
            d_model=int(args.hemi_cond_dim),
        ).to(device)

        sub_ae = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=base._parse_int_tuple(args.sub_ae_num_channels),
            latent_channels=int(args.sub_ae_latent_ch),
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=tuple(bool(int(x)) for x in str(args.sub_ae_attention_levels).split(",")),
        ).to(device)

        sub_unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=int(args.sub_ae_latent_ch),
            out_channels=int(args.sub_ae_latent_ch),
            num_res_blocks=2,
            num_channels=base._parse_int_tuple(args.sub_ldm_num_channels),
            attention_levels=(False, True, True),
            num_head_channels=base._parse_int_tuple(args.sub_ldm_num_head_channels),
            norm_num_groups=32,
            norm_eps=1e-6,
            resblock_updown=True,
            upcast_attention=True,
            use_flash_attention=False,
        ).to(device)

        # Load checkpoints.
        base._load_ckpt_into_ae(whole_ae, resolved_ckpts["whole_ae_ckpt"], device)
        base._load_model_weights_shape_safe(whole_unet, resolved_ckpts["whole_unet_ckpt"], device)
        base._load_ckpt_into_ae(hemi_ae, resolved_ckpts["hemi_ae_ckpt"], device)
        base._load_model_weights_shape_safe(hemi_unet, resolved_ckpts["hemi_unet_ckpt"], device)
        if resolved_ckpts["hemi_conditioner_ckpt"]:
            base._load_conditioner_weights_from_ckpt(
                hemi_conditioner,
                resolved_ckpts["hemi_conditioner_ckpt"],
                device,
            )
        else:
            base._load_conditioner_weights_from_ckpt(
                hemi_conditioner,
                resolved_ckpts["hemi_unet_ckpt"],
                device,
            )
        base._load_ckpt_into_ae(sub_ae, resolved_ckpts["sub_ae_ckpt"], device)
        base._load_model_weights_shape_safe(sub_unet, resolved_ckpts["sub_unet_ckpt"], device)

        for m in [whole_ae, whole_unet, hemi_ae, hemi_unet, hemi_conditioner, sub_ae, sub_unet]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        # Scale factors: CLI override > checkpoint extra > estimate from first test sample.
        whole_sf = float(args.whole_scale_factor) if float(args.whole_scale_factor) > 0 else None
        hemi_sf = float(args.hemi_scale_factor) if float(args.hemi_scale_factor) > 0 else None
        sub_sf = float(args.sub_scale_factor) if float(args.sub_scale_factor) > 0 else None

        if whole_sf is None:
            whole_sf = _read_scale_factor_from_ckpt(
                resolved_ckpts["whole_unet_ckpt"],
                device,
                keys=("whole_scale_factor", "scale_factor"),
            )
        if hemi_sf is None:
            hemi_sf = _read_scale_factor_from_ckpt(
                resolved_ckpts["hemi_unet_ckpt"],
                device,
                keys=("hemi_scale_factor", "scale_factor"),
            )
        if sub_sf is None:
            sub_sf = _read_scale_factor_from_ckpt(
                resolved_ckpts["sub_unet_ckpt"],
                device,
                keys=("sub_scale_factor", "scale_factor"),
            )

        ref_img = first_batch["image"].to(device, non_blocking=True)[:1]
        if whole_sf is None:
            with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
                z = whole_ae.encode_stage_2_inputs(ref_img)
            whole_sf = float(1.0 / max(torch.std(z).item(), 1e-8))

        if hemi_sf is None or sub_sf is None:
            xL_whole_masked = base._apply_binary_mask_bg(ref_img, mask_ctx["mL"], float(args.bg_value))
            xR_whole_masked = base._apply_binary_mask_bg(ref_img, mask_ctx["mR"], float(args.bg_value))
            xS_whole_masked = base._apply_binary_mask_bg(ref_img, mask_ctx["mS"], float(args.bg_value))
            xL = base._crop_right(xL_whole_masked, hemi_size)
            xR_masked_raw = base._crop_left(xR_whole_masked, hemi_size)
            xR = base._flip_lr(xR_masked_raw)
            xS = base._crop_sub(xS_whole_masked, sub_size)
            if hemi_sf is None:
                hemi_sf = base._estimate_scale_factor_from_batch(
                    ae=hemi_ae,
                    x=torch.cat([xL, xR], dim=0),
                    amp_enabled=amp_enabled,
                )
            if sub_sf is None:
                sub_sf = base._estimate_scale_factor_from_batch(
                    ae=sub_ae,
                    x=xS,
                    amp_enabled=amp_enabled,
                )

        # Output directory: samples/<run_ckpt_name>/
        run_ckpt_name = _run_name_from_ckpt_path(resolved_ckpts["whole_unet_ckpt"])
        outdir = os.path.join(args.samples_root, run_ckpt_name)
        os.makedirs(outdir, exist_ok=True)

        infer_cfg = vars(args).copy()
        infer_cfg["resolved_ckpts"] = resolved_ckpts
        infer_cfg["resolved_scale_factors"] = {
            "whole_scale_factor": float(whole_sf),
            "hemi_scale_factor": float(hemi_sf),
            "sub_scale_factor": float(sub_sf),
        }
        infer_cfg["run_ckpt_name"] = run_ckpt_name
        with open(os.path.join(outdir, "infer_args.json"), "w", encoding="utf-8") as f:
            json.dump(infer_cfg, f, indent=2, sort_keys=True)

        branch_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            schedule="scaled_linear_beta",
            beta_start=0.0015,
            beta_end=0.0195,
            clip_sample=False,
        )
        branch_scheduler.set_timesteps(num_inference_steps=1000)
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            schedule="scaled_linear_beta",
            beta_start=0.0015,
            beta_end=0.0195,
            clip_sample=False,
        )
        ddim_scheduler.set_timesteps(num_inference_steps=int(args.ddim_steps))
        ddim_timesteps = _scheduler_timesteps_as_int(ddim_scheduler)

        total = len(test_loader.dataset)
        if int(args.max_test_samples) > 0:
            total = min(total, int(args.max_test_samples))

        print(f"Using device: {device}")
        print(f"Output dir: {outdir}")
        print(f"Generating 1 sample per test case (count={total}, DDIM steps={int(args.ddim_steps)}).")

        index_width = max(3, len(str(max(0, total - 1))))
        manifest_path = os.path.join(outdir, "sample_manifest.json")
        existing_stub_by_index = _scan_existing_outputs(outdir=outdir, save_prefix=str(args.save_prefix))
        existing_manifest_by_index = _load_existing_manifest(manifest_path)
        manifest_by_index: dict[int, dict[str, Any]] = dict(existing_stub_by_index)
        manifest_by_index.update(existing_manifest_by_index)

        existing_indices = set(existing_stub_by_index.keys())
        existing_indices = {i for i in existing_indices if 0 <= i < int(total)}
        resume_from = _first_missing_index(existing_indices, total=int(total))

        if len(existing_indices) > 0:
            print(
                f"[resume] detected {len(existing_indices)} existing output files in {outdir}; "
                f"next missing test_index={resume_from}."
            )
        if resume_from >= int(total):
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump([manifest_by_index[k] for k in sorted(manifest_by_index.keys())], f, indent=2)
            print(f"[resume] all requested samples already exist (total={total}). Nothing to do.")
            print("Saved sample mapping to: sample_manifest.json")
            return

        generated = 0
        skipped_existing = 0
        for i, batch in enumerate(tqdm(test_loader, desc="Aux-tAux inference", ncols=120)):
            if i >= total:
                break
            if i < resume_from:
                continue
            images = batch["image"].to(device, non_blocking=True)
            if images.shape[0] != 1:
                images = images[:1]

            image_id = _extract_image_id(batch, fallback=f"test_{i:05d}")
            safe_id = _safe_component(image_id)
            out_name = f"{args.save_prefix}_{i:0{index_width}d}_{safe_id}.nii.gz"
            out_path = os.path.join(outdir, out_name)
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                skipped_existing += 1
                if i not in manifest_by_index:
                    manifest_by_index[i] = {
                        "test_index": i,
                        "imageID": str(image_id),
                        "output_file": out_name,
                        "resume_stub": True,
                    }
                continue

            t_aux_train, ddim_start_idx, steps_from_end = _resolve_taux_for_ddim(
                t_aux_arg=int(args.t_aux),
                taux_min=int(args.taux_min),
                taux_max=int(args.taux_max),
                taux_units=str(args.taux_units),
                ddim_timesteps_desc=ddim_timesteps,
            )
            denoise_timesteps = ddim_timesteps[ddim_start_idx:]
            if int(args.post_taux_steps) > 0:
                denoise_timesteps = denoise_timesteps[: int(args.post_taux_steps)]
            if len(denoise_timesteps) < 1:
                raise RuntimeError("Resolved zero DDIM steps for denoising after t_aux.")

            with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
                z_whole0 = whole_ae.encode_stage_2_inputs(images) * float(whole_sf)

            payload = base.build_expert_composite_branch(
                whole_unet=whole_unet,
                whole_ae=whole_ae,
                hemi_ae=hemi_ae,
                hemi_unet=hemi_unet,
                hemi_conditioner=hemi_conditioner,
                sub_ae=sub_ae,
                sub_unet=sub_unet,
                scheduler=branch_scheduler,
                whole_scale_factor=float(whole_sf),
                hemi_scale_factor=float(hemi_sf),
                sub_scale_factor=float(sub_sf),
                mask_ctx=mask_ctx,
                hemi_shape=hemi_size,
                sub_shape=sub_size,
                t_aux=int(t_aux_train),
                part_taux_max=int(args.part_taux_max),
                amp_enabled=amp_enabled,
                scaffold_mode=str(args.scaffold_mode),
                z_whole0_ref=z_whole0,
                compose_flip_lhemi=ddp_base.parse_bool(args.compose_flip_lhemi),
                compose_flip_rhemi=ddp_base.parse_bool(args.compose_flip_rhemi),
                compose_valid_thresh=float(args.compose_valid_thresh),
                bg_value=float(args.bg_value),
            )
            z_final0 = payload["z_mix_taux"]
            for t_int in denoise_timesteps:
                t_batch = torch.full((z_final0.shape[0],), int(t_int), device=device, dtype=torch.long)
                with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
                    eps = whole_unet(z_final0, timesteps=t_batch)
                z_final0 = base._step_prev_sample(
                    ddim_scheduler.step(model_output=eps, timestep=int(t_int), sample=z_final0)
                )
            with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
                x_final = whole_ae.decode_stage_2_outputs(z_final0 / float(whole_sf))

            affine = _extract_affine(batch)
            nib.save(
                nib.Nifti1Image(x_final[0, 0].detach().cpu().float().numpy().astype(np.float32), affine),
                out_path,
            )

            manifest_by_index[i] = {
                "test_index": i,
                "imageID": str(image_id),
                "output_file": out_name,
                "t_aux": int(t_aux_train),
                "taux_units": str(args.taux_units),
                "ddim_start_index": int(ddim_start_idx),
                "ddim_start_timestep": int(ddim_timesteps[ddim_start_idx]),
                "ddim_steps_remaining": int(steps_from_end),
                "ddim_steps_used": int(len(denoise_timesteps)),
            }
            generated += 1

        final_manifest = [manifest_by_index[k] for k in sorted(manifest_by_index.keys())]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(final_manifest, f, indent=2)

        print(
            f"Saved {generated} generated samples to: {outdir} "
            f"(resume_start={resume_from}, skipped_existing={skipped_existing})."
        )
        print("Saved sample mapping to: sample_manifest.json")
    finally:
        _cleanup_process_group(created_pg, pg_marker)


if __name__ == "__main__":
    main()
