import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import first
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from eval_utils import plot_unet_loss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


# Avoid shared-memory exhaustion in DataLoader workers (e.g., limited /dev/shm)
mp.set_sharing_strategy("file_system")


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")


def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_int_tuple(v: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(v).split(","))


def _parse_float_tuple(v: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(v).split(","))


def _resolve_path(template_dir: str, p: str) -> str:
    # Support absolute paths, direct relative paths, or template_dir + name.
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(template_dir, p)


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def _load_ckpt_into_ae(ae, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    ae.load_state_dict(state, strict=False)


def _load_conditioner_weights_from_ckpt(
    conditioner: torch.nn.Module,
    ckpt_path: str,
    device,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "conditioner_state_dict" in ckpt and ckpt["conditioner_state_dict"] is not None:
        state = ckpt["conditioner_state_dict"]
    else:
        state = ckpt.get("state_dict", ckpt)
    conditioner.load_state_dict(state, strict=False)


def _load_model_weights_shape_safe(model: torch.nn.Module, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    compatible = {}
    skipped_shape = []

    for k, v in state.items():
        if k not in model_state:
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        compatible[k] = v

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"[load_model_weights] {os.path.basename(ckpt_path)}: "
        f"loaded={len(compatible)} missing={len(missing)} unexpected={len(unexpected)} "
        f"skipped_shape={len(skipped_shape)}"
    )
    if skipped_shape:
        preview = ", ".join([f"{k}: ckpt{cs}!=model{ms}" for k, cs, ms in skipped_shape[:3]])
        print(f"[load_model_weights] shape-mismatch preview: {preview}")


def _save_ckpt(
    path: str,
    unet: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    global_step: int,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "state_dict": unet.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


def _load_ckpt_into_unet(
    unet: torch.nn.Module,
    ckpt_path: str,
    device,
    opt: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
):
    """
    Returns: (global_step, epoch, extra)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" not in ckpt:
        missing, unexpected = unet.load_state_dict(ckpt, strict=False)
        print(f"[resume] loaded raw state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        return 0, 0, {}

    missing, unexpected = unet.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[resume] loaded packaged ckpt: missing={len(missing)} unexpected={len(unexpected)}")

    if opt is not None and ckpt.get("optimizer") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            print(f"[resume] optimizer state incompatible, skipping optimizer restore: {e}")
    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[resume] scaler state incompatible, skipping scaler restore: {e}")

    gs = int(ckpt.get("global_step", 0))
    ep = int(ckpt.get("epoch", 0))
    extra = ckpt.get("extra", {}) or {}
    return gs, ep, extra


def _read_scale_factor_from_ckpt(ckpt_path: str, device) -> Optional[float]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    extra = ckpt.get("extra", None)
    if isinstance(extra, dict) and "scale_factor" in extra:
        try:
            return float(extra["scale_factor"])
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def make_dataloaders_from_csv(
    *,
    csv_path: str,
    image_key: str,
    conditions=("age", "sex", "group"),
    train_transforms=None,
    n_samples=None,
    data_split_json_path="data/patient_splits_image_ids_75_10_15.json",
    batch_size=1,
    num_workers=8,
):
    with open(data_split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    image_ids = {}
    for image_id in splits["train"]:
        image_ids[image_id] = "train"
    for image_id in splits["val"]:
        image_ids[image_id] = "val"
    for image_id in splits["test"]:
        image_ids[image_id] = "test"

    df = pd.read_csv(csv_path)
    train_data, val_data, test_data = [], [], []
    n_train_added = 0

    for _, row in df.iterrows():
        image_id = row["imageID"]
        if image_id not in image_ids:
            continue
        sample = {"image": row[image_key]}
        for c in conditions:
            if c in row:
                sample[c] = row[c]

        split = image_ids[image_id]
        if split == "train":
            if n_samples is None or n_train_added < n_samples:
                train_data.append(sample)
                n_train_added += 1
        elif split == "val":
            val_data.append(sample)
        elif split == "test":
            test_data.append(sample)

    train_ds = Dataset(data=train_data, transform=train_transforms)
    print(f"Transformed train shape (image): {train_ds[0]['image'].shape}")
    print(f"Number of training samples: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_ds = Dataset(data=val_data, transform=train_transforms)
    print(f"Number of validation samples: {len(val_ds)}")
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_ds = Dataset(data=test_data, transform=train_transforms)
    print(f"Number of test samples: {len(test_ds)}")
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_image_transforms(
    *,
    spacing: Tuple[float, float, float],
    whole_size: Tuple[int, int, int],
):
    keys = ["image"]
    channel = 0
    return transforms.Compose(
        [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        transforms.EnsureTyped(keys=keys),
        transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
        transforms.DivisiblePadd(keys=keys, k=32, mode="constant",constant_values=-1.0),
        transforms.CenterSpatialCropd(keys=keys, roi_size=whole_size),
        ]
    )


def build_template_mask_transforms(
    *,
    spacing: Tuple[float, float, float],
):
    keys = ["lhemi_mask", "rhemi_mask", "sub_mask"]
    channel = 0
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys),
            transforms.Spacingd(keys=keys, pixdim=spacing, mode=("nearest",) * len(keys)),
            transforms.Lambdad(keys=keys, func=lambda x: (x > 0).float()),
        ]
    )


def align_binary_mask_to_shape(
    mask: torch.Tensor,
    target_shape: Tuple[int, int, int],
    anchors: Tuple[str, str, str],
) -> torch.Tensor:
    """
    Align a binary mask tensor to target shape by anchor-aware crop/pad.
    Input: [C,D,H,W] or [D,H,W]
    Anchors per spatial dim:
      - "start": keep content at low index, pad/crop on high side
      - "end": keep content at high index, pad/crop on low side
      - "center": center align
    """
    x = (mask > 0).float()
    squeezed = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeezed = True
    if x.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W] or [D,H,W], got {tuple(mask.shape)}")

    for i, tgt in enumerate(target_shape):
        axis = 1 + i
        cur = int(x.shape[axis])
        if cur <= tgt:
            continue
        delta = cur - tgt
        anchor = anchors[i]
        if anchor == "start":
            start = 0
        elif anchor == "end":
            start = delta
        else:
            start = delta // 2
        end = start + tgt
        slicer = [slice(None)] * x.ndim
        slicer[axis] = slice(start, end)
        x = x[tuple(slicer)]

    pads = []
    for i in reversed(range(3)):  # W, H, D order for torch.nn.functional.pad
        axis = 1 + i
        cur = int(x.shape[axis])
        tgt = int(target_shape[i])
        delta = max(0, tgt - cur)
        anchor = anchors[i]
        if anchor == "start":
            before, after = 0, delta
        elif anchor == "end":
            before, after = delta, 0
        else:
            before = delta // 2
            after = delta - before
        pads.extend([before, after])

    if any(pads):
        x = F.pad(x, pads, mode="constant", value=0.0)

    x = (x > 0.5).float()
    if squeezed:
        x = x.squeeze(0)
    return x


def load_template_masks(
    *,
    lhemi_mask_path: str,
    rhemi_mask_path: str,
    sub_mask_path: str,
    spacing: Tuple[float, float, float],
    whole_shape: Tuple[int, int, int],
    device,
):
    for p in [lhemi_mask_path, rhemi_mask_path, sub_mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Template mask not found: {p}")

    tx = build_template_mask_transforms(spacing=spacing)
    out = tx(
        {
            "lhemi_mask": lhemi_mask_path,
            "rhemi_mask": rhemi_mask_path,
            "sub_mask": sub_mask_path,
        }
    )

    mL = align_binary_mask_to_shape(
        (out["lhemi_mask"] > 0).float(),
        target_shape=whole_shape,
        anchors=("end", "start", "start"),
    ).to(device)
    mR = align_binary_mask_to_shape(
        (out["rhemi_mask"] > 0).float(),
        target_shape=whole_shape,
        anchors=("start", "start", "start"),
    ).to(device)
    mS = align_binary_mask_to_shape(
        (out["sub_mask"] > 0).float(),
        target_shape=whole_shape,
        anchors=("start", "start", "start"),
    ).to(device)

    if mL.ndim == 4:
        mL = mL.unsqueeze(0)
    if mR.ndim == 4:
        mR = mR.unsqueeze(0)
    if mS.ndim == 4:
        mS = mS.unsqueeze(0)

    mask_sum = mL + mR + mS
    m_paste = torch.clamp(mask_sum, 0.0, 1.0)
    m_overlap = (mask_sum >= 2.0).float()
    m_known = m_paste * (1.0 - m_overlap)
    return {
        "mL": mL,
        "mR": mR,
        "mS": mS,
        "m_known": m_known,
        "m_overlap": m_overlap,
    }


# -----------------------------------------------------------------------------
# Geometry helpers (crop/flip/paste)
# -----------------------------------------------------------------------------
def downsample_mask_any_voxel(mask: torch.Tensor, latent_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Any-voxel policy using adaptive max pooling to latent resolution.
    Input mask shape: [B,1,D,H,W]
    """
    if tuple(mask.shape[-3:]) == tuple(latent_shape):
        return (mask > 0.5).float()
    m = F.adaptive_max_pool3d(mask.float(), output_size=latent_shape)
    return (m > 0).float()


def _flip_lr(x: torch.Tensor) -> torch.Tensor:
    # left-right canonicalization along first spatial axis (D in [B,C,D,H,W]).
    return torch.flip(x, dims=[2])


def _apply_binary_mask_bg(x: torch.Tensor, mask: torch.Tensor, bg_value: float) -> torch.Tensor:
    m = (mask > 0.5).float()
    return x * m + bg_value * (1.0 - m)


def _crop_left(x: torch.Tensor, hemi_shape: Tuple[int, int, int]) -> torch.Tensor:
    d, h, w = hemi_shape
    return x[:, :, :d, :h, :w]


def _crop_right(x: torch.Tensor, hemi_shape: Tuple[int, int, int]) -> torch.Tensor:
    d, h, w = hemi_shape
    return x[:, :, -d:, :h, :w]


def _crop_sub(x: torch.Tensor, sub_shape: Tuple[int, int, int]) -> torch.Tensor:
    d, h, w = sub_shape
    return x[:, :, :d, :h, :w]


def _compose_from_parts_with_unknown_overlap(
    *,
    whole_shape: Tuple[int, int, int, int, int],
    x_lhemi: torch.Tensor,
    x_rhemi_right: torch.Tensor,
    x_sub: torch.Tensor,
    mL_world: torch.Tensor,
    mR_world: torch.Tensor,
    mS_world: torch.Tensor,
    flip_lhemi_valid: bool,
    flip_rhemi_valid: bool,
    valid_thresh: float,
    bg_value: float,
) -> torch.Tensor:
    """
    Build x_coarse in whole image space.
    Overlaps are marked unknown by setting to bg_value.
    """
    b, c, d, h, w = whole_shape
    out = torch.full((b, c, d, h, w), bg_value, dtype=x_lhemi.dtype, device=x_lhemi.device)
    occupied = torch.zeros((b, 1, d, h, w), dtype=torch.bool, device=x_lhemi.device)
    # Compose directly from image-space part assets; do not template-gate at paste time.
    _ = (mL_world, mR_world, mS_world)

    # Very low thresholds (e.g., -0.995 with bg=-1) keep too much decoder background.
    # Clamp to a safer floor relative to bg to avoid box/speckle artifacts.
    valid_floor = float(bg_value) + 0.05
    effective_valid_thresh = max(float(valid_thresh), valid_floor)

    def _paste(
        src: torch.Tensor,
        x0: int,
        y0: int,
        z0: int,
        valid: Optional[torch.Tensor] = None,
    ):
        nonlocal out, occupied
        _, _, sd, sh, sw = src.shape
        x1 = min(d, x0 + sd)
        y1 = min(h, y0 + sh)
        z1 = min(w, z0 + sw)
        x0c = max(0, x0)
        y0c = max(0, y0)
        z0c = max(0, z0)
        if x0c >= x1 or y0c >= y1 or z0c >= z1:
            return

        sx0 = x0c - x0
        sy0 = y0c - y0
        sz0 = z0c - z0
        sx1 = sx0 + (x1 - x0c)
        sy1 = sy0 + (y1 - y0c)
        sz1 = sz0 + (z1 - z0c)

        srcv = src[:, :, sx0:sx1, sy0:sy1, sz0:sz1]
        if valid is None:
            v = srcv > effective_valid_thresh
        else:
            v = valid[:, :, sx0:sx1, sy0:sy1, sz0:sz1] > 0.5

        oview = occupied[:, :, x0c:x1, y0c:y1, z0c:z1]
        xview = out[:, :, x0c:x1, y0c:y1, z0c:z1]

        overlap = v & oview
        if overlap.any():
            xview = torch.where(
                overlap.expand_as(xview),
                torch.full_like(xview, bg_value),
                xview,
            )

        fresh = v & (~oview)
        if fresh.any():
            xview = torch.where(fresh.expand_as(xview), srcv, xview)
            oview = oview | fresh

        out[:, :, x0c:x1, y0c:y1, z0c:z1] = xview
        occupied[:, :, x0c:x1, y0c:y1, z0c:z1] = oview

    # Paste left hemi from image-space asset into right canvas side.
    x_lhemi_paste = _flip_lr(x_lhemi) if flip_lhemi_valid else x_lhemi
    _paste(
        x_lhemi_paste,
        x0=d - x_lhemi_paste.shape[2],
        y0=0,
        z0=0,
        valid=None,
    )

    # Paste right hemi from image-space asset into left canvas side.
    x_rhemi_paste = _flip_lr(x_rhemi_right) if flip_rhemi_valid else x_rhemi_right
    _paste(
        x_rhemi_paste,
        x0=0,
        y0=0,
        z0=0,
        valid=None,
    )

    # Paste sub from image-space asset.
    _paste(x_sub, x0=0, y0=0, z0=0, valid=None)

    return out


# -----------------------------------------------------------------------------
# Diffusion helpers
# -----------------------------------------------------------------------------
def _step_prev_sample(step_out):
    # Match older/newer scheduler APIs (object with prev_sample vs tuple/list).
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, (tuple, list)):
        return step_out[0]
    if isinstance(step_out, dict) and "prev_sample" in step_out:
        return step_out["prev_sample"]
    return step_out


def _unet_eps(
    *,
    unet: torch.nn.Module,
    z_t: torch.Tensor,
    t_batch: torch.Tensor,
    amp_enabled: bool,
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with autocast(device_type="cuda", enabled=amp_enabled):
        if context is not None:
            return unet(z_t, timesteps=t_batch, context=context)
        return unet(z_t, timesteps=t_batch)


def _reverse_ddpm_to_t0(
    *,
    unet: torch.nn.Module,
    scheduler: DDPMScheduler,
    z_t: torch.Tensor,
    t_start: int,
    amp_enabled: bool,
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    z = z_t
    for t in range(int(t_start), -1, -1):
        t_batch = torch.full((z.shape[0],), t, device=z.device, dtype=torch.long)
        eps = _unet_eps(unet=unet, z_t=z, t_batch=t_batch, amp_enabled=amp_enabled, context=context)
        z = _step_prev_sample(scheduler.step(model_output=eps, timestep=t, sample=z))
    return z


def _reverse_ddpm_from_T_to_taux(
    *,
    unet: torch.nn.Module,
    scheduler: DDPMScheduler,
    batch_size: int,
    latent_shape: Tuple[int, int, int, int],
    t_aux: int,
    device,
    amp_enabled: bool,
) -> torch.Tensor:
    z = torch.randn((batch_size,) + tuple(latent_shape), device=device)
    t_start = int(scheduler.num_train_timesteps - 1)
    for t in range(t_start, int(t_aux) - 1, -1):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        eps = _unet_eps(unet=unet, z_t=z, t_batch=t_batch, amp_enabled=amp_enabled, context=None)
        z = _step_prev_sample(scheduler.step(model_output=eps, timestep=t, sample=z))
    return z


def _predict_prev_from_eps_ddpm(
    *,
    scheduler: DDPMScheduler,
    z_t: torch.Tensor,
    eps_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample z_{t-1} from model epsilon prediction.
    Uses scheduler.step sample-wise because timesteps differ across batch.
    """
    outs = []
    for i in range(z_t.shape[0]):
        t_i = int(timesteps[i].item())
        prev = _step_prev_sample(
            scheduler.step(
                model_output=eps_pred[i : i + 1],
                timestep=t_i,
                sample=z_t[i : i + 1],
            )
        )
        outs.append(prev)
    return torch.cat(outs, dim=0)


# -----------------------------------------------------------------------------
# Part model conditioning + expert pass
# -----------------------------------------------------------------------------
class HemisphereConditioner(nn.Module):
    """
    Embedding-based conditioner for hemisphere IDs:
      0 -> left
      1 -> right (canonicalized to left orientation before model input)
    """

    def __init__(self, n_parts: int = 2, d_model: int = 64, p_drop: float = 0.1):
        super().__init__()
        self.n_parts = int(n_parts)
        self.part_emb = nn.Embedding(self.n_parts, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, part_ids: torch.Tensor) -> torch.Tensor:
        part_ids = part_ids.long().view(-1)
        if torch.any(part_ids < 0) or torch.any(part_ids >= self.n_parts):
            raise ValueError(f"part_id out of range [0, {self.n_parts - 1}]")
        ctx = self.part_emb(part_ids)
        ctx = self.drop(self.norm(ctx))
        return ctx.unsqueeze(1)  # [B,1,d_model]


@torch.no_grad()
def _run_part_expert(
    *,
    x_part_masked: torch.Tensor,
    part_ae: AutoencoderKL,
    part_unet: DiffusionModelUNet,
    scheduler: DDPMScheduler,
    scale_factor: float,
    t_aux: int,
    amp_enabled: bool,
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Part expert path: encode decoded x_taux crop, then denoise t_aux -> 0 directly.
    z_taux = part_ae.encode_stage_2_inputs(x_part_masked) * scale_factor
    z_hat0 = _reverse_ddpm_to_t0(
        unet=part_unet,
        scheduler=scheduler,
        z_t=z_taux,
        t_start=int(t_aux),
        amp_enabled=amp_enabled,
        context=context,
    )
    return part_ae.decode_stage_2_outputs(z_hat0 / scale_factor)


# -----------------------------------------------------------------------------
# TAux branch: whole scaffold -> part experts -> x_coarse -> soft additive injection
# -----------------------------------------------------------------------------
@torch.no_grad()
def build_expert_composite_branch(
    *,
    whole_unet: DiffusionModelUNet,
    whole_ae: AutoencoderKL,
    hemi_ae: AutoencoderKL,
    hemi_unet: DiffusionModelUNet,
    hemi_conditioner: HemisphereConditioner,
    sub_ae: AutoencoderKL,
    sub_unet: DiffusionModelUNet,
    scheduler: DDPMScheduler,
    whole_scale_factor: float,
    hemi_scale_factor: float,
    sub_scale_factor: float,
    mask_ctx: Dict[str, torch.Tensor],
    hemi_shape: Tuple[int, int, int],
    sub_shape: Tuple[int, int, int],
    t_aux: int,
    part_taux_max: int,
    amp_enabled: bool,
    scaffold_mode: str,
    z_whole0_ref: torch.Tensor,
    compose_flip_lhemi: bool,
    compose_flip_rhemi: bool,
    compose_valid_thresh: float,
    bg_value: float,
    inject_alpha: float,
):
    """
    Build expert composite and latent injection ingredients at a shared t_aux.

    scaffold_mode:
      - "sample": sample whole z from noise down to t_aux.
      - "forward_noise": use q(z_whole0_ref, t_aux) as a cheaper scaffold.
    """
    batch_size = z_whole0_ref.shape[0]
    latent_shape = tuple(z_whole0_ref.shape[1:])
    t_part_aux = int(t_aux)
    if int(part_taux_max) >= 0:
        t_part_aux = min(t_part_aux, int(part_taux_max))

    if scaffold_mode == "sample":
        z_whole_taux = _reverse_ddpm_from_T_to_taux(
            unet=whole_unet,
            scheduler=scheduler,
            batch_size=batch_size,
            latent_shape=latent_shape,
            t_aux=t_aux,
            device=z_whole0_ref.device,
            amp_enabled=amp_enabled,
        )
    elif scaffold_mode == "forward_noise":
        t_batch = torch.full((batch_size,), int(t_aux), device=z_whole0_ref.device, dtype=torch.long)
        z_whole_taux = scheduler.add_noise(
            original_samples=z_whole0_ref,
            noise=torch.randn_like(z_whole0_ref),
            timesteps=t_batch,
        )
    else:
        raise ValueError(f"Unknown scaffold_mode: {scaffold_mode}")

    x_whole_taux = whole_ae.decode_stage_2_outputs(z_whole_taux / whole_scale_factor)

    # Build part inputs from decoded whole scaffold at t_aux:
    # apply full-volume template masks first, then crop to part shapes.
    mL_world = mask_ctx["mL"].expand(batch_size, -1, -1, -1, -1)
    mR_world = mask_ctx["mR"].expand(batch_size, -1, -1, -1, -1)
    mS_world = mask_ctx["mS"].expand(batch_size, -1, -1, -1, -1)

    x_lhemi_whole_masked = _apply_binary_mask_bg(x_whole_taux, mL_world, bg_value)
    x_rhemi_whole_masked = _apply_binary_mask_bg(x_whole_taux, mR_world, bg_value)
    x_sub_whole_masked = _apply_binary_mask_bg(x_whole_taux, mS_world, bg_value)

    # With current template placement, lhemi occupies high-index x and rhemi low-index x.
    x_lhemi_in_masked = _crop_right(x_lhemi_whole_masked, hemi_shape)
    x_rhemi_masked_raw = _crop_left(x_rhemi_whole_masked, hemi_shape)
    x_rhemi_in_masked = _flip_lr(x_rhemi_masked_raw)
    x_sub_in_masked = _crop_sub(x_sub_whole_masked, sub_shape)

    # Run frozen part experts once at shared t_aux.
    left_ids = torch.zeros((batch_size,), device=z_whole0_ref.device, dtype=torch.long)
    right_ids = torch.ones((batch_size,), device=z_whole0_ref.device, dtype=torch.long)
    ctx_left = hemi_conditioner(left_ids)
    ctx_right = hemi_conditioner(right_ids)

    x_lhemi_hat = _run_part_expert(
        x_part_masked=x_lhemi_in_masked,
        part_ae=hemi_ae,
        part_unet=hemi_unet,
        scheduler=scheduler,
        scale_factor=hemi_scale_factor,
        t_aux=t_part_aux,
        amp_enabled=amp_enabled,
        context=ctx_left,
    )
    x_rhemi_hat_leftcanon = _run_part_expert(
        x_part_masked=x_rhemi_in_masked,
        part_ae=hemi_ae,
        part_unet=hemi_unet,
        scheduler=scheduler,
        scale_factor=hemi_scale_factor,
        t_aux=t_part_aux,
        amp_enabled=amp_enabled,
        context=ctx_right,
    )
    x_sub_hat = _run_part_expert(
        x_part_masked=x_sub_in_masked,
        part_ae=sub_ae,
        part_unet=sub_unet,
        scheduler=scheduler,
        scale_factor=sub_scale_factor,
        t_aux=t_part_aux,
        amp_enabled=amp_enabled,
        context=None,
    )

    # Compose from image-space part assets directly:
    # left side <- lhemi, right side <- rhemi (right-world), sub <- sub.
    x_lhemi_comp = x_lhemi_hat
    x_rhemi_hat_right = _flip_lr(x_rhemi_hat_leftcanon)
    x_rhemi_comp = x_rhemi_hat_right

    x_coarse = _compose_from_parts_with_unknown_overlap(
        whole_shape=tuple(x_whole_taux.shape),
        x_lhemi=x_lhemi_comp,
        x_rhemi_right=x_rhemi_comp,
        x_sub=x_sub_hat,
        mL_world=mask_ctx["mL"].expand(batch_size, -1, -1, -1, -1),
        mR_world=mask_ctx["mR"].expand(batch_size, -1, -1, -1, -1),
        mS_world=mask_ctx["mS"].expand(batch_size, -1, -1, -1, -1),
        # Use part assets directly without extra compose-time flips.
        flip_lhemi_valid=False,
        flip_rhemi_valid=False,
        valid_thresh=compose_valid_thresh,
        bg_value=bg_value,
    )

    z_coarse0 = whole_ae.encode_stage_2_inputs(x_coarse) * whole_scale_factor
    t_batch = torch.full((batch_size,), int(t_aux), device=z_whole0_ref.device, dtype=torch.long)
    z_coarse_taux = scheduler.add_noise(
        original_samples=z_coarse0,
        noise=torch.randn_like(z_coarse0),
        timesteps=t_batch,
    )

    mask_lat = downsample_mask_any_voxel(
        mask_ctx["m_known"].expand(batch_size, -1, -1, -1, -1),
        latent_shape=tuple(z_coarse0.shape[-3:]),
    )

    # Seam band in image space: regions present in whole scaffold but absent in coarse composition.
    m_head = (x_whole_taux > (bg_value + 0.05)).float()
    m_coarse = (x_coarse > (bg_value + 0.05)).float()
    m_seam = torch.clamp(m_head - m_coarse, min=0.0, max=1.0)
    m_seam = (m_seam > 0.5).float()
    m_seam_lat = downsample_mask_any_voxel(m_seam, latent_shape=tuple(z_coarse0.shape[-3:]))

    # Soft additive injection guidance (ablation):
    # z_mix = z_whole + alpha * mask * (z_coarse - z_whole)
    # alpha=1 recovers hard replacement over masked regions.
    z_inject_delta = mask_lat * (z_coarse_taux - z_whole_taux)
    z_mix_taux = z_whole_taux + float(inject_alpha) * z_inject_delta

    return {
        "z_whole_taux": z_whole_taux,
        "x_whole_taux": x_whole_taux,
        "x_lhemi_in_masked": x_lhemi_in_masked,
        "x_rhemi_in_masked": x_rhemi_in_masked,
        "x_sub_in_masked": x_sub_in_masked,
        "x_lhemi_hat": x_lhemi_comp,
        "x_rhemi_hat_leftcanon": x_rhemi_hat_leftcanon,
        "x_rhemi_hat_right": x_rhemi_hat_right,
        "x_sub_hat": x_sub_hat,
        "x_coarse": x_coarse,
        "z_coarse0": z_coarse0,
        "z_mix_taux": z_mix_taux,
        "z_inject_delta": z_inject_delta,
        "mask_lat": mask_lat,
        "m_seam_lat": m_seam_lat,
        "t_aux": int(t_aux),
        "t_part_aux": int(t_part_aux),
    }


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------
def _extract_affine_from_tensor(x: torch.Tensor) -> np.ndarray:
    affine = np.eye(4, dtype=np.float32)
    if hasattr(x, "affine"):
        aff = x.affine
        if isinstance(aff, torch.Tensor):
            aff = aff.detach().cpu().numpy()
        aff = np.asarray(aff)
        if aff.ndim == 3:
            aff = aff[0]
        if aff.shape == (4, 4):
            affine = aff.astype(np.float32)
    return affine


def _save_nii_from_tensor(x: torch.Tensor, affine: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = x.detach().cpu().float().numpy()
    nib.save(nib.Nifti1Image(arr, affine), out_path)


@torch.no_grad()
def eval_ldm_loss_only(
    *,
    whole_unet: DiffusionModelUNet,
    whole_ae: AutoencoderKL,
    val_loader,
    device,
    whole_scale_factor: float,
    torch_autocast: bool,
    val_batches: int = 4,
):
    whole_unet.eval()
    whole_ae.eval()

    ddpm = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    amp_enabled = bool(torch_autocast and device.type == "cuda")

    val_loss = 0.0
    n_seen = 0
    for i, batch in enumerate(val_loader):
        if i >= val_batches:
            break
        images = batch["image"].to(device)
        z = whole_ae.encode_stage_2_inputs(images) * whole_scale_factor
        noise = torch.randn_like(z)
        t = torch.randint(0, ddpm.num_train_timesteps, (images.shape[0],), device=device).long()
        z_t = ddpm.add_noise(original_samples=z, noise=noise, timesteps=t)
        with autocast(device_type="cuda", enabled=amp_enabled):
            eps = whole_unet(z_t, timesteps=t)
            loss = F.mse_loss(eps.float(), noise.float())
        val_loss += float(loss.item())
        n_seen += 1
    return {"val_loss_simple": val_loss / max(1, n_seen)}


@torch.no_grad()
def eval_taux_fast(
    *,
    whole_unet: DiffusionModelUNet,
    whole_ae: AutoencoderKL,
    hemi_ae: AutoencoderKL,
    hemi_unet: DiffusionModelUNet,
    hemi_conditioner: HemisphereConditioner,
    sub_ae: AutoencoderKL,
    sub_unet: DiffusionModelUNet,
    val_loader,
    train_loader,
    device,
    whole_scale_factor: float,
    hemi_scale_factor: float,
    sub_scale_factor: float,
    mask_ctx: Dict[str, torch.Tensor],
    hemi_shape: Tuple[int, int, int],
    sub_shape: Tuple[int, int, int],
    taux_min: int,
    taux_max: int,
    part_taux_max: int,
    scaffold_mode: str,
    compose_flip_lhemi: bool,
    compose_flip_rhemi: bool,
    compose_valid_thresh: float,
    bg_value: float,
    inject_alpha: float,
    torch_autocast: bool,
    outdir: str,
    global_step: int,
    eval_n: int,
    val_batches: int,
):
    whole_unet.eval()
    whole_ae.eval()
    hemi_ae.eval()
    hemi_unet.eval()
    hemi_conditioner.eval()
    sub_ae.eval()
    sub_unet.eval()

    ddpm = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    amp_enabled = bool(torch_autocast and device.type == "cuda")

    val_loss = 0.0
    n_seen = 0
    for i, batch in enumerate(val_loader):
        if i >= val_batches:
            break
        images = batch["image"].to(device)
        z = whole_ae.encode_stage_2_inputs(images) * whole_scale_factor
        noise = torch.randn_like(z)
        t = torch.randint(0, ddpm.num_train_timesteps, (images.shape[0],), device=device).long()
        z_t = ddpm.add_noise(original_samples=z, noise=noise, timesteps=t)
        with autocast(device_type="cuda", enabled=amp_enabled):
            eps = whole_unet(z_t, timesteps=t)
            loss = F.mse_loss(eps.float(), noise.float())
        val_loss += float(loss.item())
        n_seen += 1

    # Reference sample for affine + latent shape.
    try:
        ref_batch = first(val_loader)
    except Exception:
        ref_batch = first(train_loader)
    ref_affine = _extract_affine_from_tensor(ref_batch["image"])
    ref_img = ref_batch["image"].to(device)[:1]
    z_ref0 = whole_ae.encode_stage_2_inputs(ref_img) * whole_scale_factor

    step_dir = os.path.join(outdir, "eval_samples", f"step{global_step:09d}")
    os.makedirs(step_dir, exist_ok=True)

    for i in range(eval_n):
        t_aux = random.randint(int(taux_min), int(taux_max))
        payload = build_expert_composite_branch(
            whole_unet=whole_unet,
            whole_ae=whole_ae,
            hemi_ae=hemi_ae,
            hemi_unet=hemi_unet,
            hemi_conditioner=hemi_conditioner,
            sub_ae=sub_ae,
            sub_unet=sub_unet,
            scheduler=ddpm,
            whole_scale_factor=whole_scale_factor,
            hemi_scale_factor=hemi_scale_factor,
            sub_scale_factor=sub_scale_factor,
            mask_ctx=mask_ctx,
            hemi_shape=hemi_shape,
            sub_shape=sub_shape,
            t_aux=t_aux,
            part_taux_max=part_taux_max,
            amp_enabled=amp_enabled,
            scaffold_mode=scaffold_mode,
            z_whole0_ref=z_ref0,
            compose_flip_lhemi=compose_flip_lhemi,
            compose_flip_rhemi=compose_flip_rhemi,
            compose_valid_thresh=compose_valid_thresh,
            bg_value=bg_value,
            inject_alpha=inject_alpha,
        )

        z_final0 = _reverse_ddpm_to_t0(
            unet=whole_unet,
            scheduler=ddpm,
            z_t=payload["z_mix_taux"],
            t_start=t_aux,
            amp_enabled=amp_enabled,
            context=None,
        )
        x_final = whole_ae.decode_stage_2_outputs(z_final0 / whole_scale_factor)

        sample_dir = os.path.join(step_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Official outputs.
        _save_nii_from_tensor(payload["x_lhemi_hat"][0, 0], ref_affine, os.path.join(sample_dir, "lhemi.nii.gz"))
        _save_nii_from_tensor(payload["x_rhemi_hat_right"][0, 0], ref_affine, os.path.join(sample_dir, "rhemi.nii.gz"))
        _save_nii_from_tensor(payload["x_sub_hat"][0, 0], ref_affine, os.path.join(sample_dir, "sub.nii.gz"))
        _save_nii_from_tensor(x_final[0, 0], ref_affine, os.path.join(sample_dir, "whole_brain.nii.gz"))

        # Required debug outputs.
        _save_nii_from_tensor(payload["x_whole_taux"][0, 0], ref_affine, os.path.join(sample_dir, "x_whole_taux.nii.gz"))
        _save_nii_from_tensor(payload["x_lhemi_in_masked"][0, 0], ref_affine, os.path.join(sample_dir, "x_lhemi_in_masked.nii.gz"))
        _save_nii_from_tensor(payload["x_rhemi_in_masked"][0, 0], ref_affine, os.path.join(sample_dir, "x_rhemi_in_masked.nii.gz"))
        _save_nii_from_tensor(payload["x_sub_in_masked"][0, 0], ref_affine, os.path.join(sample_dir, "x_sub_in_masked.nii.gz"))
        _save_nii_from_tensor(payload["x_coarse"][0, 0], ref_affine, os.path.join(sample_dir, "x_coarse.nii.gz"))

        torch.save(
            {
                "global_step": global_step,
                "sample_index": i,
                "t_aux": int(t_aux),
                "t_part_aux": int(payload["t_part_aux"]),
                "scaffold_mode": scaffold_mode,
                "rhemi_saved_flip_from_leftcanon": True,
                "compose_mode": "image_space_direct_parts_soft_add",
                "compose_assignment": "left=rhemi,right=lhemi,sub=sub",
                "inject_alpha": float(inject_alpha),
            },
            os.path.join(sample_dir, "meta.pt"),
        )

    return {"val_loss": val_loss / max(1, n_seen)}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
@torch.no_grad()
def _estimate_scale_factor_from_batch(
    *,
    ae: AutoencoderKL,
    x: torch.Tensor,
    amp_enabled: bool,
) -> float:
    with autocast(device_type="cuda", enabled=amp_enabled):
        z = ae.encode_stage_2_inputs(x)
    return float(1.0 / max(torch.std(z).item(), 1e-8))


def train_ldm_aux_taux(
    *,
    whole_unet: DiffusionModelUNet,
    whole_ae: AutoencoderKL,
    hemi_ae: AutoencoderKL,
    hemi_unet: DiffusionModelUNet,
    hemi_conditioner: HemisphereConditioner,
    sub_ae: AutoencoderKL,
    sub_unet: DiffusionModelUNet,
    train_loader,
    val_loader,
    mask_ctx: Dict[str, torch.Tensor],
    hemi_shape: Tuple[int, int, int],
    sub_shape: Tuple[int, int, int],
    max_steps: int,
    lr: float,
    whole_scale_factor: Optional[float],
    hemi_scale_factor: Optional[float],
    sub_scale_factor: Optional[float],
    lambda_aux: float,
    lambda_seam: float,
    aux_every: int,
    taux_min: int,
    taux_max: int,
    part_taux_max: int,
    scaffold_mode: str,
    eval_scaffold_mode: str,
    compose_flip_lhemi: bool,
    compose_flip_rhemi: bool,
    compose_valid_thresh: float,
    bg_value: float,
    inject_alpha: float,
    torch_autocast: bool,
    device,
    outdir: str,
    ckpt_every: int,
    last_every: int,
    eval_every: int,
    simple_eval_every: int,
    eval_n: int,
    eval_val_batches: int,
    simple_eval_val_batches: int,
    resume_ckpt: str,
):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )

    # Freeze all non-trainable modules.
    for m in [whole_ae, hemi_ae, hemi_unet, hemi_conditioner, sub_ae, sub_unet]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    amp_enabled = bool(torch_autocast and device.type == "cuda")
    aux_every = max(1, int(aux_every))

    # Scale factors.
    if whole_scale_factor is None:
        with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
            z = whole_ae.encode_stage_2_inputs(first(train_loader)["image"].to(device))
        whole_scale_factor = float(1.0 / max(torch.std(z).item(), 1e-8))
        print(f"[scale] whole_scale_factor auto={whole_scale_factor:.6f}")

    if hemi_scale_factor is None or sub_scale_factor is None:
        ref_img = first(train_loader)["image"].to(device)[:1]
        xL_whole_masked = _apply_binary_mask_bg(ref_img, mask_ctx["mL"], bg_value)
        xR_whole_masked = _apply_binary_mask_bg(ref_img, mask_ctx["mR"], bg_value)
        xS_whole_masked = _apply_binary_mask_bg(ref_img, mask_ctx["mS"], bg_value)
        xL = _crop_right(xL_whole_masked, hemi_shape)
        xR_masked_raw = _crop_left(xR_whole_masked, hemi_shape)
        xR = _flip_lr(xR_masked_raw)
        xS = _crop_sub(xS_whole_masked, sub_shape)

        if hemi_scale_factor is None:
            hemi_scale_factor = _estimate_scale_factor_from_batch(
                ae=hemi_ae,
                x=torch.cat([xL, xR], dim=0),
                amp_enabled=amp_enabled,
            )
            print(f"[scale] hemi_scale_factor auto={hemi_scale_factor:.6f}")
        if sub_scale_factor is None:
            sub_scale_factor = _estimate_scale_factor_from_batch(ae=sub_ae, x=xS, amp_enabled=amp_enabled)
            print(f"[scale] sub_scale_factor auto={sub_scale_factor:.6f}")

    optimizer = torch.optim.AdamW(whole_unet.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler("cuda", enabled=amp_enabled)

    global_step = 0
    start_epoch = 0
    history = {
        "train_loss": [],
        "simple_eval": [],
        "eval": [],
        "loss_curve": [],
        "epoch_loss_curve": [],
    }

    if resume_ckpt:
        global_step, start_epoch, extra = _load_ckpt_into_unet(
            whole_unet,
            resume_ckpt,
            device,
            opt=optimizer,
            scaler=scaler,
        )
        if isinstance(extra, dict):
            history = extra.get("history", history)
            if "whole_scale_factor" in extra:
                whole_scale_factor = float(extra["whole_scale_factor"])
            if "hemi_scale_factor" in extra:
                hemi_scale_factor = float(extra["hemi_scale_factor"])
            if "sub_scale_factor" in extra:
                sub_scale_factor = float(extra["sub_scale_factor"])
        if not isinstance(history.get("loss_curve"), list):
            history["loss_curve"] = []
        if not isinstance(history.get("epoch_loss_curve"), list):
            history["epoch_loss_curve"] = []
        print(
            "[resume] "
            f"global_step={global_step}, start_epoch~={start_epoch}, "
            f"sf_whole={whole_scale_factor:.6f}, sf_hemi={hemi_scale_factor:.6f}, sf_sub={sub_scale_factor:.6f}"
        )

    if global_step >= max_steps:
        print(f"[resume] global_step ({global_step}) >= max_steps ({max_steps}), nothing to train.")
        return

    whole_unet.train()
    os.makedirs(outdir, exist_ok=True)

    data_iter = iter(train_loader)
    steps_per_epoch = max(1, int(len(train_loader)))

    def _update_epoch_loss_curve():
        raw = history.get("loss_curve", [])
        epoch_curve = []
        for i in range(0, len(raw), steps_per_epoch):
            chunk = raw[i : i + steps_per_epoch]
            if chunk:
                epoch_curve.append(float(sum(chunk) / len(chunk)))
        history["epoch_loss_curve"] = epoch_curve

    running = 0.0
    running_n = 0

    pbar = tqdm(range(global_step, max_steps), ncols=130)
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
            z_whole0 = whole_ae.encode_stage_2_inputs(images) * whole_scale_factor

        # Standard DDPM objective on whole-brain latent.
        noise = torch.randn_like(z_whole0)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
        z_t = scheduler.add_noise(original_samples=z_whole0, noise=noise, timesteps=timesteps)
        with autocast(device_type="cuda", enabled=amp_enabled):
            eps_pred = whole_unet(z_t, timesteps=timesteps)
            loss_std = F.mse_loss(eps_pred.float(), noise.float())

        do_aux = (aux_every <= 1) or (step % aux_every == 0)
        if do_aux:
            # Shared t_aux for this optimization step.
            t_aux = random.randint(int(taux_min), int(taux_max))

            # Build expert composite branch and z_coarse0 for aux guidance.
            with torch.no_grad():
                payload = build_expert_composite_branch(
                    whole_unet=whole_unet,
                    whole_ae=whole_ae,
                    hemi_ae=hemi_ae,
                    hemi_unet=hemi_unet,
                    hemi_conditioner=hemi_conditioner,
                    sub_ae=sub_ae,
                    sub_unet=sub_unet,
                    scheduler=scheduler,
                    whole_scale_factor=whole_scale_factor,
                    hemi_scale_factor=hemi_scale_factor,
                    sub_scale_factor=sub_scale_factor,
                    mask_ctx=mask_ctx,
                    hemi_shape=hemi_shape,
                    sub_shape=sub_shape,
                    t_aux=t_aux,
                    part_taux_max=part_taux_max,
                    amp_enabled=amp_enabled,
                    scaffold_mode=scaffold_mode,
                    z_whole0_ref=z_whole0,
                    compose_flip_lhemi=compose_flip_lhemi,
                    compose_flip_rhemi=compose_flip_rhemi,
                    compose_valid_thresh=compose_valid_thresh,
                    bg_value=bg_value,
                    inject_alpha=inject_alpha,
                )
                z_coarse0 = payload["z_coarse0"]
                mask_lat = payload["mask_lat"]
                m_seam_lat = payload["m_seam_lat"]

            # Auxiliary latent trajectory loss only for t <= t_aux.
            active = timesteps <= int(t_aux)
            if bool(active.any().item()):
                z_pred_tm1 = _predict_prev_from_eps_ddpm(
                    scheduler=scheduler,
                    z_t=z_t,
                    eps_pred=eps_pred,
                    timesteps=timesteps,
                )
                t_prev = torch.clamp(timesteps - 1, min=0)
                z_coarse_tm1 = scheduler.add_noise(
                    original_samples=z_coarse0,
                    noise=torch.randn_like(z_coarse0),
                    timesteps=t_prev,
                )
                z_whole_tm1 = scheduler.add_noise(
                    original_samples=z_whole0,
                    noise=torch.randn_like(z_whole0),
                    timesteps=t_prev,
                )

                m = mask_lat
                if m.shape[0] != z_pred_tm1.shape[0]:
                    m = m.expand(z_pred_tm1.shape[0], -1, -1, -1, -1)
                diff = (z_pred_tm1[active] - z_coarse_tm1[active]) * m[active]
                denom = m[active].sum() * z_pred_tm1.shape[1] + 1e-6
                loss_aux = diff.float().pow(2).sum() / denom.float()

                m_seam = m_seam_lat
                if m_seam.shape[0] != z_pred_tm1.shape[0]:
                    m_seam = m_seam.expand(z_pred_tm1.shape[0], -1, -1, -1, -1)
                diff_seam = (z_pred_tm1[active] - z_whole_tm1[active]) * m_seam[active]
                denom_seam = m_seam[active].sum() * z_pred_tm1.shape[1] + 1e-6
                loss_seam = diff_seam.float().pow(2).sum() / denom_seam.float()
            else:
                loss_aux = torch.zeros((), device=device, dtype=loss_std.dtype)
                loss_seam = torch.zeros((), device=device, dtype=loss_std.dtype)
        else:
            t_aux = -1
            loss_aux = torch.zeros((), device=device, dtype=loss_std.dtype)
            loss_seam = torch.zeros((), device=device, dtype=loss_std.dtype)

        loss = loss_std + float(lambda_aux) * loss_aux + float(lambda_seam) * loss_seam

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step = step + 1

        loss_item = float(loss.item())
        running += loss_item
        running_n += 1
        avg_loss = running / max(1, running_n)
        history["loss_curve"].append(loss_item)

        epoch = int(max(0, global_step - 1) // steps_per_epoch)
        pbar.set_description(f"step {global_step}/{max_steps} (ep~{epoch})")
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.6f}",
                "std": f"{float(loss_std.item()):.5f}",
                "aux": f"{float(loss_aux.item()):.5f}",
                "seam": f"{float(loss_seam.item()):.5f}",
                "taux": "-" if t_aux < 0 else int(t_aux),
                "aux_step": int(do_aux),
                "sf": f"{whole_scale_factor:.4f}",
            }
        )

        if global_step % 200 == 0:
            history["train_loss"].append(
                {
                    "step": global_step,
                    "loss": avg_loss,
                    "loss_std": float(loss_std.item()),
                    "loss_aux": float(loss_aux.item()),
                    "loss_seam": float(loss_seam.item()),
                }
            )

        if (last_every > 0) and (global_step % last_every == 0):
            _update_epoch_loss_curve()
            _save_ckpt(
                os.path.join(outdir, "UNET_last.pt"),
                whole_unet,
                optimizer,
                scaler,
                global_step=global_step,
                epoch=epoch,
                extra={
                    "whole_scale_factor": float(whole_scale_factor),
                    "hemi_scale_factor": float(hemi_scale_factor),
                    "sub_scale_factor": float(sub_scale_factor),
                    "history": history,
                },
            )
            plot_unet_loss(
                history["epoch_loss_curve"],
                title=f"UNET Epoch-Average Loss_step{global_step}",
                outdir=outdir,
                filename="UNET_loss.png",
            )

        if (ckpt_every > 0) and (global_step % ckpt_every == 0):
            _update_epoch_loss_curve()
            ckpt_path = os.path.join(outdir, f"UNET_step{global_step:09d}.pt")
            _save_ckpt(
                ckpt_path,
                whole_unet,
                optimizer,
                scaler,
                global_step=global_step,
                epoch=epoch,
                extra={
                    "whole_scale_factor": float(whole_scale_factor),
                    "hemi_scale_factor": float(hemi_scale_factor),
                    "sub_scale_factor": float(sub_scale_factor),
                    "history": history,
                },
            )

        if val_loader is not None and len(val_loader) > 0 and (eval_every > 0) and (global_step % eval_every == 0):
            metrics = eval_taux_fast(
                whole_unet=whole_unet,
                whole_ae=whole_ae,
                hemi_ae=hemi_ae,
                hemi_unet=hemi_unet,
                hemi_conditioner=hemi_conditioner,
                sub_ae=sub_ae,
                sub_unet=sub_unet,
                val_loader=val_loader,
                train_loader=train_loader,
                device=device,
                whole_scale_factor=whole_scale_factor,
                hemi_scale_factor=hemi_scale_factor,
                sub_scale_factor=sub_scale_factor,
                mask_ctx=mask_ctx,
                hemi_shape=hemi_shape,
                sub_shape=sub_shape,
                taux_min=taux_min,
                taux_max=taux_max,
                part_taux_max=part_taux_max,
                scaffold_mode=eval_scaffold_mode,
                compose_flip_lhemi=compose_flip_lhemi,
                compose_flip_rhemi=compose_flip_rhemi,
                compose_valid_thresh=compose_valid_thresh,
                bg_value=bg_value,
                inject_alpha=inject_alpha,
                torch_autocast=torch_autocast,
                outdir=outdir,
                global_step=global_step,
                eval_n=eval_n,
                val_batches=eval_val_batches,
            )
            history["eval"].append({"step": global_step, **metrics})
            whole_unet.train()
        elif val_loader is not None and len(val_loader) > 0 and (simple_eval_every > 0) and (global_step % simple_eval_every == 0):
            metrics = eval_ldm_loss_only(
                whole_unet=whole_unet,
                whole_ae=whole_ae,
                val_loader=val_loader,
                device=device,
                whole_scale_factor=whole_scale_factor,
                torch_autocast=torch_autocast,
                val_batches=simple_eval_val_batches,
            )
            history["simple_eval"].append({"step": global_step, **metrics})
            whole_unet.train()

    _update_epoch_loss_curve()
    final_path = os.path.join(outdir, "UNET_last.pt")
    _save_ckpt(
        final_path,
        whole_unet,
        optimizer,
        scaler,
        global_step=global_step,
        epoch=int(global_step // max(1, steps_per_epoch)),
        extra={
            "whole_scale_factor": float(whole_scale_factor),
            "hemi_scale_factor": float(hemi_scale_factor),
            "sub_scale_factor": float(sub_scale_factor),
            "history": history,
        },
    )
    plot_unet_loss(
        history["epoch_loss_curve"],
        title=f"UNET Epoch-Average Loss_step{global_step}",
        outdir=outdir,
        filename="UNET_loss.png",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file with argument defaults.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Train whole-brain 3D LDM with Aux-tAux soft additive injection guidance.",
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
    ap.add_argument("--torch_autocast", default=True)
    ap.add_argument("--n_samples", default="ALL")

    # Template masks
    ap.add_argument("--template_dir", default="data/template")
    ap.add_argument("--lhemi_template_name", default="lhemi_mask.nii.gz")
    ap.add_argument("--rhemi_template_name", default="rhemi_mask.nii.gz")
    ap.add_argument("--sub_template_name", default="sub_mask.nii.gz")

    # Whole model checkpoints
    ap.add_argument("--whole_ae_ckpt", default="", help="Required: whole-brain AE checkpoint.")
    ap.add_argument("--whole_unet_ckpt", default="", help="Required warm-start for trainable whole UNet.")
    ap.add_argument("--resume_ckpt", default="", help="Optional packaged UNET_step*.pt for exact resume.")

    # Hemi expert checkpoints
    ap.add_argument("--hemi_ae_ckpt", default="", help="Required: hemi AE checkpoint.")
    ap.add_argument("--hemi_unet_ckpt", default="", help="Required: hemi UNet checkpoint.")
    ap.add_argument(
        "--hemi_conditioner_ckpt",
        default="",
        help="Optional conditioner checkpoint. If empty, tries to load from hemi_unet_ckpt conditioner_state_dict.",
    )

    # Sub expert checkpoints
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

    # Aux-tAux training controls
    ap.add_argument("--max_steps", type=int, default=120_000)
    ap.add_argument("--ldm_lr", type=float, default=1e-4)
    ap.add_argument("--lambda_aux", type=float, default=1.0)
    ap.add_argument(
        "--lambda_seam",
        type=float,
        default=0.0,
        help="Weight for seam-band latent loss in m_seam_lat (toward whole latent trajectory).",
    )
    ap.add_argument("--aux_every", type=int, default=1, help="Run expensive TAux branch every N steps.")
    ap.add_argument("--taux_min", type=int, default=50)
    ap.add_argument("--taux_max", type=int, default=200)
    ap.add_argument(
        "--part_taux_max",
        type=int,
        default=-1,
        help="If >=0, cap part-expert denoising start t by min(t_aux, part_taux_max).",
    )
    ap.add_argument(
        "--scaffold_mode",
        choices=["sample", "forward_noise"],
        default="sample",
        help="How to obtain z_whole_taux for TAux scaffold during training.",
    )
    ap.add_argument(
        "--eval_scaffold_mode",
        default="",
        help="Optional eval scaffold mode override ('sample' or 'forward_noise'). "
        "If empty, uses --scaffold_mode.",
    )

    # Compose controls (for data orientation differences).
    ap.add_argument("--compose_flip_lhemi", default=False)
    ap.add_argument("--compose_flip_rhemi", default=False)
    ap.add_argument("--compose_valid_thresh", type=float, default=-0.995)
    ap.add_argument("--bg_value", type=float, default=-1.0)
    ap.add_argument(
        "--inject_alpha",
        type=float,
        default=0.35,
        help="Soft additive injection strength for z_mix_taux: z + alpha * mask * (z_coarse - z).",
    )

    # Step-based logging / eval
    ap.add_argument("--ckpt_every", type=int, default=10_000)
    ap.add_argument("--last_every", type=int, default=1_000)
    ap.add_argument("--eval_every", type=int, default=10_000)
    ap.add_argument("--simple_eval_every", type=int, default=2_000)
    ap.add_argument("--eval_n", type=int, default=2)
    ap.add_argument("--eval_val_batches", type=int, default=8)
    ap.add_argument("--simple_eval_val_batches", type=int, default=4)

    ap.add_argument("--outdir", default="ckpts/UNET")
    ap.add_argument("--out_prefix", default="whole_brain_aux_taux_add_UNET")
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

    required_ckpts = [
        ("whole_ae_ckpt", args.whole_ae_ckpt),
        ("whole_unet_ckpt", args.whole_unet_ckpt),
        ("hemi_ae_ckpt", args.hemi_ae_ckpt),
        ("hemi_unet_ckpt", args.hemi_unet_ckpt),
        ("sub_ae_ckpt", args.sub_ae_ckpt),
        ("sub_unet_ckpt", args.sub_unet_ckpt),
    ]
    for key, path in required_ckpts:
        if not path and not (key == "whole_unet_ckpt" and args.resume_ckpt):
            ap.error(f"--{key} is required.")

    if int(args.taux_min) < 0 or int(args.taux_max) >= 1000 or int(args.taux_min) > int(args.taux_max):
        ap.error("Require 0 <= taux_min <= taux_max < 1000.")
    if int(args.part_taux_max) >= 1000:
        ap.error("Require part_taux_max < 1000 (or -1 to disable cap).")
    if int(args.aux_every) < 1:
        ap.error("Require aux_every >= 1.")
    valid_scaffold = {"sample", "forward_noise"}
    if str(args.scaffold_mode) not in valid_scaffold:
        ap.error("Require scaffold_mode in {'sample','forward_noise'}.")
    if str(args.eval_scaffold_mode).strip() != "" and str(args.eval_scaffold_mode) not in valid_scaffold:
        ap.error("Require eval_scaffold_mode in {'sample','forward_noise'} (or empty).")
    if float(args.inject_alpha) < 0.0:
        ap.error("Require inject_alpha >= 0.")
    if float(args.lambda_seam) < 0.0:
        ap.error("Require lambda_seam >= 0.")

    os.makedirs(args.outdir, exist_ok=True)
    experiment_dir = os.path.join(args.outdir, f"{args.out_prefix}_{args.out_postfix}")
    os.makedirs(experiment_dir, exist_ok=True)
    args.outdir = experiment_dir
    print(f"\nOutput dir: {args.outdir}\n")

    cfg = vars(args).copy()
    with open(os.path.join(args.outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    spacing = _parse_float_tuple(args.spacing)
    size = _parse_int_tuple(args.size)
    hemi_size = _parse_int_tuple(args.hemi_size)
    sub_size = _parse_int_tuple(args.sub_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_all(1017)

    n_samples = None if str(args.n_samples).upper() == "ALL" else int(args.n_samples)
    train_transforms = build_image_transforms(spacing=spacing, whole_size=size)
    train_loader, val_loader, _ = make_dataloaders_from_csv(
        csv_path=args.csv,
        image_key=args.whole_key,
        conditions=("age", "sex", "group", 'condition'),
        train_transforms=train_transforms,
        n_samples=n_samples,
        data_split_json_path=args.data_split_json_path,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    template_dir = args.template_dir
    lhemi_mask_path = _resolve_path(template_dir, args.lhemi_template_name)
    rhemi_mask_path = _resolve_path(template_dir, args.rhemi_template_name)
    sub_mask_path = _resolve_path(template_dir, args.sub_template_name)
    whole_shape = tuple(int(v) for v in first(train_loader)["image"].shape[-3:])
    mask_ctx = load_template_masks(
        lhemi_mask_path=lhemi_mask_path,
        rhemi_mask_path=rhemi_mask_path,
        sub_mask_path=sub_mask_path,
        spacing=spacing,
        whole_shape=whole_shape,
        device=device,
    )

    # Build models.
    whole_ae = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=_parse_int_tuple(args.whole_ae_num_channels),
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
        num_channels=_parse_int_tuple(args.whole_ldm_num_channels),
        attention_levels=(False, True, True),
        num_head_channels=_parse_int_tuple(args.whole_ldm_num_head_channels),
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
        num_channels=_parse_int_tuple(args.hemi_ae_num_channels),
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
        num_channels=_parse_int_tuple(args.hemi_ldm_num_channels),
        attention_levels=(False, True, True),
        num_head_channels=_parse_int_tuple(args.hemi_ldm_num_head_channels),
        with_conditioning=True,
        cross_attention_dim=int(args.hemi_cond_dim),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        upcast_attention=True,
        use_flash_attention=False,
    ).to(device)
    hemi_conditioner = HemisphereConditioner(
        n_parts=int(args.hemi_n_parts),
        d_model=int(args.hemi_cond_dim),
    ).to(device)

    sub_ae = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=_parse_int_tuple(args.sub_ae_num_channels),
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
        num_channels=_parse_int_tuple(args.sub_ldm_num_channels),
        attention_levels=(False, True, True),
        num_head_channels=_parse_int_tuple(args.sub_ldm_num_head_channels),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        upcast_attention=True,
        use_flash_attention=False,
    ).to(device)

    # Load checkpoints.
    _load_ckpt_into_ae(whole_ae, args.whole_ae_ckpt, device)
    if args.whole_unet_ckpt:
        _load_model_weights_shape_safe(whole_unet, args.whole_unet_ckpt, device)

    _load_ckpt_into_ae(hemi_ae, args.hemi_ae_ckpt, device)
    _load_model_weights_shape_safe(hemi_unet, args.hemi_unet_ckpt, device)
    if args.hemi_conditioner_ckpt:
        _load_conditioner_weights_from_ckpt(hemi_conditioner, args.hemi_conditioner_ckpt, device)
    else:
        _load_conditioner_weights_from_ckpt(hemi_conditioner, args.hemi_unet_ckpt, device)

    _load_ckpt_into_ae(sub_ae, args.sub_ae_ckpt, device)
    _load_model_weights_shape_safe(sub_unet, args.sub_unet_ckpt, device)

    # Scale factors: args override > ckpt extra > auto estimate in train loop.
    whole_sf = args.whole_scale_factor if args.whole_scale_factor > 0 else None
    hemi_sf = args.hemi_scale_factor if args.hemi_scale_factor > 0 else None
    sub_sf = args.sub_scale_factor if args.sub_scale_factor > 0 else None
    if whole_sf is None:
        for p in [args.whole_unet_ckpt, args.resume_ckpt]:
            if p:
                whole_sf = _read_scale_factor_from_ckpt(p, device)
                if whole_sf is not None:
                    break
    if hemi_sf is None:
        hemi_sf = _read_scale_factor_from_ckpt(args.hemi_unet_ckpt, device)
    if sub_sf is None:
        sub_sf = _read_scale_factor_from_ckpt(args.sub_unet_ckpt, device)

    train_ldm_aux_taux(
        whole_unet=whole_unet,
        whole_ae=whole_ae,
        hemi_ae=hemi_ae,
        hemi_unet=hemi_unet,
        hemi_conditioner=hemi_conditioner,
        sub_ae=sub_ae,
        sub_unet=sub_unet,
        train_loader=train_loader,
        val_loader=val_loader,
        mask_ctx=mask_ctx,
        hemi_shape=hemi_size,
        sub_shape=sub_size,
        max_steps=int(args.max_steps),
        lr=float(args.ldm_lr),
        whole_scale_factor=whole_sf,
        hemi_scale_factor=hemi_sf,
        sub_scale_factor=sub_sf,
        lambda_aux=float(args.lambda_aux),
        lambda_seam=float(args.lambda_seam),
        aux_every=int(args.aux_every),
        taux_min=int(args.taux_min),
        taux_max=int(args.taux_max),
        part_taux_max=int(args.part_taux_max),
        scaffold_mode=str(args.scaffold_mode),
        eval_scaffold_mode=(
            str(args.eval_scaffold_mode)
            if str(args.eval_scaffold_mode).strip() != ""
            else str(args.scaffold_mode)
        ),
        compose_flip_lhemi=str2bool(args.compose_flip_lhemi),
        compose_flip_rhemi=str2bool(args.compose_flip_rhemi),
        compose_valid_thresh=float(args.compose_valid_thresh),
        bg_value=float(args.bg_value),
        inject_alpha=float(args.inject_alpha),
        torch_autocast=str2bool(args.torch_autocast),
        device=device,
        outdir=args.outdir,
        ckpt_every=int(args.ckpt_every),
        last_every=int(args.last_every),
        eval_every=int(args.eval_every),
        simple_eval_every=int(args.simple_eval_every),
        eval_n=int(args.eval_n),
        eval_val_batches=int(args.eval_val_batches),
        simple_eval_val_batches=int(args.simple_eval_val_batches),
        resume_ckpt=args.resume_ckpt,
    )


if __name__ == "__main__":
    main()
