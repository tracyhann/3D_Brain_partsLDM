import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Tuple
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.metrics.regression import compute_ms_ssim, compute_ssim_and_cs
from torch import nn

from eval_utils import fid_from_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDICALNET_PROJECT_CKPT_DIR = os.path.join(PROJECT_ROOT, "ckpts", "medicalnet")
MEDICALNET_HF_DEFAULT_URLS: Dict[str, str] = {
    "medicalnet_resnet10_23datasets": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/main/resnet_10_23dataset.pth",
    "medicalnet_resnet50_23datasets": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50_23dataset.pth",
}
MEDICALNET_CKPT_NAMES: Dict[str, str] = {
    "medicalnet_resnet10_23datasets": "resnet_10_23dataset.pth",
    "medicalnet_resnet50_23datasets": "resnet_50_23dataset.pth",
}


def _canonical_hf_url(url: str) -> str:
    u = str(url).strip()
    if not u:
        return ""
    if "huggingface.co" in u and "/blob/" in u:
        u = u.replace("/blob/", "/resolve/")
    if "huggingface.co" in u and "/resolve/" in u and "download=true" not in u:
        sep = "&" if "?" in u else "?"
        u = f"{u}{sep}download=true"
    return u


def _seed_medicalnet_ckpt_from_hf(
    *,
    model_name: str,
    hf_url: str,
    force_download: bool,
    dst_dir: str,
) -> str:
    ckpt_name = MEDICALNET_CKPT_NAMES.get(str(model_name), "")
    url = _canonical_hf_url(str(hf_url).strip())
    if not url:
        url = _canonical_hf_url(MEDICALNET_HF_DEFAULT_URLS.get(str(model_name), ""))
    if not ckpt_name and url:
        ckpt_name = os.path.basename(url.split("?", 1)[0])
    if not ckpt_name or not url:
        return ""

    ckpt_dir = os.path.abspath(str(dst_dir).strip())
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, ckpt_name)

    if os.path.exists(dst) and not bool(force_download):
        return dst

    tmp = f"{dst}.tmp"
    print(f"Downloading MedicalNet checkpoint from HuggingFace: {url}")
    with urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)
    print(f"Saved MedicalNet checkpoint: {dst}")
    return dst


def _resolve_project_medicalnet_ckpt(
    *,
    model_name: str,
    ckpt_path: str,
    ckpt_dir: str,
) -> str:
    model = str(model_name).strip()
    ckpt_name = MEDICALNET_CKPT_NAMES.get(model, "")
    candidates: List[str] = []

    p = str(ckpt_path).strip()
    if p:
        candidates.append(p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p))

    d = str(ckpt_dir).strip()
    if d and ckpt_name:
        base_dir = d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)
        candidates.append(os.path.join(base_dir, ckpt_name))

    if ckpt_name:
        candidates.append(os.path.join(MEDICALNET_PROJECT_CKPT_DIR, ckpt_name))
        candidates.append(os.path.join(PROJECT_ROOT, "ckpts", ckpt_name))

    seen = set()
    for cand in candidates:
        if not cand:
            continue
        c = os.path.abspath(cand)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    return ""


def _seed_medicalnet_ckpt_to_torch_hub(
    *,
    model_name: str,
    local_ckpt_path: str,
    force: bool,
) -> str:
    src = os.path.abspath(str(local_ckpt_path).strip())
    if not src or not os.path.isfile(src):
        return ""

    ckpt_name = MEDICALNET_CKPT_NAMES.get(str(model_name), "").strip() or os.path.basename(src)
    hub_dir = torch.hub.get_dir()
    hub_ckpt_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(hub_ckpt_dir, exist_ok=True)
    dst = os.path.join(hub_ckpt_dir, ckpt_name)

    if os.path.exists(dst) and not bool(force):
        return dst
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst

    shutil.copy2(src, dst)
    return dst


def _normalize_image_id(v) -> str:
    s = str(v).strip()
    if s.endswith(".0"):
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
    return s


def _strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _parse_image_id_from_generated_name(path: str) -> str:
    """
    Parse image_id from generated filename pattern: *_[image_id].nii.gz
    """
    stem = _strip_nii_ext(os.path.basename(path))
    # Canonical naming from infer_3d_brain_ldm_steps.py:
    # sample_{idx}_{safe_image_id}.nii.gz
    m = re.match(r"^sample_\d+_(.+)$", stem)
    if m:
        return m.group(1).strip()
    if "_" not in stem:
        return stem.strip()
    return stem.rsplit("_", 1)[-1].strip()


def _safe_component(x) -> str:
    s = str(x)
    s = re.sub(r"[^\w.\-]+", "-", s)
    s = s.strip("-")
    return s if s else "unknown"


def _resolve_existing_path(path: str, generated_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path

    candidates = [
        path,
        os.path.join(PROJECT_ROOT, path),
        os.path.join(generated_dir, path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return path


def _load_generated_args(generated_dir: str) -> Dict:
    args_path = os.path.join(generated_dir, "args.json")
    if not os.path.exists(args_path):
        return {}
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _build_test_id_to_real_path(
    *,
    csv_path: str,
    data_split_json_path: str,
    image_key: str,
    generated_dir: str,
) -> Tuple[Dict[str, str], str]:
    """
    Reuses the split logic used in standard train LDM scripts:
    select rows whose imageID is in test split.
    """
    with open(data_split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = {_normalize_image_id(v) for v in splits["test"]}
    df = pd.read_csv(csv_path)

    cols = set(df.columns.tolist())
    chosen_image_key = str(image_key).strip()
    if chosen_image_key not in cols:
        # Common schemas in this repo.
        for fallback in ("image", "whole_brain"):
            if fallback in cols:
                chosen_image_key = fallback
                break
    if chosen_image_key not in cols:
        raise ValueError(
            f"Could not resolve image column. Requested='{image_key}'. "
            f"Available columns: {sorted(cols)}"
        )

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        image_id = _normalize_image_id(row["imageID"])
        if image_id not in test_ids:
            continue
        real_path = _resolve_existing_path(str(row[chosen_image_key]), generated_dir)
        out[image_id] = real_path
    return out, chosen_image_key


def _find_generated_files(generated_dir: str) -> List[str]:
    files: List[str] = []
    for fn in sorted(os.listdir(generated_dir)):
        if fn.endswith(".nii.gz") or fn.endswith(".nii"):
            files.append(os.path.join(generated_dir, fn))
    return files


def _pair_generated_with_test_real(
    *,
    generated_files: List[str],
    test_id_to_real_path: Dict[str, str],
) -> Tuple[List[Dict[str, str]], List[str], List[str], List[str]]:
    pairs = []
    missing_in_test = []
    unparsable = []
    duplicate_ids = []

    safe_id_to_test_ids: Dict[str, List[str]] = {}
    for test_id in test_id_to_real_path.keys():
        safe_id = _normalize_image_id(_safe_component(test_id))
        safe_id_to_test_ids.setdefault(safe_id, []).append(test_id)

    def _resolve_to_test_id(gen_path: str) -> str:
        # Try direct parse first.
        parsed = _parse_image_id_from_generated_name(gen_path)
        candidates = [parsed]

        stem = _strip_nii_ext(os.path.basename(gen_path))
        if stem not in candidates:
            candidates.append(stem)
        if "_" in stem:
            tail = stem.rsplit("_", 1)[-1].strip()
            if tail not in candidates:
                candidates.append(tail)

        for cand in candidates:
            nid = _normalize_image_id(cand)
            if nid in test_id_to_real_path:
                return nid
            safe_hits = safe_id_to_test_ids.get(nid, [])
            if len(safe_hits) == 1:
                return safe_hits[0]

        # Fallback: match by longest safe-id suffix (handles IDs with underscores).
        hits = []
        for safe_id, test_ids in safe_id_to_test_ids.items():
            if stem == safe_id or stem.endswith(f"_{safe_id}"):
                for tid in test_ids:
                    hits.append((len(safe_id), tid))
        if not hits:
            return ""
        hits = sorted(hits, key=lambda x: x[0], reverse=True)
        best_len = hits[0][0]
        best_ids = sorted({tid for l, tid in hits if l == best_len})
        if len(best_ids) == 1:
            return best_ids[0]
        return ""

    seen = set()
    for fpath in generated_files:
        parsed_image_id = _parse_image_id_from_generated_name(fpath)
        image_id = _resolve_to_test_id(fpath)
        if not image_id:
            if parsed_image_id:
                missing_in_test.append(parsed_image_id)
            else:
                unparsable.append(fpath)
            continue

        if image_id in seen:
            duplicate_ids.append(image_id)
            continue
        seen.add(image_id)

        if image_id not in test_id_to_real_path:
            missing_in_test.append(image_id)
            continue

        pairs.append(
            {
                "image_id": image_id,
                "fake": os.path.abspath(fpath),
                "real": test_id_to_real_path[image_id],
            }
        )

    pairs = sorted(pairs, key=lambda x: x["image_id"])
    return pairs, missing_in_test, unparsable, duplicate_ids


def _build_pair_transforms(
    crop_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float] | None,
):
    def _pick_channel0(x):
        # x from LoadImaged + EnsureChannelFirstd can be [C,D,H,W] or [D,H,W]
        if x.ndim == 4:
            return x[0]
        return x

    keys = ["real", "fake"]
    tx = [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Lambdad(keys=keys, func=_pick_channel0),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
    ]

    # Match infer_3d_brain_ldm_steps preprocessing to ensure real/fake alignment.
    if spacing is not None:
        tx.append(transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", "bilinear")))
    tx.extend(
        [
            transforms.DivisiblePadd(keys=keys, k=32, mode="constant", constant_values=-1.0),
            # Guarantee fixed tensor shape for pairwise metrics.
            transforms.SpatialPadd(keys=keys, spatial_size=crop_size, mode="constant", constant_values=-1.0),
            transforms.CenterSpatialCropd(keys=keys, roi_size=crop_size),
            # Disable MONAI MetaTensor tracking after spatial ops.
            transforms.EnsureTyped(keys=keys, track_meta=False),
        ]
    )

    return transforms.Compose(tx)


def _build_mask_transform(
    crop_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float] | None,
):
    def _pick_channel0(x):
        if x.ndim == 4:
            return x[0]
        return x

    tx = [
        transforms.LoadImaged(keys=["mask"]),
        transforms.EnsureChannelFirstd(keys=["mask"]),
        transforms.Lambdad(keys=["mask"], func=_pick_channel0),
        transforms.EnsureChannelFirstd(keys=["mask"], channel_dim="no_channel"),
    ]
    if spacing is not None:
        tx.append(transforms.Spacingd(keys=["mask"], pixdim=spacing, mode="nearest"))
    tx.extend(
        [
            transforms.DivisiblePadd(keys=["mask"], k=32, mode="constant", constant_values=0.0),
            transforms.SpatialPadd(keys=["mask"], spatial_size=crop_size, mode="constant", constant_values=0.0),
            transforms.CenterSpatialCropd(keys=["mask"], roi_size=crop_size),
            transforms.EnsureTyped(keys=["mask"], track_meta=False),
        ]
    )
    return transforms.Compose(tx)


def _load_aligned_mask(
    *,
    mask_path: str,
    crop_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float] | None,
    device: torch.device,
) -> torch.Tensor:
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    tx = _build_mask_transform(crop_size=crop_size, spacing=spacing)
    out = tx({"mask": mask_path})
    mask = out["mask"]
    if hasattr(mask, "as_tensor"):
        mask = mask.as_tensor()
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        mask = mask.unsqueeze(0)
    if mask.ndim != 5:
        raise ValueError(f"Expected mask to become [1,1,D,H,W], got {tuple(mask.shape)} from {mask_path}")
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    return (mask > 0.5).float().to(device)


def _expand_mask_to_batch(mask: torch.Tensor, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    m = mask.to(device=device, dtype=dtype)
    if m.shape[0] == 1 and batch_size > 1:
        m = m.expand(batch_size, -1, -1, -1, -1)
    if m.shape[0] != batch_size:
        raise ValueError(f"Mask batch size mismatch: mask={m.shape[0]}, batch={batch_size}")
    return m


def _apply_mask_with_bg(x: torch.Tensor, mask: torch.Tensor, *, bg_value: float = -1.0) -> torch.Tensor:
    m = (mask > 0.5).to(dtype=x.dtype, device=x.device)
    return x * m + float(bg_value) * (1.0 - m)


def _bbox_from_mask(mask: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
    # mask expected [1,1,D,H,W] (or broadcastable with batch=1).
    if mask.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(mask.shape)}")
    m = (mask[0, 0] > 0.5)
    nz = torch.nonzero(m, as_tuple=False)
    if nz.numel() == 0:
        d, h, w = int(mask.shape[2]), int(mask.shape[3]), int(mask.shape[4])
        return 0, d, 0, h, 0, w
    d0 = int(nz[:, 0].min().item())
    d1 = int(nz[:, 0].max().item()) + 1
    h0 = int(nz[:, 1].min().item())
    h1 = int(nz[:, 1].max().item()) + 1
    w0 = int(nz[:, 2].min().item())
    w1 = int(nz[:, 2].max().item()) + 1
    return d0, d1, h0, h1, w0, w1


def _crop_to_bbox(x: torch.Tensor, bbox: Tuple[int, int, int, int, int, int]) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
    d0, d1, h0, h1, w0, w1 = bbox
    return x[:, :, d0:d1, h0:h1, w0:w1]


def _masked_mean_per_sample(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.ndim != x.ndim:
        raise ValueError(f"Mask dim mismatch: x={tuple(x.shape)}, mask={tuple(mask.shape)}")

    if mask.shape[-3:] != x.shape[-3:] and x.ndim == 5:
        mask = F.interpolate(mask.float(), size=x.shape[-3:], mode="nearest")

    if mask.shape != x.shape:
        if mask.shape[0] != x.shape[0]:
            raise ValueError(f"Mask batch mismatch: x={tuple(x.shape)}, mask={tuple(mask.shape)}")
        if mask.shape[1] == 1 and x.shape[1] != 1:
            mask = mask.expand(-1, x.shape[1], -1, -1, -1)
        if mask.shape != x.shape:
            raise ValueError(f"Mask shape mismatch: x={tuple(x.shape)}, mask={tuple(mask.shape)}")
    b = x.shape[0]
    xf = x.reshape(b, -1)
    mf = mask.reshape(b, -1).to(dtype=x.dtype)
    denom = mf.sum(dim=1).clamp_min(eps)
    return (xf * mf).sum(dim=1) / denom


def _compute_masked_ssim3d_values(
    *,
    y_pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    data_range: float,
    kernel_type: str,
    kernel_size: int,
    kernel_sigma: float,
) -> torch.Tensor:
    ssim_map, _ = compute_ssim_and_cs(
        y_pred=y_pred,
        y=y,
        spatial_dims=3,
        data_range=float(data_range),
        kernel_type=str(kernel_type).lower(),
        kernel_size=(int(kernel_size), int(kernel_size), int(kernel_size)),
        kernel_sigma=(float(kernel_sigma), float(kernel_sigma), float(kernel_sigma)),
    )
    return _masked_mean_per_sample(ssim_map, mask)


def _compute_masked_ms_ssim_values(
    *,
    y_pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    data_range: float,
    kernel_type: str,
    kernel_size: int,
    kernel_sigma: float,
    weights: Tuple[float, ...],
) -> torch.Tensor:
    x_i = y_pred
    y_i = y
    m_i = (mask > 0.5).float()
    w = tuple(float(v) for v in weights)
    if len(w) == 0:
        raise ValueError("MS-SSIM weights must be non-empty.")

    cs_vals: List[torch.Tensor] = []
    ssim_last: torch.Tensor | None = None
    for level in range(len(w)):
        ssim_map, cs_map = compute_ssim_and_cs(
            y_pred=x_i,
            y=y_i,
            spatial_dims=3,
            data_range=float(data_range),
            kernel_type=str(kernel_type).lower(),
            kernel_size=(int(kernel_size), int(kernel_size), int(kernel_size)),
            kernel_sigma=(float(kernel_sigma), float(kernel_sigma), float(kernel_sigma)),
        )
        ssim_level = _masked_mean_per_sample(ssim_map, m_i)
        ssim_last = ssim_level
        if level < len(w) - 1:
            cs_level = _masked_mean_per_sample(cs_map, m_i)
            cs_vals.append(cs_level)
            x_i = F.avg_pool3d(x_i, kernel_size=2, stride=2)
            y_i = F.avg_pool3d(y_i, kernel_size=2, stride=2)
            m_i = (F.max_pool3d(m_i, kernel_size=2, stride=2) > 0.0).float()

    assert ssim_last is not None
    out = torch.ones_like(ssim_last)
    for i, wi in enumerate(w[:-1]):
        out = out * torch.clamp(cs_vals[i], min=1e-8).pow(float(wi))
    out = out * torch.clamp(ssim_last, min=1e-8).pow(float(w[-1]))
    return out


def _linear_mmd(feats_real: torch.Tensor, feats_fake: torch.Tensor) -> float:
    """
    Linear-kernel MMD^2 in feature space.
    """
    x = feats_real.float()
    y = feats_fake.float()
    xx = torch.mm(x, x.t()).mean()
    yy = torch.mm(y, y.t()).mean()
    xy = torch.mm(x, y.t()).mean()
    return float((xx + yy - 2.0 * xy).item())


def _parse_float_tuple(v: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(v).split(",") if str(x).strip())
    if len(vals) == 0:
        raise ValueError(f"Expected non-empty float tuple string, got: {v}")
    return vals


def _renorm_weights(weights: Tuple[float, ...]) -> Tuple[float, ...]:
    s = float(sum(weights))
    if s <= 0:
        raise ValueError(f"MS-SSIM weights must sum to > 0, got: {weights}")
    return tuple(float(w / s) for w in weights)


def normalize_mri(vol, mask, pct=(1, 99)):
    """
    vol: np.ndarray (any dtype)
    mask: boolean array same shape as vol (optional)
    robust: if True, use percentiles within mask instead of global min/max
    pct: percentile bounds (low, high) for robust scaling
    """
    x = vol.astype(np.float32)
    mask = mask.astype(bool)

    vals = x[mask] if mask.any() else x.ravel()
    lo, hi = np.percentile(vals, [pct[0], pct[1]])
    # avoid degeneracy
    if hi <= lo:
        lo, hi = x.min(), x.max()

    v = np.clip(x, lo, hi)
    v = (v - lo) / (hi - lo + 1e-8)  # -> [0,1]
    v = v * 2.0 - 1.0  # -> [-1,1]

    return v


def _normalize_volume_minus1_1(
    x: torch.Tensor,
    *,
    q_low: float,
    q_high: float,
    head_mask_threshold: float = -0.95,
) -> torch.Tensor:
    # Per-volume robust min-max to [-1,1], computed within derived head mask only.
    # Voxels outside the head mask are forced to -1.
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
    if not (0.0 <= q_low < q_high <= 100.0):
        raise ValueError(f"Invalid percentile range: ({q_low}, {q_high})")

    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    out_np = np.full_like(x_np, -1.0, dtype=np.float32)
    pct = (float(q_low), float(q_high))

    bsz, ch = x_np.shape[:2]
    for bi in range(bsz):
        for ci in range(ch):
            vol = x_np[bi, ci]
            head_mask = vol > float(head_mask_threshold)
            norm_vol = normalize_mri(vol, head_mask, pct=pct)
            out_np[bi, ci] = np.where(head_mask, norm_vol, -1.0).astype(np.float32)

    return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)


class _InceptionV3Features(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "torchvision with Inception-V3 is required for paper-style FID/MMD. "
                "Install torchvision or use --dist_feature_mode pooled."
            ) from e

        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Identity()
        model.AuxLogits = None
        model.eval()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


class _MedicalNet3DFeatures(nn.Module):
    """
    3D feature extractor backed by MedicalNet models from:
    https://github.com/warvito/MedicalNet-models

    Input expected as [B,1,D,H,W].
    """

    def __init__(self, *, net: str, verbose: bool = False) -> None:
        super().__init__()
        # Keep compatibility with the same loading approach used in evaluation_ARC.py.
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "warvito/MedicalNet-models",
            model=str(net),
            verbose=bool(verbose),
            trust_repo=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"MedicalNet expects [B,1,D,H,W], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            # Keep behavior explicit if someone passes multi-channel tensors.
            x = x[:, :1]

        # Follow Med3D-style normalization per sample volume.
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std

        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        # Convert to [B,D] feature vectors for MMD.
        if out.ndim == 5:
            out = F.adaptive_avg_pool3d(out, output_size=1).flatten(1)
        elif out.ndim > 2:
            out = out.flatten(1)
        return out


def _parse_slice_axes(v: str) -> Tuple[int, ...]:
    mapping = {"sag": 2, "cor": 3, "ax": 4}
    out: List[int] = []
    for tok in str(v).split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in mapping:
            raise ValueError(f"Unknown slice axis '{t}'. Use comma list from: sag,cor,ax")
        ax = mapping[t]
        if ax not in out:
            out.append(ax)
    if not out:
        raise ValueError(f"No valid slice axes parsed from: {v}")
    return tuple(out)


def _choose_slice_indices(size: int, n_slices: int, margin: int) -> List[int]:
    if size <= 0:
        return [0]
    m = max(0, int(margin))
    start = m
    end = max(start + 1, size - m)
    if end <= start:
        start, end = 0, size
    span = max(1, end - start)
    n = max(1, int(n_slices))
    if n >= span:
        return list(range(start, end))
    idx = torch.linspace(start, end - 1, steps=n).round().long().unique().tolist()
    if len(idx) == 0:
        idx = [start + span // 2]
    return [int(i) for i in idx]


def _volume_to_slice_rgb01(
    vol: torch.Tensor,
    axis: int,
    slice_idx: int,
    *,
    clamp01: bool = True,
) -> torch.Tensor:
    # vol: [B,1,D,H,W], expected near [-1,1]
    if axis == 2:
        x = vol[:, :, slice_idx, :, :]
    elif axis == 3:
        x = vol[:, :, :, slice_idx, :]
    elif axis == 4:
        x = vol[:, :, :, :, slice_idx]
    else:
        raise ValueError(f"axis must be one of (2,3,4), got {axis}")
    x = (x + 1.0) * 0.5
    if clamp01:
        x = x.clamp(0.0, 1.0)
    return x.repeat(1, 3, 1, 1).contiguous()


def _volume_to_slice_2d(vol: torch.Tensor, axis: int, slice_idx: int) -> torch.Tensor:
    # vol: [B,1,D,H,W] -> [B,1,H,W] for the selected axis/slice.
    if axis == 2:
        return vol[:, :, slice_idx, :, :]
    if axis == 3:
        return vol[:, :, :, slice_idx, :]
    if axis == 4:
        return vol[:, :, :, :, slice_idx]
    raise ValueError(f"axis must be one of (2,3,4), got {axis}")


def _rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_features: int,
    sigma: float,
) -> Tuple[float, float]:
    # Returns (mmd2, sigma_used)
    x = x.float()
    y = y.float()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"MMD inputs must be [N,D], got {tuple(x.shape)} and {tuple(y.shape)}")
    n = min(x.shape[0], y.shape[0], int(max_features))
    if n < 2:
        return float("nan"), float("nan")

    if x.shape[0] != n:
        x = x[:n]
    if y.shape[0] != n:
        y = y[:n]

    z = torch.cat([x, y], dim=0)
    sigma_used = float(sigma)
    if sigma_used <= 0:
        zz = z
        if zz.shape[0] > 1024:
            zz = zz[:1024]
        d2 = torch.cdist(zz, zz, p=2).pow(2)
        vals = d2[d2 > 0]
        sigma2 = float(vals.median().item()) if vals.numel() > 0 else 1.0
        sigma_used = float(max(1e-8, sigma2 ** 0.5))
    sigma2 = max(1e-12, sigma_used * sigma_used)

    kxx = torch.exp(-torch.cdist(x, x, p=2).pow(2) / (2.0 * sigma2))
    kyy = torch.exp(-torch.cdist(y, y, p=2).pow(2) / (2.0 * sigma2))
    kxy = torch.exp(-torch.cdist(x, y, p=2).pow(2) / (2.0 * sigma2))

    # Unbiased MMD^2
    n_f = float(n)
    kxx_sum = kxx.sum() - kxx.diag().sum()
    kyy_sum = kyy.sum() - kyy.diag().sum()
    term_xx = kxx_sum / (n_f * (n_f - 1.0))
    term_yy = kyy_sum / (n_f * (n_f - 1.0))
    term_xy = kxy.mean()
    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return float(max(0.0, mmd2.item())), sigma_used


def run_pairwise_eval(
    *,
    pairs: List[Dict[str, str]],
    crop_size: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    feature_pool: Tuple[int, int, int],
    spacing: Tuple[float, float, float] | None,
    dist_feature_mode: str,
    fid_slice_axes: Tuple[int, ...],
    fid_slices_per_axis: int,
    fid_slice_margin: int,
    mmd_kernel: str,
    mmd_rbf_sigma: float,
    mmd_max_features: int,
    intensity_mode: str,
    norm_percentiles: Tuple[float, float],
    ssim2d_data_range: float,
    ssim2d_kernel_type: str,
    ssim2d_kernel_size: int,
    ssim2d_kernel_sigma: float,
    ms_kernel_type: str,
    ms_kernel_size: int,
    ms_kernel_sigma: float,
    ms_data_range: float,
    ms_weights: Tuple[float, ...],
    seam_mask_path: str,
    lhemi_mask_path: str,
    rhemi_mask_path: str,
    sub_mask_path: str,
    use_medicalnet_mmd3d: bool,
    medicalnet_model: str,
    medicalnet_ckpt_path: str,
    medicalnet_ckpt_dir: str,
    medicalnet_hf_url: str,
    medicalnet_hf_force_download: bool,
    medicalnet_verbose: bool,
    medicalnet_strict: bool,
    device: torch.device,
) -> Dict[str, float]:
    if len(pairs) == 0:
        raise RuntimeError("No valid real/fake pairs were found.")

    ds = Dataset(data=pairs, transform=_build_pair_transforms(crop_size=crop_size, spacing=spacing))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Eval dataset size (paired): {len(ds)}")
    first_item = ds[0]
    print(f"Transformed real shape: {tuple(first_item['real'].shape)}")
    print(f"Transformed fake shape: {tuple(first_item['fake'].shape)}")

    region_names = ("lhemi", "rhemi", "sub")
    template_masks = {
        "seam": _load_aligned_mask(mask_path=seam_mask_path, crop_size=crop_size, spacing=spacing, device=device),
        "lhemi": _load_aligned_mask(mask_path=lhemi_mask_path, crop_size=crop_size, spacing=spacing, device=device),
        "rhemi": _load_aligned_mask(mask_path=rhemi_mask_path, crop_size=crop_size, spacing=spacing, device=device),
        "sub": _load_aligned_mask(mask_path=sub_mask_path, crop_size=crop_size, spacing=spacing, device=device),
    }
    for name, m in template_masks.items():
        print(f"Loaded template mask '{name}': shape={tuple(m.shape)}, voxels={int((m > 0.5).sum().item())}")
    seam_bbox = _bbox_from_mask(template_masks["seam"])
    region_bboxes: Dict[str, Tuple[int, int, int, int, int, int]] = {
        region: _bbox_from_mask(template_masks[region]) for region in region_names
    }
    sd0, sd1, sh0, sh1, sw0, sw1 = seam_bbox
    print(f"Region bbox 'seam': D[{sd0}:{sd1}] H[{sh0}:{sh1}] W[{sw0}:{sw1}]")
    for region, bbox in region_bboxes.items():
        d0, d1, h0, h1, w0, w1 = bbox
        print(f"Region bbox '{region}': D[{d0}:{d1}] H[{h0}:{h1}] W[{w0}:{w1}]")

    ms_ssim_values: List[float] = []
    seam_ssim_values: List[float] = []
    seam_ms_ssim_values: List[float] = []
    region_ms_ssim_values: Dict[str, List[float]] = {k: [] for k in region_names}
    ssim2d_values: List[float] = []
    ssim2d_axis_values: Dict[int, List[float]] = {a: [] for a in fid_slice_axes}
    feats_real: List[torch.Tensor] = []
    feats_fake: List[torch.Tensor] = []
    seam_feats_real: List[torch.Tensor] = []
    seam_feats_fake: List[torch.Tensor] = []
    region_feats_real: Dict[str, List[torch.Tensor]] = {k: [] for k in region_names}
    region_feats_fake: Dict[str, List[torch.Tensor]] = {k: [] for k in region_names}
    n_vox = 0
    n_real_oob = 0
    n_fake_oob = 0
    n_real_changed = 0
    n_fake_changed = 0
    dist_feature_mode = str(dist_feature_mode).lower().strip()
    if dist_feature_mode not in {"pooled", "inception2d"}:
        raise ValueError(f"Unsupported --dist_feature_mode={dist_feature_mode}")
    mmd_kernel = str(mmd_kernel).lower().strip()
    if mmd_kernel not in {"linear", "rbf"}:
        raise ValueError(f"Unsupported --mmd_kernel={mmd_kernel}")
    intensity_mode = str(intensity_mode).lower().strip()
    if intensity_mode not in {"clamp", "raw", "normalize_fake", "normalize_both"}:
        raise ValueError(f"Unsupported --intensity_mode={intensity_mode}")
    ssim2d_kernel_type = str(ssim2d_kernel_type).lower().strip()
    if ssim2d_kernel_type not in {"gaussian", "uniform"}:
        raise ValueError(f"Unsupported --ssim2d_kernel_type={ssim2d_kernel_type}")

    feat_net = None
    if dist_feature_mode == "inception2d":
        feat_net = _InceptionV3Features().to(device)
        feat_net.eval()
    med3d_net = None
    med3d_init_error = ""
    med3d_project_ckpt_path = ""
    med3d_project_ckpt_error = ""
    med3d_hub_seed_path = ""
    med3d_hub_seed_error = ""
    med3d_hf_seed_path = ""
    med3d_hf_seed_error = ""
    if bool(use_medicalnet_mmd3d):
        med3d_project_ckpt_path = _resolve_project_medicalnet_ckpt(
            model_name=str(medicalnet_model),
            ckpt_path=str(medicalnet_ckpt_path),
            ckpt_dir=str(medicalnet_ckpt_dir),
        )
        if med3d_project_ckpt_path:
            print(f"MedicalNet project checkpoint found: {med3d_project_ckpt_path}")

        try:
            if not med3d_project_ckpt_path:
                med3d_hf_seed_path = _seed_medicalnet_ckpt_from_hf(
                    model_name=str(medicalnet_model),
                    hf_url=str(medicalnet_hf_url),
                    force_download=bool(medicalnet_hf_force_download),
                    dst_dir=str(medicalnet_ckpt_dir),
                )
                if med3d_hf_seed_path:
                    med3d_project_ckpt_path = med3d_hf_seed_path
                    print(f"MedicalNet checkpoint downloaded to project ckpts: {med3d_hf_seed_path}")
        except Exception as e:
            med3d_hf_seed_error = f"{type(e).__name__}: {e}"
            print(f"WARNING: MedicalNet HF pre-download failed: {med3d_hf_seed_error}")

        try:
            if med3d_project_ckpt_path:
                med3d_hub_seed_path = _seed_medicalnet_ckpt_to_torch_hub(
                    model_name=str(medicalnet_model),
                    local_ckpt_path=med3d_project_ckpt_path,
                    force=bool(medicalnet_hf_force_download),
                )
                if med3d_hub_seed_path:
                    print(f"MedicalNet checkpoint seeded to torch hub cache: {med3d_hub_seed_path}")
            else:
                med3d_project_ckpt_error = (
                    f"No local checkpoint found for model '{medicalnet_model}' in "
                    f"ckpt_path='{medicalnet_ckpt_path}' or ckpt_dir='{medicalnet_ckpt_dir}'."
                )
                print(f"WARNING: {med3d_project_ckpt_error}")
        except Exception as e:
            med3d_hub_seed_error = f"{type(e).__name__}: {e}"
            print(f"WARNING: MedicalNet local->hub seeding failed: {med3d_hub_seed_error}")

        try:
            med3d_net = _MedicalNet3DFeatures(
                net=str(medicalnet_model),
                verbose=bool(medicalnet_verbose),
            ).to(device)
            med3d_net.eval()
            print(f"MedicalNet 3D MMD enabled with backbone: {medicalnet_model}")
        except Exception as e:
            med3d_init_error = f"{type(e).__name__}: {e}"
            if bool(medicalnet_strict):
                raise
            print(
                "WARNING: MedicalNet 3D init failed; continuing without MedicalNet MMD. "
                f"Error: {med3d_init_error}"
            )
            med3d_net = None

    med3d_feats_real_whole: List[torch.Tensor] = []
    med3d_feats_fake_whole: List[torch.Tensor] = []
    med3d_feats_real_seam: List[torch.Tensor] = []
    med3d_feats_fake_seam: List[torch.Tensor] = []
    med3d_feats_real_region: Dict[str, List[torch.Tensor]] = {k: [] for k in region_names}
    med3d_feats_fake_region: Dict[str, List[torch.Tensor]] = {k: [] for k in region_names}

    def _plain_tensor(x: torch.Tensor) -> torch.Tensor:
        # MONAI may yield MetaTensor; convert to plain Tensor for stable downstream ops.
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        return x

    for batch in loader:
        real_raw = _plain_tensor(batch["real"]).to(device).float()
        fake_raw = _plain_tensor(batch["fake"]).to(device).float()

        real_oob = ((real_raw < -1.0) | (real_raw > 1.0))
        fake_oob = ((fake_raw < -1.0) | (fake_raw > 1.0))

        n_vox += int(real_raw.numel())
        n_real_oob += int(real_oob.sum().item())
        n_fake_oob += int(fake_oob.sum().item())

        if intensity_mode == "clamp":
            real = real_raw.clamp(-1.0, 1.0)
            fake = fake_raw.clamp(-1.0, 1.0)
        elif intensity_mode == "raw":
            real = real_raw
            fake = fake_raw
        elif intensity_mode == "normalize_fake":
            real = real_raw
            fake = _normalize_volume_minus1_1(
                fake_raw,
                q_low=float(norm_percentiles[0]),
                q_high=float(norm_percentiles[1]),
            )
        elif intensity_mode == "normalize_both":
            real = _normalize_volume_minus1_1(
                real_raw,
                q_low=float(norm_percentiles[0]),
                q_high=float(norm_percentiles[1]),
            )
            fake = _normalize_volume_minus1_1(
                fake_raw,
                q_low=float(norm_percentiles[0]),
                q_high=float(norm_percentiles[1]),
            )
        else:
            raise RuntimeError(f"Unhandled intensity_mode={intensity_mode}")

        n_real_changed += int((real != real_raw).sum().item())
        n_fake_changed += int((fake != fake_raw).sum().item())

        bsz = int(real.shape[0])
        seam_mask_b = _expand_mask_to_batch(template_masks["seam"], bsz, device=device, dtype=real.dtype)
        seam_ssim_batch = _compute_masked_ssim3d_values(
            y_pred=fake,
            y=real,
            mask=seam_mask_b,
            data_range=float(ms_data_range),
            kernel_type=str(ms_kernel_type).lower(),
            kernel_size=int(ms_kernel_size),
            kernel_sigma=float(ms_kernel_sigma),
        )
        seam_ssim_values.extend(float(v) for v in seam_ssim_batch.detach().cpu().tolist())

        seam_ms_batch = _compute_masked_ms_ssim_values(
            y_pred=fake,
            y=real,
            mask=seam_mask_b,
            data_range=float(ms_data_range),
            kernel_type=str(ms_kernel_type).lower(),
            kernel_size=int(ms_kernel_size),
            kernel_sigma=float(ms_kernel_sigma),
            weights=ms_weights,
        )
        seam_ms_ssim_values.extend(float(v) for v in seam_ms_batch.detach().cpu().tolist())
        seam_real = _apply_mask_with_bg(real, seam_mask_b, bg_value=-1.0)
        seam_fake = _apply_mask_with_bg(fake, seam_mask_b, bg_value=-1.0)
        seam_real_crop = _crop_to_bbox(seam_real, seam_bbox)
        seam_fake_crop = _crop_to_bbox(seam_fake, seam_bbox)

        region_real: Dict[str, torch.Tensor] = {}
        region_fake: Dict[str, torch.Tensor] = {}
        region_real_crop: Dict[str, torch.Tensor] = {}
        region_fake_crop: Dict[str, torch.Tensor] = {}
        for region in region_names:
            region_mask_b = _expand_mask_to_batch(template_masks[region], bsz, device=device, dtype=real.dtype)
            region_ms_batch = _compute_masked_ms_ssim_values(
                y_pred=fake,
                y=real,
                mask=region_mask_b,
                data_range=float(ms_data_range),
                kernel_type=str(ms_kernel_type).lower(),
                kernel_size=int(ms_kernel_size),
                kernel_sigma=float(ms_kernel_sigma),
                weights=ms_weights,
            )
            region_ms_ssim_values[region].extend(float(v) for v in region_ms_batch.detach().cpu().tolist())
            region_real[region] = _apply_mask_with_bg(real, region_mask_b, bg_value=-1.0)
            region_fake[region] = _apply_mask_with_bg(fake, region_mask_b, bg_value=-1.0)
            bbox = region_bboxes[region]
            region_real_crop[region] = _crop_to_bbox(region_real[region], bbox)
            region_fake_crop[region] = _crop_to_bbox(region_fake[region], bbox)

        if med3d_net is not None:
            fr_whole = _plain_tensor(med3d_net(real.contiguous())).detach().cpu()
            ff_whole = _plain_tensor(med3d_net(fake.contiguous())).detach().cpu()
            med3d_feats_real_whole.append(fr_whole)
            med3d_feats_fake_whole.append(ff_whole)

            fr_seam = _plain_tensor(med3d_net(seam_real_crop.contiguous())).detach().cpu()
            ff_seam = _plain_tensor(med3d_net(seam_fake_crop.contiguous())).detach().cpu()
            med3d_feats_real_seam.append(fr_seam)
            med3d_feats_fake_seam.append(ff_seam)

            for region in region_names:
                fr_reg = _plain_tensor(med3d_net(region_real_crop[region].contiguous())).detach().cpu()
                ff_reg = _plain_tensor(med3d_net(region_fake_crop[region].contiguous())).detach().cpu()
                med3d_feats_real_region[region].append(fr_reg)
                med3d_feats_fake_region[region].append(ff_reg)

        # Pairwise metric: MONAI MS-SSIM per sample.
        batch_ms = compute_ms_ssim(
            y_pred=fake,
            y=real,
            spatial_dims=3,
            data_range=float(ms_data_range),
            kernel_type=str(ms_kernel_type).lower(),
            kernel_size=int(ms_kernel_size),
            kernel_sigma=float(ms_kernel_sigma),
            weights=ms_weights,
        )
        ms_vals = batch_ms.reshape(batch_ms.shape[0], -1).mean(dim=1).detach().cpu().tolist()
        ms_ssim_values.extend(float(v) for v in ms_vals)

        # Slice-wise 2D SSIM per view using the same axis/slice policy as FID.
        for axis in fid_slice_axes:
            idxs = _choose_slice_indices(size=int(real.shape[axis]), n_slices=fid_slices_per_axis, margin=fid_slice_margin)
            for s in idxs:
                real_2d = _volume_to_slice_2d(real, axis=axis, slice_idx=s)
                fake_2d = _volume_to_slice_2d(fake, axis=axis, slice_idx=s)
                ssim2d_map, _ = compute_ssim_and_cs(
                    y_pred=fake_2d,
                    y=real_2d,
                    spatial_dims=2,
                    data_range=float(ssim2d_data_range),
                    kernel_type=ssim2d_kernel_type,
                    kernel_size=(int(ssim2d_kernel_size), int(ssim2d_kernel_size)),
                    kernel_sigma=(float(ssim2d_kernel_sigma), float(ssim2d_kernel_sigma)),
                )
                vals = ssim2d_map.reshape(ssim2d_map.shape[0], -1).mean(dim=1).detach().cpu().tolist()
                ssim2d_values.extend(float(v) for v in vals)
                ssim2d_axis_values[axis].extend(float(v) for v in vals)

        if dist_feature_mode == "pooled":
            fr = F.adaptive_avg_pool3d(real.float(), output_size=feature_pool).flatten(1).cpu()
            ff = F.adaptive_avg_pool3d(fake.float(), output_size=feature_pool).flatten(1).cpu()
            feats_real.append(fr)
            feats_fake.append(ff)
            fr_seam = F.adaptive_avg_pool3d(seam_real_crop.float(), output_size=feature_pool).flatten(1).cpu()
            ff_seam = F.adaptive_avg_pool3d(seam_fake_crop.float(), output_size=feature_pool).flatten(1).cpu()
            seam_feats_real.append(fr_seam)
            seam_feats_fake.append(ff_seam)
            for region in region_names:
                fr_reg = F.adaptive_avg_pool3d(region_real_crop[region].float(), output_size=feature_pool).flatten(1).cpu()
                ff_reg = F.adaptive_avg_pool3d(region_fake_crop[region].float(), output_size=feature_pool).flatten(1).cpu()
                region_feats_real[region].append(fr_reg)
                region_feats_fake[region].append(ff_reg)
        else:
            assert feat_net is not None
            for axis in fid_slice_axes:
                idxs = _choose_slice_indices(size=int(real.shape[axis]), n_slices=fid_slices_per_axis, margin=fid_slice_margin)
                for s in idxs:
                    # Inception expects bounded [0,1] inputs.
                    real_2d = _volume_to_slice_rgb01(real, axis=axis, slice_idx=s, clamp01=True)
                    fake_2d = _volume_to_slice_rgb01(fake, axis=axis, slice_idx=s, clamp01=True)
                    fr = _plain_tensor(feat_net(real_2d)).detach().cpu()
                    ff = _plain_tensor(feat_net(fake_2d)).detach().cpu()
                    feats_real.append(fr)
                    feats_fake.append(ff)
            for axis in fid_slice_axes:
                idxs = _choose_slice_indices(
                    size=int(seam_real_crop.shape[axis]),
                    n_slices=fid_slices_per_axis,
                    margin=fid_slice_margin,
                )
                for s in idxs:
                    real_2d_seam = _volume_to_slice_rgb01(seam_real_crop, axis=axis, slice_idx=s, clamp01=True)
                    fake_2d_seam = _volume_to_slice_rgb01(seam_fake_crop, axis=axis, slice_idx=s, clamp01=True)
                    fr_seam = _plain_tensor(feat_net(real_2d_seam)).detach().cpu()
                    ff_seam = _plain_tensor(feat_net(fake_2d_seam)).detach().cpu()
                    seam_feats_real.append(fr_seam)
                    seam_feats_fake.append(ff_seam)
            for region in region_names:
                rr = region_real_crop[region]
                rf = region_fake_crop[region]
                for axis in fid_slice_axes:
                    idxs = _choose_slice_indices(size=int(rr.shape[axis]), n_slices=fid_slices_per_axis, margin=fid_slice_margin)
                    for s in idxs:
                        real_2d_reg = _volume_to_slice_rgb01(rr, axis=axis, slice_idx=s, clamp01=True)
                        fake_2d_reg = _volume_to_slice_rgb01(rf, axis=axis, slice_idx=s, clamp01=True)
                        fr_reg = _plain_tensor(feat_net(real_2d_reg)).detach().cpu()
                        ff_reg = _plain_tensor(feat_net(fake_2d_reg)).detach().cpu()
                        region_feats_real[region].append(fr_reg)
                        region_feats_fake[region].append(ff_reg)

    feats_real_t = _plain_tensor(torch.cat(feats_real, dim=0))
    feats_fake_t = _plain_tensor(torch.cat(feats_fake, dim=0))

    fid = float(fid_from_features(feats_fake=feats_fake_t, feats_real=feats_real_t))
    if mmd_kernel == "linear":
        mmd = float(_linear_mmd(feats_real_t, feats_fake_t))
        mmd_sigma_used = float("nan")
    else:
        mmd, mmd_sigma_used = _rbf_mmd2(
            feats_real_t,
            feats_fake_t,
            max_features=int(mmd_max_features),
            sigma=float(mmd_rbf_sigma),
        )

    region_metric_vals: Dict[str, float] = {}
    for region in region_names:
        feats_real_reg = _plain_tensor(torch.cat(region_feats_real[region], dim=0))
        feats_fake_reg = _plain_tensor(torch.cat(region_feats_fake[region], dim=0))
        fid_reg = float(fid_from_features(feats_fake=feats_fake_reg, feats_real=feats_real_reg))
        if mmd_kernel == "linear":
            mmd_reg = float(_linear_mmd(feats_real_reg, feats_fake_reg))
            mmd_reg_sigma = float("nan")
        else:
            mmd_reg, mmd_reg_sigma = _rbf_mmd2(
                feats_real_reg,
                feats_fake_reg,
                max_features=int(mmd_max_features),
                sigma=float(mmd_rbf_sigma),
            )
        region_metric_vals[f"fid_{region}"] = fid_reg
        region_metric_vals[f"mmd_{region}"] = float(mmd_reg)
        region_metric_vals[f"mmd_rbf_sigma_{region}"] = float(mmd_reg_sigma)
        region_metric_vals[f"feature_dim_{region}"] = int(feats_real_reg.shape[1])
        region_metric_vals[f"num_feature_vectors_{region}"] = int(feats_real_reg.shape[0])

    seam_feats_real_t = _plain_tensor(torch.cat(seam_feats_real, dim=0))
    seam_feats_fake_t = _plain_tensor(torch.cat(seam_feats_fake, dim=0))
    if mmd_kernel == "linear":
        seam_mmd = float(_linear_mmd(seam_feats_real_t, seam_feats_fake_t))
        seam_mmd_sigma = float("nan")
    else:
        seam_mmd, seam_mmd_sigma = _rbf_mmd2(
            seam_feats_real_t,
            seam_feats_fake_t,
            max_features=int(mmd_max_features),
            sigma=float(mmd_rbf_sigma),
        )

    med3d_metric_vals: Dict[str, float] = {}
    if med3d_net is not None and len(med3d_feats_real_whole) > 0:
        med3d_whole_real = _plain_tensor(torch.cat(med3d_feats_real_whole, dim=0))
        med3d_whole_fake = _plain_tensor(torch.cat(med3d_feats_fake_whole, dim=0))
        if mmd_kernel == "linear":
            med3d_mmd_whole = float(_linear_mmd(med3d_whole_real, med3d_whole_fake))
            med3d_sigma_whole = float("nan")
        else:
            med3d_mmd_whole, med3d_sigma_whole = _rbf_mmd2(
                med3d_whole_real,
                med3d_whole_fake,
                max_features=int(mmd_max_features),
                sigma=float(mmd_rbf_sigma),
            )
        med3d_metric_vals["mmd_med3d_whole"] = float(med3d_mmd_whole)
        med3d_metric_vals["mmd_rbf_sigma_med3d_whole"] = float(med3d_sigma_whole)
        med3d_metric_vals["feature_dim_med3d_whole"] = int(med3d_whole_real.shape[1])
        med3d_metric_vals["num_feature_vectors_med3d_whole"] = int(med3d_whole_real.shape[0])

        med3d_seam_real = _plain_tensor(torch.cat(med3d_feats_real_seam, dim=0))
        med3d_seam_fake = _plain_tensor(torch.cat(med3d_feats_fake_seam, dim=0))
        if mmd_kernel == "linear":
            med3d_mmd_seam = float(_linear_mmd(med3d_seam_real, med3d_seam_fake))
            med3d_sigma_seam = float("nan")
        else:
            med3d_mmd_seam, med3d_sigma_seam = _rbf_mmd2(
                med3d_seam_real,
                med3d_seam_fake,
                max_features=int(mmd_max_features),
                sigma=float(mmd_rbf_sigma),
            )
        med3d_metric_vals["mmd_med3d_seam"] = float(med3d_mmd_seam)
        med3d_metric_vals["mmd_rbf_sigma_med3d_seam"] = float(med3d_sigma_seam)
        med3d_metric_vals["feature_dim_med3d_seam"] = int(med3d_seam_real.shape[1])
        med3d_metric_vals["num_feature_vectors_med3d_seam"] = int(med3d_seam_real.shape[0])

        med3d_part_vals: List[float] = []
        for region in region_names:
            med3d_real_reg = _plain_tensor(torch.cat(med3d_feats_real_region[region], dim=0))
            med3d_fake_reg = _plain_tensor(torch.cat(med3d_feats_fake_region[region], dim=0))
            if mmd_kernel == "linear":
                med3d_mmd_reg = float(_linear_mmd(med3d_real_reg, med3d_fake_reg))
                med3d_sigma_reg = float("nan")
            else:
                med3d_mmd_reg, med3d_sigma_reg = _rbf_mmd2(
                    med3d_real_reg,
                    med3d_fake_reg,
                    max_features=int(mmd_max_features),
                    sigma=float(mmd_rbf_sigma),
                )
            med3d_metric_vals[f"mmd_med3d_{region}"] = float(med3d_mmd_reg)
            med3d_metric_vals[f"mmd_rbf_sigma_med3d_{region}"] = float(med3d_sigma_reg)
            med3d_metric_vals[f"feature_dim_med3d_{region}"] = int(med3d_real_reg.shape[1])
            med3d_metric_vals[f"num_feature_vectors_med3d_{region}"] = int(med3d_real_reg.shape[0])
            med3d_part_vals.append(float(med3d_mmd_reg))

        if len(med3d_part_vals) > 0:
            med3d_metric_vals["mmd_med3d_part_mean"] = float(sum(med3d_part_vals) / len(med3d_part_vals))

    ms_ssim_mean = float(sum(ms_ssim_values) / max(1, len(ms_ssim_values)))
    ms_ssim_std = float(pd.Series(ms_ssim_values).std(ddof=0) if len(ms_ssim_values) > 1 else 0.0)
    seam_ssim_mean = float(sum(seam_ssim_values) / max(1, len(seam_ssim_values)))
    seam_ssim_std = float(pd.Series(seam_ssim_values).std(ddof=0) if len(seam_ssim_values) > 1 else 0.0)
    seam_ms_ssim_mean = float(sum(seam_ms_ssim_values) / max(1, len(seam_ms_ssim_values)))
    seam_ms_ssim_std = float(pd.Series(seam_ms_ssim_values).std(ddof=0) if len(seam_ms_ssim_values) > 1 else 0.0)
    region_ms_metric_vals: Dict[str, float] = {}
    for region in region_names:
        vals = region_ms_ssim_values[region]
        mean_v = float(sum(vals) / max(1, len(vals)))
        std_v = float(pd.Series(vals).std(ddof=0) if len(vals) > 1 else 0.0)
        region_ms_metric_vals[f"{region}_ms_ssim_mean"] = mean_v
        region_ms_metric_vals[f"{region}_ms_ssim_std"] = std_v
    ssim2d_mean = float(sum(ssim2d_values) / max(1, len(ssim2d_values)))
    ssim2d_std = float(pd.Series(ssim2d_values).std(ddof=0) if len(ssim2d_values) > 1 else 0.0)
    axis_to_name = {2: "sag", 3: "cor", 4: "ax"}
    real_oob_frac = float(n_real_oob / max(1, n_vox))
    fake_oob_frac = float(n_fake_oob / max(1, n_vox))
    real_changed_frac = float(n_real_changed / max(1, n_vox))
    fake_changed_frac = float(n_fake_changed / max(1, n_vox))
    total_vox_per_sample = int(np.prod(crop_size))
    seam_mask_vox = int((template_masks["seam"] > 0.5).sum().item())

    metrics = {
        "num_pairs": int(len(pairs)),
        "ms_ssim_mean": ms_ssim_mean,
        "ms_ssim_std": ms_ssim_std,
        "seam_ssim_mean": seam_ssim_mean,
        "seam_ssim_std": seam_ssim_std,
        "seam_ms_ssim_mean": seam_ms_ssim_mean,
        "seam_ms_ssim_std": seam_ms_ssim_std,
        "seam_mask_voxels": seam_mask_vox,
        "seam_mask_frac": float(seam_mask_vox / max(1, total_vox_per_sample)),
        "ssim2d_mean": ssim2d_mean,
        "ssim2d_std": ssim2d_std,
        "fid": fid,
        "mmd": mmd,
        "mmd_seam": float(seam_mmd),
        "mmd_rbf_sigma_seam": float(seam_mmd_sigma),
        "feature_dim_seam": int(seam_feats_real_t.shape[1]),
        "num_feature_vectors_seam": int(seam_feats_real_t.shape[0]),
        "dist_feature_mode": dist_feature_mode,
        "feature_dim": int(feats_real_t.shape[1]),
        "num_feature_vectors": int(feats_real_t.shape[0]),
        "mmd_kernel": mmd_kernel,
        "mmd_rbf_sigma": float(mmd_sigma_used),
        "mmd_max_features": int(mmd_max_features),
        "use_medicalnet_mmd3d": bool(use_medicalnet_mmd3d),
        "medicalnet_enabled": bool(med3d_net is not None),
        "medicalnet_model": str(medicalnet_model),
        "medicalnet_ckpt_path": str(medicalnet_ckpt_path),
        "medicalnet_ckpt_dir": str(medicalnet_ckpt_dir),
        "medicalnet_project_ckpt_path": med3d_project_ckpt_path,
        "medicalnet_project_ckpt_error": med3d_project_ckpt_error,
        "medicalnet_hub_seed_path": med3d_hub_seed_path,
        "medicalnet_hub_seed_error": med3d_hub_seed_error,
        "medicalnet_hf_url": _canonical_hf_url(str(medicalnet_hf_url)),
        "medicalnet_hf_force_download": bool(medicalnet_hf_force_download),
        "medicalnet_hf_seed_path": med3d_hf_seed_path,
        "medicalnet_hf_seed_error": med3d_hf_seed_error,
        "medicalnet_strict": bool(medicalnet_strict),
        "medicalnet_init_error": med3d_init_error,
        "intensity_mode": intensity_mode,
        "norm_percentiles": [float(norm_percentiles[0]), float(norm_percentiles[1])],
        "fid_slice_axes": [int(a) for a in fid_slice_axes],
        "fid_slices_per_axis": int(fid_slices_per_axis),
        "fid_slice_margin": int(fid_slice_margin),
        "ssim2d_data_range": float(ssim2d_data_range),
        "ssim2d_kernel_type": ssim2d_kernel_type,
        "ssim2d_kernel_size": int(ssim2d_kernel_size),
        "ssim2d_kernel_sigma": float(ssim2d_kernel_sigma),
        "ms_ssim_kernel_type": str(ms_kernel_type).lower(),
        "ms_ssim_kernel_size": int(ms_kernel_size),
        "ms_ssim_kernel_sigma": float(ms_kernel_sigma),
        "ms_ssim_data_range": float(ms_data_range),
        "ms_ssim_weights": [float(w) for w in ms_weights],
        "clamp_applied": bool(intensity_mode == "clamp"),
        "clamp_min": -1.0,
        "clamp_max": 1.0,
        "real_oob_frac": real_oob_frac,
        "fake_oob_frac": fake_oob_frac,
        "real_changed_frac": real_changed_frac,
        "fake_changed_frac": fake_changed_frac,
        "real_clamped_frac": real_oob_frac if intensity_mode == "clamp" else 0.0,
        "fake_clamped_frac": fake_oob_frac if intensity_mode == "clamp" else 0.0,
        "seam_mask_path": seam_mask_path,
        "lhemi_mask_path": lhemi_mask_path,
        "rhemi_mask_path": rhemi_mask_path,
        "sub_mask_path": sub_mask_path,
        "lhemi_mask_voxels": int((template_masks["lhemi"] > 0.5).sum().item()),
        "rhemi_mask_voxels": int((template_masks["rhemi"] > 0.5).sum().item()),
        "sub_mask_voxels": int((template_masks["sub"] > 0.5).sum().item()),
    }
    metrics.update(region_metric_vals)
    metrics.update(med3d_metric_vals)
    metrics.update(region_ms_metric_vals)
    for axis, vals in ssim2d_axis_values.items():
        name = axis_to_name.get(axis, f"axis{axis}")
        mean_v = float(sum(vals) / max(1, len(vals)))
        std_v = float(pd.Series(vals).std(ddof=0) if len(vals) > 1 else 0.0)
        metrics[f"ssim2d_{name}_mean"] = mean_v
        metrics[f"ssim2d_{name}_std"] = std_v
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Pairwise MRI evaluation (SSIM/FID/MMD) for generated vs test-real samples.")
    ap.add_argument("--csv", default="", help="CSV with imageID and image column. If omitted, tries generated_dir/args.json.")
    ap.add_argument(
        "--data_split_json_path",
        default="",
        help="Train/val/test split JSON. If omitted, tries generated_dir/args.json then project default.",
    )
    ap.add_argument(
        "--image_key",
        default="",
        help="Column name for real image paths. If omitted, auto-detects (prefers image, then whole_brain).",
    )
    ap.add_argument("--generated_dir", required=True, help="Directory with generated files named *_[image_id].nii.gz")

    ap.add_argument("--crop_size", default="128,128,128", help="Foreground crop + center crop target size.")
    ap.add_argument(
        "--spacing",
        default="",
        help="Optional target spacing (e.g., 1.5,1.5,1.5). "
        "If empty, auto-uses generated_dir/args.json spacing when available.",
    )
    ap.add_argument(
        "--dist_feature_mode",
        default="inception2d",
        choices=["inception2d", "pooled"],
        help="Distribution-feature backbone for FID/MMD. 'inception2d' is paper-style.",
    )
    ap.add_argument(
        "--feature_pool",
        default="4,4,4",
        help="Adaptive pooling size (only used when --dist_feature_mode pooled).",
    )
    ap.add_argument("--fid_slice_axes", default="ax,cor,sag", help="Slice axes for inception2d mode: comma list of ax,cor,sag.")
    ap.add_argument("--fid_slices_per_axis", type=int, default=8, help="Number of slices per axis in inception2d mode.")
    ap.add_argument("--fid_slice_margin", type=int, default=8, help="Skip this many border slices in inception2d mode.")
    ap.add_argument("--mmd_kernel", default="rbf", choices=["rbf", "linear"])
    ap.add_argument("--mmd_rbf_sigma", type=float, default=-1.0, help="RBF sigma for MMD; <=0 uses median heuristic.")
    ap.add_argument("--mmd_max_features", type=int, default=2048, help="Max feature vectors per set for RBF-MMD.")
    ap.add_argument(
        "--use_medicalnet_mmd3d",
        action="store_true",
        help="Also compute 3D MMD in MedicalNet feature space for whole/seam/parts.",
    )
    ap.add_argument(
        "--medicalnet_model",
        default="medicalnet_resnet10_23datasets",
        help="MedicalNet torch.hub model name (e.g. medicalnet_resnet10_23datasets).",
    )
    ap.add_argument(
        "--medicalnet_ckpt_dir",
        default="ckpts/medicalnet",
        help="Project directory for MedicalNet checkpoints (preferred source).",
    )
    ap.add_argument(
        "--medicalnet_ckpt_path",
        default="",
        help="Optional explicit MedicalNet checkpoint path. Overrides directory lookup.",
    )
    ap.add_argument(
        "--medicalnet_hf_url",
        default="",
        help="Optional HuggingFace checkpoint URL used to pre-seed torch hub cache. "
        "Both '/blob/' and '/resolve/' links are accepted.",
    )
    ap.add_argument(
        "--medicalnet_hf_force_download",
        action="store_true",
        help="Force re-download of the MedicalNet checkpoint from HuggingFace.",
    )
    ap.add_argument(
        "--medicalnet_verbose",
        action="store_true",
        help="Enable verbose torch.hub output for MedicalNet loading.",
    )
    ap.add_argument(
        "--medicalnet_strict",
        action="store_true",
        help="Fail evaluation if MedicalNet cannot be initialized/downloaded.",
    )
    ap.add_argument(
        "--intensity_mode",
        default="normalize_both",
        choices=["clamp", "raw", "normalize_fake", "normalize_both"],
        help="How to handle intensity range before metrics. Default normalizes both real and generated volumes to [-1,1].",
    )
    ap.add_argument(
        "--norm_percentiles",
        default="0.5,99.5",
        help="Percentiles for normalize_* modes (q_low,q_high).",
    )
    ap.add_argument("--no_clamp", action="store_true", help="Disable [-1,1] clamping during evaluation.")
    ap.add_argument("--ssim2d_data_range", type=float, default=2.0, help="2D SSIM data range (2.0 for expected [-1,1]).")
    ap.add_argument("--ssim2d_kernel_type", default="gaussian", choices=["gaussian", "uniform"])
    ap.add_argument("--ssim2d_kernel_size", type=int, default=11)
    ap.add_argument("--ssim2d_kernel_sigma", type=float, default=1.5)
    ap.add_argument("--ms_data_range", type=float, default=2.0, help="MS-SSIM data range (2.0 for expected [-1,1]).")
    ap.add_argument("--ms_kernel_type", default="gaussian", choices=["gaussian", "uniform"])
    ap.add_argument("--ms_kernel_size", type=int, default=11)
    ap.add_argument("--ms_kernel_sigma", type=float, default=1.5)
    ap.add_argument(
        "--ms_weights",
        default="0.0448,0.2856,0.3001",
        help="Comma-separated MS-SSIM scale weights. Defaults to 3-scale (renormalized) for 128^3 volumes.",
    )
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--outdir", default="", help="Output directory. Default: generated_dir.")
    ap.add_argument("--run_name", default="", help="Optional subdirectory under outdir.")
    ap.add_argument("--max_pairs", type=int, default=-1, help="Optional cap on number of paired samples (debug).")
    ap.add_argument("--seam_mask_path", default="data/template/dilated_r2/seam_mask.nii.gz")
    ap.add_argument("--lhemi_mask_path", default="data/template/dilated_r2/lhemi_mask.nii.gz")
    ap.add_argument("--rhemi_mask_path", default="data/template/dilated_r2/rhemi_mask.nii.gz")
    ap.add_argument("--sub_mask_path", default="data/template/dilated_r2/sub_mask.nii.gz")

    args = ap.parse_args()

    crop_size = tuple(int(v.strip()) for v in args.crop_size.split(","))
    feature_pool = tuple(int(v.strip()) for v in args.feature_pool.split(","))
    fid_slice_axes = _parse_slice_axes(args.fid_slice_axes)
    ms_weights = _renorm_weights(_parse_float_tuple(args.ms_weights))
    norm_percentiles = _parse_float_tuple(args.norm_percentiles)
    if len(norm_percentiles) != 2:
        ap.error(f"--norm_percentiles expects exactly 2 values, got: {norm_percentiles}")
    intensity_mode = str(args.intensity_mode).strip().lower()
    if bool(args.no_clamp) and intensity_mode == "clamp":
        intensity_mode = "raw"

    generated_dir = os.path.abspath(args.generated_dir)
    generated_args = _load_generated_args(generated_dir)
    spacing: Tuple[float, float, float] | None = None
    spacing_str = str(args.spacing).strip()
    if not spacing_str:
        spacing_str = str(generated_args.get("spacing", "")).strip()
    if spacing_str:
        spacing_vals = _parse_float_tuple(spacing_str)
        if len(spacing_vals) != 3:
            ap.error(f"--spacing expects 3 values, got: {spacing_vals}")
        spacing = spacing_vals  # type: ignore[assignment]

    csv_path = str(args.csv).strip() or str(generated_args.get("csv", "")).strip()
    if not csv_path:
        ap.error("--csv is required unless generated_dir/args.json contains 'csv'.")
    csv_path = _resolve_existing_path(csv_path, generated_dir)
    if not os.path.exists(csv_path):
        ap.error(f"CSV not found: {csv_path}")

    split_path = str(args.data_split_json_path).strip()
    if not split_path:
        split_path = str(generated_args.get("data_split_json_path", "")).strip()
    if not split_path:
        split_path = "data/patient_splits_image_ids_75_10_15.json"
    split_path = _resolve_existing_path(split_path, generated_dir)
    if not os.path.exists(split_path):
        ap.error(f"Split JSON not found: {split_path}")

    seam_mask_path = _resolve_existing_path(str(args.seam_mask_path).strip(), generated_dir)
    lhemi_mask_path = _resolve_existing_path(str(args.lhemi_mask_path).strip(), generated_dir)
    rhemi_mask_path = _resolve_existing_path(str(args.rhemi_mask_path).strip(), generated_dir)
    sub_mask_path = _resolve_existing_path(str(args.sub_mask_path).strip(), generated_dir)
    for p in [seam_mask_path, lhemi_mask_path, rhemi_mask_path, sub_mask_path]:
        if not os.path.exists(p):
            ap.error(f"Template mask not found: {p}")

    image_key = str(args.image_key).strip()
    if not image_key:
        # Generated args may come from train/infer scripts with different key names.
        image_key = str(generated_args.get("image_key", "")).strip() or str(generated_args.get("whole_key", "")).strip()

    medicalnet_ckpt_dir = str(args.medicalnet_ckpt_dir).strip() or MEDICALNET_PROJECT_CKPT_DIR
    if not os.path.isabs(medicalnet_ckpt_dir):
        medicalnet_ckpt_dir = os.path.join(PROJECT_ROOT, medicalnet_ckpt_dir)
    medicalnet_ckpt_dir = os.path.abspath(medicalnet_ckpt_dir)

    medicalnet_ckpt_path = str(args.medicalnet_ckpt_path).strip()
    if medicalnet_ckpt_path:
        medicalnet_ckpt_path = _resolve_existing_path(medicalnet_ckpt_path, generated_dir)
        if not os.path.isabs(medicalnet_ckpt_path):
            medicalnet_ckpt_path = os.path.join(PROJECT_ROOT, medicalnet_ckpt_path)
        medicalnet_ckpt_path = os.path.abspath(medicalnet_ckpt_path)

    base_outdir = generated_dir if not str(args.outdir).strip() else os.path.abspath(args.outdir)
    outdir = os.path.join(base_outdir, args.run_name) if str(args.run_name).strip() else base_outdir
    os.makedirs(outdir, exist_ok=True)

    # Log everything to run.log for quick inspection.
    log_path = os.path.join(outdir, "run.log")
    log = open(log_path, "w", buffering=1)
    sys.stdout = log
    sys.stderr = log

    resolved_args = vars(args).copy()
    resolved_args["generated_dir"] = generated_dir
    resolved_args["csv"] = csv_path
    resolved_args["data_split_json_path"] = split_path
    resolved_args["image_key"] = image_key
    resolved_args["spacing_resolved"] = spacing
    resolved_args["seam_mask_path"] = seam_mask_path
    resolved_args["lhemi_mask_path"] = lhemi_mask_path
    resolved_args["rhemi_mask_path"] = rhemi_mask_path
    resolved_args["sub_mask_path"] = sub_mask_path
    resolved_args["medicalnet_ckpt_dir_resolved"] = medicalnet_ckpt_dir
    resolved_args["medicalnet_ckpt_path_resolved"] = medicalnet_ckpt_path
    eval_args_path = os.path.join(outdir, "eval_args.json")
    with open(eval_args_path, "w", encoding="utf-8") as f:
        json.dump(resolved_args, f, indent=2, sort_keys=True)

    print(f"Run dir: {outdir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Generated dir: {generated_dir}")
    print(f"CSV: {csv_path}")
    print(f"Split JSON: {split_path}")
    print(f"Saved eval args: {eval_args_path}")
    print(
        "MS-SSIM config: "
        f"data_range={float(args.ms_data_range)}, "
        f"kernel=({args.ms_kernel_type},{int(args.ms_kernel_size)},{float(args.ms_kernel_sigma)}), "
        f"weights={ms_weights}"
    )
    print(
        "2D SSIM config: "
        f"data_range={float(args.ssim2d_data_range)}, "
        f"kernel=({args.ssim2d_kernel_type},{int(args.ssim2d_kernel_size)},{float(args.ssim2d_kernel_sigma)})"
    )
    print(f"Spatial alignment config: spacing={spacing}, crop_size={crop_size}, divisible_pad_k=32")
    print(
        "Dist-metric config: "
        f"mode={args.dist_feature_mode}, "
        f"feature_pool={feature_pool}, "
        f"fid_axes={fid_slice_axes}, slices_per_axis={int(args.fid_slices_per_axis)}, margin={int(args.fid_slice_margin)}, "
        f"mmd_kernel={args.mmd_kernel}, mmd_rbf_sigma={float(args.mmd_rbf_sigma)}, mmd_max_features={int(args.mmd_max_features)}"
    )
    print(
        "MedicalNet 3D MMD config: "
        f"enabled={bool(args.use_medicalnet_mmd3d)}, "
        f"model={str(args.medicalnet_model)}, "
        f"ckpt_dir={medicalnet_ckpt_dir}, "
        f"ckpt_path={medicalnet_ckpt_path or '(auto by model)'}, "
        f"hf_url={_canonical_hf_url(str(args.medicalnet_hf_url)) or '(default by model)'}, "
        f"hf_force_download={bool(args.medicalnet_hf_force_download)}, "
        f"verbose={bool(args.medicalnet_verbose)}, "
        f"strict={bool(args.medicalnet_strict)}"
    )
    print(
        f"Intensity config: mode={intensity_mode}, "
        f"range=[-1.0,1.0], norm_percentiles={norm_percentiles}, "
        f"legacy_no_clamp={bool(args.no_clamp)}"
    )
    print(
        "Template masks: "
        f"seam={seam_mask_path}, "
        f"lhemi={lhemi_mask_path}, "
        f"rhemi={rhemi_mask_path}, "
        f"sub={sub_mask_path}"
    )

    test_id_to_real, resolved_image_key = _build_test_id_to_real_path(
        csv_path=csv_path,
        data_split_json_path=split_path,
        image_key=image_key,
        generated_dir=generated_dir,
    )
    print(f"Resolved image column: {resolved_image_key}")
    print(f"Test IDs in CSV/split map: {len(test_id_to_real)}")

    generated_files = _find_generated_files(generated_dir)
    print(f"Generated NIfTI files found: {len(generated_files)}")

    pairs, missing_in_test, unparsable, duplicate_ids = _pair_generated_with_test_real(
        generated_files=generated_files,
        test_id_to_real_path=test_id_to_real,
    )
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    total_test_ids = int(len(test_id_to_real))
    paired_ids = {str(p["image_id"]) for p in pairs}
    missing_generated_ids = sorted([str(k) for k in test_id_to_real.keys() if str(k) not in paired_ids])
    matched_test_ids = int(len(pairs))
    unmatched_test_ids = int(len(missing_generated_ids))
    matched_test_coverage_pct = float(100.0 * matched_test_ids / max(1, total_test_ids))

    print(f"Paired samples: {len(pairs)}")
    print(f"Unparsable generated filenames: {len(unparsable)}")
    print(f"Generated IDs not in test split: {len(missing_in_test)}")
    print(f"Duplicate generated IDs skipped: {len(duplicate_ids)}")
    print(
        f"Matched test coverage: {matched_test_ids}/{total_test_ids} "
        f"({matched_test_coverage_pct:.1f}%)"
    )
    if unmatched_test_ids > 0:
        print(
            f"Running metrics on matched subset only; "
            f"missing generated samples for {unmatched_test_ids} test IDs."
        )

    if len(unparsable) > 0:
        print("Unparsable examples (first 10):")
        for x in unparsable[:10]:
            print(f"  {x}")

    if len(missing_in_test) > 0:
        print("Missing IDs (first 20):")
        for x in missing_in_test[:20]:
            print(f"  {x}")
    if unmatched_test_ids > 0:
        print("Test IDs without generated match (first 20):")
        for x in missing_generated_ids[:20]:
            print(f"  {x}")

    pair_manifest_path = os.path.join(outdir, "paired_samples.csv")
    pd.DataFrame(pairs).to_csv(pair_manifest_path, index=False)
    pair_report = {
        "num_generated_files": len(generated_files),
        "num_test_ids_total": total_test_ids,
        "num_test_ids_matched": matched_test_ids,
        "num_test_ids_unmatched": unmatched_test_ids,
        "matched_test_coverage_pct": matched_test_coverage_pct,
        "num_pairs": len(pairs),
        "num_unparsable": len(unparsable),
        "num_missing_in_test": len(missing_in_test),
        "num_duplicate_ids": len(duplicate_ids),
        "resolved_image_key": resolved_image_key,
        "unparsable_files": unparsable,
        "missing_in_test": missing_in_test,
        "duplicate_ids": duplicate_ids,
        "missing_generated_in_output": missing_generated_ids,
    }
    pair_report_path = os.path.join(outdir, "pairing_report.json")
    with open(pair_report_path, "w", encoding="utf-8") as f:
        json.dump(pair_report, f, indent=2, sort_keys=True)
    print(f"Saved pairing manifest: {pair_manifest_path}")
    print(f"Saved pairing report: {pair_report_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics = run_pairwise_eval(
        pairs=pairs,
        crop_size=crop_size,
        batch_size=int(args.batch),
        num_workers=int(args.workers),
        feature_pool=feature_pool,
        spacing=spacing,
        dist_feature_mode=str(args.dist_feature_mode),
        fid_slice_axes=fid_slice_axes,
        fid_slices_per_axis=int(args.fid_slices_per_axis),
        fid_slice_margin=int(args.fid_slice_margin),
        mmd_kernel=str(args.mmd_kernel),
        mmd_rbf_sigma=float(args.mmd_rbf_sigma),
        mmd_max_features=int(args.mmd_max_features),
        intensity_mode=intensity_mode,
        norm_percentiles=(float(norm_percentiles[0]), float(norm_percentiles[1])),
        ssim2d_data_range=float(args.ssim2d_data_range),
        ssim2d_kernel_type=str(args.ssim2d_kernel_type),
        ssim2d_kernel_size=int(args.ssim2d_kernel_size),
        ssim2d_kernel_sigma=float(args.ssim2d_kernel_sigma),
        ms_kernel_type=str(args.ms_kernel_type),
        ms_kernel_size=int(args.ms_kernel_size),
        ms_kernel_sigma=float(args.ms_kernel_sigma),
        ms_data_range=float(args.ms_data_range),
        ms_weights=ms_weights,
        seam_mask_path=seam_mask_path,
        lhemi_mask_path=lhemi_mask_path,
        rhemi_mask_path=rhemi_mask_path,
        sub_mask_path=sub_mask_path,
        use_medicalnet_mmd3d=bool(args.use_medicalnet_mmd3d),
        medicalnet_model=str(args.medicalnet_model),
        medicalnet_ckpt_path=medicalnet_ckpt_path,
        medicalnet_ckpt_dir=medicalnet_ckpt_dir,
        medicalnet_hf_url=str(args.medicalnet_hf_url),
        medicalnet_hf_force_download=bool(args.medicalnet_hf_force_download),
        medicalnet_verbose=bool(args.medicalnet_verbose),
        medicalnet_strict=bool(args.medicalnet_strict),
        device=device,
    )

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Saved metrics to: {os.path.join(outdir, 'metrics.json')}")


if __name__ == "__main__":
    main()
