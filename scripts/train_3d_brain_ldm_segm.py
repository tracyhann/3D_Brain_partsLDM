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
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.transforms import MapTransform
from monai.utils import first
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from eval_utils import plot_unet_loss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


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


def _load_ckpt_into_ae(ae, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    ae.load_state_dict(state, strict=False)


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
) -> Tuple[int, int, Dict[str, Any]]:
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

    # Crop first (if source is larger than target)
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

    # Then pad (if source is smaller than target)
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


class AlignPartMasksToWholed(MapTransform):
    """
    Align part masks into whole-image space:
      - lhemi: pad/crop on the right side (keep left side fixed)
      - rhemi: pad/crop on the left side (keep right side fixed)
      - sub: keep bottom/left fixed, pad/crop above/right
      - whole mask: aligned to whole image shape without explicit directional shift
    """

    def __init__(
        self,
        *,
        image_key: str,
        whole_mask_key: str,
        lhemi_mask_key: str,
        rhemi_mask_key: str,
        sub_mask_key: str,
    ):
        super().__init__(keys=[image_key, whole_mask_key, lhemi_mask_key, rhemi_mask_key, sub_mask_key])
        self.image_key = image_key
        self.whole_mask_key = whole_mask_key
        self.lhemi_mask_key = lhemi_mask_key
        self.rhemi_mask_key = rhemi_mask_key
        self.sub_mask_key = sub_mask_key

    def __call__(self, data):
        d = dict(data)
        target_shape = tuple(int(v) for v in d[self.image_key].shape[-3:])

        d[self.whole_mask_key] = align_binary_mask_to_shape(
            d[self.whole_mask_key], target_shape, anchors=("start", "start", "start")
        )
        d[self.lhemi_mask_key] = align_binary_mask_to_shape(
            d[self.lhemi_mask_key], target_shape, anchors=("start", "start", "start")
        )
        d[self.rhemi_mask_key] = align_binary_mask_to_shape(
            d[self.rhemi_mask_key], target_shape, anchors=("end", "start", "start")
        )
        d[self.sub_mask_key] = align_binary_mask_to_shape(
            d[self.sub_mask_key], target_shape, anchors=("start", "start", "start")
        )
        return d


def build_transforms(
    *,
    image_key: str,
    whole_mask_key: str,
    lhemi_mask_key: str,
    rhemi_mask_key: str,
    sub_mask_key: str,
    spacing: Tuple[float, float, float],
    whole_size: Tuple[int, int, int],
):
    channel = 0
    keys = [image_key, whole_mask_key, lhemi_mask_key, rhemi_mask_key, sub_mask_key]
    mask_keys = [whole_mask_key, lhemi_mask_key, rhemi_mask_key, sub_mask_key]
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys),
            transforms.Spacingd(keys=[image_key], pixdim=spacing, mode=("bilinear",)),
            transforms.Spacingd(keys=mask_keys, pixdim=spacing, mode=("nearest",) * len(mask_keys)),
            transforms.Lambdad(keys=mask_keys, func=lambda x: (x > 0).float()),
            AlignPartMasksToWholed(
                image_key=image_key,
                whole_mask_key=whole_mask_key,
                lhemi_mask_key=lhemi_mask_key,
                rhemi_mask_key=rhemi_mask_key,
                sub_mask_key=sub_mask_key,
            ),
            transforms.DivisiblePadd(keys=[image_key], k=32, mode="constant", constant_values=-1.0),
            transforms.DivisiblePadd(keys=mask_keys, k=32, mode="constant", constant_values=0.0),
            transforms.CenterSpatialCropd(keys=keys, roi_size=whole_size),
            transforms.Lambdad(keys=mask_keys, func=lambda x: (x > 0).float()),
        ]
    )


def make_dataloaders_from_csv(
    *,
    csv_path: str,
    image_key: str,
    whole_mask_key: str,
    lhemi_mask_key: str,
    rhemi_mask_key: str,
    sub_mask_key: str,
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

        sample = {
            "image": row[image_key],
            "whole_brain_mask": row[whole_mask_key],
            "lhemi_mask": row[lhemi_mask_key],
            "rhemi_mask": row[rhemi_mask_key],
            "sub_mask": row[sub_mask_key],
        }
        for c in conditions:
            if c in row:
                sample[c] = row[c]

        split = image_ids[image_id]
        if split == "train":
            train_data.append(sample)
            n_train_added += 1
            if n_samples is not None and n_train_added >= n_samples:
                break
        elif split == "val":
            val_data.append(sample)
        elif split == "test":
            test_data.append(sample)

    train_ds = Dataset(data=train_data, transform=train_transforms)
    print(f"Transformed train shape (image): {train_ds[0]['image'].shape}")
    print(f"Transformed train shape (lhemi_mask): {train_ds[0]['lhemi_mask'].shape}")
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


def downsample_binary_mask(mask: torch.Tensor, latent_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(mask.shape[-3:]) != tuple(latent_shape):
        mask = F.interpolate(mask.float(), size=latent_shape, mode="nearest")
    return (mask > 0.5).float()


def build_mask_condition_latents(
    *,
    batch: Dict[str, torch.Tensor],
    device,
    latent_shape: Tuple[int, int, int],
) -> torch.Tensor:
    whole_mask = (batch["whole_brain_mask"].to(device) > 0).float()
    lhemi_mask = (batch["lhemi_mask"].to(device) > 0).float()
    rhemi_mask = (batch["rhemi_mask"].to(device) > 0).float()
    sub_mask = (batch["sub_mask"].to(device) > 0).float()

    whole_lat = downsample_binary_mask(whole_mask, latent_shape)
    lhemi_lat = downsample_binary_mask(lhemi_mask, latent_shape)
    rhemi_lat = downsample_binary_mask(rhemi_mask, latent_shape)
    sub_lat = downsample_binary_mask(sub_mask, latent_shape)

    # Required concat order: (lhemi, rhemi, sub, whole)
    return torch.cat([lhemi_lat, rhemi_lat, sub_lat, whole_lat], dim=1)


@torch.no_grad()
def eval_ldm_loss_only(
    *,
    unet: torch.nn.Module,
    autoencoder: AutoencoderKL,
    val_loader,
    device,
    scale_factor: float,
    torch_autocast: bool,
    val_batches: int = 4,
):
    unet.eval()
    autoencoder.eval()

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
        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(images) * scale_factor
        cond_lat = build_mask_condition_latents(
            batch=batch,
            device=device,
            latent_shape=tuple(z.shape[-3:]),
        )

        noise = torch.randn_like(z)
        t = torch.randint(0, ddpm.num_train_timesteps, (images.shape[0],), device=device).long()
        z_t = ddpm.add_noise(original_samples=z, noise=noise, timesteps=t)

        with autocast(device_type="cuda", enabled=amp_enabled):
            model_input = torch.cat([z_t, cond_lat], dim=1)
            noise_pred = unet(model_input, timesteps=t)
            loss = F.mse_loss(noise_pred.float(), noise.float())

        val_loss += float(loss.item())
        n_seen += 1

    return {"val_loss_simple": val_loss / max(1, n_seen)}


@torch.no_grad()
def eval_ldm_fast(
    *,
    unet: torch.nn.Module,
    autoencoder: AutoencoderKL,
    val_loader,
    device,
    scale_factor: float,
    torch_autocast: bool,
    outdir: str,
    global_step: int,
    eval_n: int = 8,
    ddim_steps: int = 50,
    val_batches: int = 8,
):
    unet.eval()
    autoencoder.eval()
    amp_enabled = bool(torch_autocast and device.type == "cuda")

    ddpm = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )

    val_loss = 0.0
    n_seen = 0
    cond_latents = []
    latent_ref_shape = None
    ref_affine = np.eye(4, dtype=np.float32)

    for i, batch in enumerate(val_loader):
        if i >= max(val_batches, eval_n):
            break

        images = batch["image"].to(device)
        if i == 0:
            ref_affine = _extract_affine_from_tensor(batch["image"])

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(images) * scale_factor
        latent_ref_shape = tuple(z.shape[1:])  # [C,D,H,W]

        cond_lat = build_mask_condition_latents(
            batch=batch,
            device=device,
            latent_shape=tuple(z.shape[-3:]),
        )
        if len(cond_latents) < eval_n:
            cond_latents.append(cond_lat[:1])

        if i < val_batches:
            noise = torch.randn_like(z)
            t = torch.randint(0, ddpm.num_train_timesteps, (images.shape[0],), device=device).long()
            z_t = ddpm.add_noise(original_samples=z, noise=noise, timesteps=t)
            with autocast(device_type="cuda", enabled=amp_enabled):
                model_input = torch.cat([z_t, cond_lat], dim=1)
                noise_pred = unet(model_input, timesteps=t)
                loss = F.mse_loss(noise_pred.float(), noise.float())
            val_loss += float(loss.item())
            n_seen += 1

    if len(cond_latents) == 0 or latent_ref_shape is None:
        return {"val_loss": 0.0, "latent_std": 0.0, "self_l2": 0.0}

    while len(cond_latents) < eval_n:
        cond_latents.append(cond_latents[0].clone())
    cond_eval = torch.cat(cond_latents[:eval_n], dim=0)

    z = torch.randn((eval_n,) + latent_ref_shape, device=device)
    ddim = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=False,
    )
    ddim.set_timesteps(num_inference_steps=ddim_steps)

    for t in ddim.timesteps:
        t_int = int(t.item()) if hasattr(t, "item") else int(t)
        t_batch = torch.full((eval_n,), t_int, device=device, dtype=torch.long)
        with autocast(device_type="cuda", enabled=amp_enabled):
            model_input = torch.cat([z, cond_eval], dim=1)
            eps = unet(model_input, timesteps=t_batch)
        z, _ = ddim.step(eps, t_int, z)

    with torch.no_grad():
        x_gen = autoencoder.decode_stage_2_outputs(z / scale_factor)
        z_back = autoencoder.encode_stage_2_inputs(x_gen) * scale_factor
    latent_std = float(z_back.std().item())

    x_flat = x_gen.float().view(eval_n, -1)
    x_flat = (x_flat - x_flat.mean(dim=1, keepdim=True)) / (x_flat.std(dim=1, keepdim=True) + 1e-6)
    dsum = 0.0
    cnt = 0
    for i in range(eval_n):
        for j in range(i + 1, eval_n):
            dsum += torch.mean((x_flat[i] - x_flat[j]) ** 2).item()
            cnt += 1
    self_l2 = float(dsum / max(1, cnt))

    step_dir = os.path.join(outdir, "eval_samples", f"step{global_step:09d}")
    os.makedirs(step_dir, exist_ok=True)
    x_gen_cpu = x_gen.detach().cpu()
    for i in range(x_gen_cpu.shape[0]):
        arr = x_gen_cpu[i, 0].numpy()
        nib.save(nib.Nifti1Image(arr, ref_affine), os.path.join(step_dir, f"sample_{i:03d}.nii.gz"))

    torch.save(
        {
            "global_step": global_step,
            "val_loss": val_loss / max(1, n_seen),
            "latent_std": latent_std,
            "self_l2": self_l2,
            "x_gen": x_gen_cpu,
        },
        os.path.join(step_dir, "eval_snapshot.pt"),
    )

    return {
        "val_loss": val_loss / max(1, n_seen),
        "latent_std": latent_std,
        "self_l2": self_l2,
    }


def train_ldm_steps(
    *,
    unet,
    train_loader,
    val_loader,
    autoencoder,
    max_steps: int,
    lr: float = 1e-4,
    scale_factor: Optional[float] = None,
    torch_autocast: bool = True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    outdir: str = "ckpts",
    ckpt_every: int = 10_000,
    last_every: int = 1_000,
    eval_every: int = 5_000,
    simple_eval_every: int = 2_000,
    eval_n: int = 8,
    ddim_steps: int = 50,
    eval_val_batches: int = 8,
    simple_eval_val_batches: int = 4,
    resume_ckpt: str = "",
):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    amp_enabled = bool(torch_autocast and device.type == "cuda")

    if scale_factor is None:
        with torch.no_grad(), torch.autocast("cuda", enabled=amp_enabled):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
        scale_factor = float(1.0 / torch.std(z).item())
        print(f"[LDM-segm] scale_factor set to {scale_factor}")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)
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
            unet, resume_ckpt, device, opt=optimizer, scaler=scaler
        )
        if isinstance(extra, dict):
            history = extra.get("history", history)
            if "scale_factor" in extra:
                scale_factor = float(extra["scale_factor"])
        if not isinstance(history.get("loss_curve"), list):
            history["loss_curve"] = []
        if not isinstance(history.get("epoch_loss_curve"), list):
            history["epoch_loss_curve"] = []
        print(f"[resume] global_step={global_step}, start_epoch~={start_epoch}, scale_factor={scale_factor}")

    if global_step >= max_steps:
        print(f"[resume] global_step ({global_step}) >= max_steps ({max_steps}), nothing to train.")
        return

    unet.train()
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

    pbar = tqdm(range(global_step, max_steps), ncols=110)
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(images) * scale_factor
            cond_lat = build_mask_condition_latents(
                batch=batch,
                device=device,
                latent_shape=tuple(z.shape[-3:]),
            )

        noise = torch.randn_like(z)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
        z_t = scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)

        with autocast(device_type="cuda", enabled=amp_enabled):
            model_input = torch.cat([z_t, cond_lat], dim=1)
            noise_pred = unet(model_input, timesteps=timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step = step + 1

        running += float(loss.item())
        running_n += 1
        avg_loss = running / max(1, running_n)
        history["loss_curve"].append(float(loss.item()))

        epoch = int(max(0, global_step - 1) // steps_per_epoch)
        pbar.set_description(f"step {global_step}/{max_steps} (ep~{epoch})")
        pbar.set_postfix({"loss": f"{avg_loss:.6f}", "sf": f"{scale_factor:.4f}"})

        if global_step % 200 == 0:
            history["train_loss"].append({"step": global_step, "loss": avg_loss})

        if (last_every > 0) and (global_step % last_every == 0):
            _update_epoch_loss_curve()
            _save_ckpt(
                os.path.join(outdir, "UNET_last.pt"),
                unet,
                optimizer,
                scaler,
                global_step=global_step,
                epoch=epoch,
                extra={"scale_factor": float(scale_factor), "history": history},
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
                unet,
                optimizer,
                scaler,
                global_step=global_step,
                epoch=epoch,
                extra={"scale_factor": float(scale_factor), "history": history},
            )

        if val_loader is not None and (eval_every > 0) and (global_step % eval_every == 0):
            metrics = eval_ldm_fast(
                unet=unet,
                autoencoder=autoencoder,
                val_loader=val_loader,
                device=device,
                scale_factor=scale_factor,
                torch_autocast=torch_autocast,
                outdir=outdir,
                global_step=global_step,
                eval_n=eval_n,
                ddim_steps=ddim_steps,
                val_batches=eval_val_batches,
            )
            history["eval"].append({"step": global_step, **metrics})
            unet.train()
        elif val_loader is not None and (simple_eval_every > 0) and (global_step % simple_eval_every == 0):
            metrics = eval_ldm_loss_only(
                unet=unet,
                autoencoder=autoencoder,
                val_loader=val_loader,
                device=device,
                scale_factor=scale_factor,
                torch_autocast=torch_autocast,
                val_batches=simple_eval_val_batches,
            )
            history["simple_eval"].append({"step": global_step, **metrics})
            unet.train()

    _update_epoch_loss_curve()
    final_path = os.path.join(outdir, "UNET_last.pt")
    _save_ckpt(
        final_path,
        unet,
        optimizer,
        scaler,
        global_step=global_step,
        epoch=int(global_step // steps_per_epoch),
        extra={"scale_factor": float(scale_factor), "history": history},
    )
    plot_unet_loss(
        history["epoch_loss_curve"],
        title=f"UNET Epoch-Average Loss_step{global_step}",
        outdir=outdir,
        filename="UNET_loss.png",
    )


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file with argument defaults.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Train MONAI 3D LDM with segmentation-mask latent concat conditioning.",
        parents=[pre_ap],
    )
    ap.add_argument("--csv", default="data/processed_parts/whole_brain+3parts+masks_0206.csv")
    ap.add_argument("--data_split_json_path", default="data/patient_splits_image_ids_75_10_15.json")

    ap.add_argument("--image_key", default="whole_brain")
    ap.add_argument("--whole_mask_key", default="whole_brain_mask")
    ap.add_argument("--lhemi_mask_key", default="lhemi_mask")
    ap.add_argument("--rhemi_mask_key", default="rhemi_mask")
    ap.add_argument("--sub_mask_key", default="sub_mask")

    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Whole-brain crop D,H,W")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True)
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--train_val_split", type=float, default=0.1)
    # compatibility with train_3d_brain_ldm_steps.py configs (unused in this script)
    ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="ldm")

    ap.add_argument("--ae_latent_ch", type=int, default=8)
    ap.add_argument("--ae_num_channels", default="64,128,256")
    ap.add_argument("--ae_attention_levels", default="0,0,0")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt")

    ap.add_argument("--max_steps", type=int, default=120_000)
    ap.add_argument("--ckpt_every", type=int, default=10_000)
    ap.add_argument("--last_every", type=int, default=1_000)
    ap.add_argument("--eval_every", type=int, default=10_000)
    ap.add_argument("--simple_eval_every", type=int, default=2_000)
    ap.add_argument("--eval_n", type=int, default=8)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--eval_val_batches", type=int, default=8)
    ap.add_argument("--simple_eval_val_batches", type=int, default=4)
    ap.add_argument("--resume_ckpt", default="", help="Packaged UNET_step*.pt to resume")

    ap.add_argument("--ldm_lr", type=float, default=1e-4)
    ap.add_argument("--ldm_num_channels", default="256,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_ckpt", default="", help="Warm-start UNet weights")
    # compatibility with train_3d_brain_ldm_steps.py configs (unused in this script)
    ap.add_argument("--ldm_epochs", type=int, default=150)
    ap.add_argument("--ldm_use_cond", default="True")
    ap.add_argument("--ldm_sample_every", type=int, default=25)

    ap.add_argument("--outdir", default="ckpts")
    ap.add_argument("--out_prefix", default="")
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

    os.makedirs(args.outdir, exist_ok=True)
    experiment_dir = os.path.join(args.outdir, f"{args.out_prefix}_{args.out_postfix}")
    os.makedirs(experiment_dir, exist_ok=True)
    args.outdir = experiment_dir

    print(f"\nOutput dir: {args.outdir}\n")

    cfg = vars(args).copy()
    for k, v in list(cfg.items()):
        if isinstance(v, tuple):
            cfg[k] = list(v)
    with open(os.path.join(args.outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    if args.stage == "ae":
        print("Stage 'ae' is not implemented in this script. Use a pretrained AE and stage='ldm'.")
        return

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size = tuple(int(x) for x in args.size.split(","))

    train_transforms = build_transforms(
        image_key="image",
        whole_mask_key="whole_brain_mask",
        lhemi_mask_key="lhemi_mask",
        rhemi_mask_key="rhemi_mask",
        sub_mask_key="sub_mask",
        spacing=spacing,
        whole_size=size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device}\n")

    seed_all(1017)

    try:
        n_samples = int(args.n_samples)
    except Exception:
        n_samples = None

    train_loader, val_loader, _ = make_dataloaders_from_csv(
        csv_path=args.csv,
        image_key=args.image_key,
        whole_mask_key=args.whole_mask_key,
        lhemi_mask_key=args.lhemi_mask_key,
        rhemi_mask_key=args.rhemi_mask_key,
        sub_mask_key=args.sub_mask_key,
        conditions=("age", "sex", "group"),
        train_transforms=train_transforms,
        n_samples=n_samples,
        data_split_json_path=args.data_split_json_path,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
    ae_latent_ch = int(args.ae_latent_ch)
    ae_attention_levels = tuple(bool(int(x)) for x in args.ae_attention_levels.split(","))

    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=ae_num_channels,
        latent_channels=ae_latent_ch,
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=ae_attention_levels,
    )
    autoencoder.to(device)
    _load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)

    ldm_num_channels = tuple(int(x) for x in args.ldm_num_channels.split(","))
    ldm_num_head_channels = tuple(int(x) for x in args.ldm_num_head_channels.split(","))

    # Noisy latent + 4 binary mask latent channels.
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=ae_latent_ch + 4,
        out_channels=ae_latent_ch,
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
    unet.to(device)

    scale_factor = None
    if args.ldm_ckpt:
        _, _, extra = _load_ckpt_into_unet(unet, args.ldm_ckpt, device)
        if isinstance(extra, dict) and ("scale_factor" in extra):
            scale_factor = float(extra["scale_factor"])
            print(f"Loading scale factor = {scale_factor}")

    torch_autocast = str2bool(args.torch_autocast)

    train_ldm_steps(
        unet=unet,
        train_loader=train_loader,
        val_loader=val_loader,
        autoencoder=autoencoder,
        max_steps=args.max_steps,
        lr=args.ldm_lr,
        scale_factor=scale_factor,
        torch_autocast=torch_autocast,
        device=device,
        outdir=args.outdir,
        ckpt_every=args.ckpt_every,
        last_every=args.last_every,
        eval_every=args.eval_every,
        simple_eval_every=args.simple_eval_every,
        eval_n=args.eval_n,
        ddim_steps=args.ddim_steps,
        eval_val_batches=args.eval_val_batches,
        simple_eval_val_batches=args.simple_eval_val_batches,
        resume_ckpt=args.resume_ckpt,
    )


if __name__ == "__main__":
    main()