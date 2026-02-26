import os
import json
import random
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from monai import transforms
from monai.data import Dataset, DataLoader
from monai.utils import first

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from eval_utils import plot_unet_loss


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
        opt.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    gs = int(ckpt.get("global_step", 0))
    ep = int(ckpt.get("epoch", 0))
    extra = ckpt.get("extra", {}) or {}
    return gs, ep, extra


def build_transforms(keys: List[str], spacing: Tuple[float, float, float], whole_size: Tuple[int, int, int]):
    channel = 0
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys),
            transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear",) * len(keys)),
            transforms.DivisiblePadd(keys=["whole"], k=32, mode="constant", constant_values=-1.0),
            transforms.CenterSpatialCropd(keys=["whole"], roi_size=whole_size),
        ]
    )


def make_dataloaders_from_csv(
    csv_path,
    whole_key,
    lhemi_key,
    rhemi_key,
    sub_key,
    conditions=("age", "sex", "vol", "group"),
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
            "whole": row[whole_key],
            "lhemi": row[lhemi_key],
            "rhemi": row[rhemi_key],
            "sub": row[sub_key],
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
    print(f"Transformed data shape (whole): {train_ds[0]['whole'].shape}")
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
        num_workers=num_workers,
        pin_memory=True,
    )

    test_ds = Dataset(data=test_data, transform=train_transforms)
    print(f"Number of test samples: {len(test_ds)}")
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_coarse_brain_from_parts(
    whole: torch.Tensor,
    lhemi: torch.Tensor,
    rhemi: torch.Tensor,
    sub: torch.Tensor,
    bg_value: float = -1.0,
    valid_thresh: float = -0.995,
) -> torch.Tensor:
    """
    Build coarse brain in whole image space by pasting parts.
    Overlap voxels are reset to bg_value.
    """
    b, _, d, h, w = whole.shape
    coarse = torch.full_like(whole, fill_value=bg_value)
    occupied = torch.zeros((b, 1, d, h, w), device=whole.device, dtype=torch.bool)

    def _paste(src: torch.Tensor, x0: int, y0: int, z0: int):
        nonlocal coarse, occupied
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
        valid = srcv > valid_thresh

        cview = coarse[:, :, x0c:x1, y0c:y1, z0c:z1]
        oview = occupied[:, :, x0c:x1, y0c:y1, z0c:z1]

        overlap = valid & oview
        if overlap.any():
            cview = torch.where(overlap, torch.full_like(cview, bg_value), cview)

        fresh = valid & (~oview)
        if fresh.any():
            cview = torch.where(fresh, srcv, cview)
            oview = oview | fresh

        coarse[:, :, x0c:x1, y0c:y1, z0c:z1] = cview
        occupied[:, :, x0c:x1, y0c:y1, z0c:z1] = oview

    # left hemi: paste from left
    _paste(lhemi, x0=0, y0=0, z0=0)
    # right hemi: paste from right
    _paste(rhemi, x0=d - rhemi.shape[2], y0=0, z0=0)
    # sub: paste at origin in y/z as defined by crop logic
    _paste(sub, x0=0, y0=0, z0=0)

    return coarse


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


@torch.no_grad()
def eval_ldm_loss_only(
    *,
    unet: torch.nn.Module,
    autoencoder: AutoencoderKL,
    val_loader,
    device,
    scale_factor: float,
    torch_autocast: bool,
    coarse_valid_thresh: float,
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

        whole = batch["whole"].to(device)
        lhemi = batch["lhemi"].to(device)
        rhemi = batch["rhemi"].to(device)
        sub = batch["sub"].to(device)
        coarse = build_coarse_brain_from_parts(
            whole, lhemi, rhemi, sub, valid_thresh=coarse_valid_thresh
        )

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(whole) * scale_factor
            z_ctx = autoencoder.encode_stage_2_inputs(coarse) * scale_factor

        noise = torch.randn_like(z)
        t = torch.randint(0, ddpm.num_train_timesteps, (whole.shape[0],), device=device).long()
        z_t = ddpm.add_noise(original_samples=z, noise=noise, timesteps=t)

        with autocast(device_type="cuda", enabled=amp_enabled):
            model_input = torch.cat([z_t, z_ctx], dim=1)
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
    coarse_valid_thresh: float,
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
    context_latents = []
    ref_affine = np.eye(4, dtype=np.float32)

    for i, batch in enumerate(val_loader):
        if i >= max(val_batches, eval_n):
            break

        whole = batch["whole"].to(device)
        lhemi = batch["lhemi"].to(device)
        rhemi = batch["rhemi"].to(device)
        sub = batch["sub"].to(device)
        coarse = build_coarse_brain_from_parts(
            whole, lhemi, rhemi, sub, valid_thresh=coarse_valid_thresh
        )

        if i == 0:
            ref_affine = _extract_affine_from_tensor(batch["whole"])

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(whole) * scale_factor
            z_ctx = autoencoder.encode_stage_2_inputs(coarse) * scale_factor

        if len(context_latents) < eval_n:
            context_latents.append(z_ctx[:1])

        if i < val_batches:
            noise = torch.randn_like(z)
            t = torch.randint(0, ddpm.num_train_timesteps, (whole.shape[0],), device=device).long()
            z_t = ddpm.add_noise(original_samples=z, noise=noise, timesteps=t)

            with autocast(device_type="cuda", enabled=amp_enabled):
                model_input = torch.cat([z_t, z_ctx], dim=1)
                noise_pred = unet(model_input, timesteps=t)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            val_loss += float(loss.item())
            n_seen += 1

    if len(context_latents) == 0:
        return {"val_loss": 0.0, "latent_std": 0.0, "self_l2": 0.0}

    # repeat first context if not enough samples
    while len(context_latents) < eval_n:
        context_latents.append(context_latents[0].clone())

    z_ctx_eval = torch.cat(context_latents[:eval_n], dim=0)
    z = torch.randn_like(z_ctx_eval)

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
            model_input = torch.cat([z, z_ctx_eval], dim=1)
            eps = unet(model_input, timesteps=t_batch)
        z, _ = ddim.step(eps, t_int, z)

    with torch.no_grad():
        x_gen = autoencoder.decode_stage_2_outputs(z / scale_factor)

    with torch.no_grad():
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

    metrics = {
        "val_loss": val_loss / max(1, n_seen),
        "latent_std": latent_std,
        "self_l2": self_l2,
    }
    return metrics


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
    coarse_valid_thresh: float = -0.995,
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

    if scale_factor is None:
        with torch.no_grad(), torch.autocast("cuda", enabled=torch_autocast):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["whole"].to(device))
        scale_factor = float(1.0 / torch.std(z).item())
        print(f"[LDM] scale_factor set to {scale_factor}")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler("cuda", enabled=bool(torch_autocast and device.type == "cuda"))

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

    unet.train()
    os.makedirs(outdir, exist_ok=True)

    data_iter = iter(train_loader)
    steps_per_epoch = max(1, int(len(train_loader)))

    def _update_epoch_loss_curve():
        raw = history.get("loss_curve", [])
        epoch_curve = []
        for i in range(0, len(raw), steps_per_epoch):
            chunk = raw[i: i + steps_per_epoch]
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

        whole = batch["whole"].to(device)
        lhemi = batch["lhemi"].to(device)
        rhemi = batch["rhemi"].to(device)
        sub = batch["sub"].to(device)
        coarse = build_coarse_brain_from_parts(
            whole, lhemi, rhemi, sub, valid_thresh=coarse_valid_thresh
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(whole) * scale_factor
            z_ctx = autoencoder.encode_stage_2_inputs(coarse) * scale_factor

        noise = torch.randn_like(z)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (whole.shape[0],), device=device).long()
        z_t = scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)

        with autocast(device_type="cuda", enabled=torch_autocast and device.type == "cuda"):
            model_input = torch.cat([z_t, z_ctx], dim=1)
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
                coarse_valid_thresh=coarse_valid_thresh,
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
                coarse_valid_thresh=coarse_valid_thresh,
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
        description="Train MONAI 3D LDM with coarse-brain latent concat conditioning.",
        parents=[pre_ap],
    )
    ap.add_argument("--csv", default="data/processed_parts/whole_brain+3parts+masks_0206.csv")
    ap.add_argument("--data_split_json_path", default="data/patient_splits_image_ids_75_10_15.json")

    ap.add_argument("--whole_key", default="whole_brain")
    ap.add_argument("--lhemi_key", default="lhemi")
    ap.add_argument("--rhemi_key", default="rhemi")
    ap.add_argument("--sub_key", default="sub")

    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Whole-brain crop D,H,W")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True)
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--train_val_split", type=float, default=0.1)
    ap.add_argument("--coarse_valid_thresh", type=float, default=-0.995)
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

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size = tuple(int(x) for x in args.size.split(","))

    keys = ["whole", "lhemi", "rhemi", "sub"]
    train_transforms = build_transforms(keys=keys, spacing=spacing, whole_size=size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device}\n")

    seed_all(1017)

    try:
        n_samples = int(args.n_samples)
    except Exception:
        n_samples = None

    train_loader, val_loader, _ = make_dataloaders_from_csv(
        csv_path=args.csv,
        whole_key=args.whole_key,
        lhemi_key=args.lhemi_key,
        rhemi_key=args.rhemi_key,
        sub_key=args.sub_key,
        conditions=("age", "sex", "vol", "group"),
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

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=ae_latent_ch * 2,
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
        coarse_valid_thresh=float(args.coarse_valid_thresh),
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
