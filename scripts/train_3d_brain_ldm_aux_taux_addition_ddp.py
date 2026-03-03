"""
DDP trainer for train_3d_brain_ldm_aux_taux_addition.py.

Example (8 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_3d_brain_ldm_aux_taux_addition_ddp.py \
  --config configs/whole_brain_aux_taux_spacing1p5_ADDITION.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

# Import single-GPU script as base utilities.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_3d_brain_ldm_aux_taux_addition as base  # noqa: E402


def parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t"}


def setup_ddp() -> Tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs.")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def broadcast_str(msg: str, src: int = 0) -> str:
    payload = [msg]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _dist_mean_value(v: torch.Tensor, world_size: int) -> float:
    t = v.detach().to(dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / max(1, world_size))


def _dist_mean_float(v: float, device: torch.device, world_size: int) -> float:
    t = torch.tensor(float(v), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / max(1, world_size))


def make_dataloaders_from_csv_ddp(
    *,
    csv_path: str,
    image_key: str,
    rank: int,
    conditions=("age", "sex", "group", "condition"),
    train_transforms=None,
    n_samples=None,
    data_split_json_path="data/patient_splits_image_ids_75_10_15.json",
    batch_size=1,
    num_workers=8,
    seed: int = 1017,
):
    with open(data_split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    image_ids: Dict[Any, str] = {}
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
            if c in row.index:
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
    val_ds = Dataset(data=val_data, transform=train_transforms)
    test_ds = Dataset(data=test_data, transform=train_transforms)

    if len(train_ds) == 0:
        raise ValueError("No training samples found after applying split and n_samples filters.")

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Eval runs on rank-0 only.
    val_loader = None
    test_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, train_sampler


def train_ldm_aux_taux_ddp(
    *,
    ddp_whole_unet: DDP,
    whole_ae: base.AutoencoderKL,
    hemi_ae: base.AutoencoderKL,
    hemi_unet: base.DiffusionModelUNet,
    hemi_conditioner: base.HemisphereConditioner,
    sub_ae: base.AutoencoderKL,
    sub_unet: base.DiffusionModelUNet,
    train_loader,
    train_sampler: DistributedSampler,
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
    rank: int,
    world_size: int,
    local_rank: int,
):
    scheduler = base.DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )

    for m in [whole_ae, hemi_ae, hemi_unet, hemi_conditioner, sub_ae, sub_unet]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    amp_enabled = bool(torch_autocast and device.type == "cuda")
    aux_every = max(1, int(aux_every))

    # Scale factors.
    if whole_scale_factor is None:
        with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
            z = whole_ae.encode_stage_2_inputs(next(iter(train_loader))["image"].to(device))
            local_std = torch.std(z).detach()
        std_tensor = local_std.to(device=device, dtype=torch.float64)
        dist.all_reduce(std_tensor, op=dist.ReduceOp.SUM)
        std_tensor = std_tensor / max(1, world_size)
        whole_scale_factor = float(1.0 / max(std_tensor.item(), 1e-8))
        if rank == 0:
            print(f"[scale] whole_scale_factor auto={whole_scale_factor:.6f}")

    if hemi_scale_factor is None or sub_scale_factor is None:
        ref_img = next(iter(train_loader))["image"].to(device)[:1]
        xL_whole_masked = base._apply_binary_mask_bg(ref_img, mask_ctx["mL"], bg_value)
        xR_whole_masked = base._apply_binary_mask_bg(ref_img, mask_ctx["mR"], bg_value)
        xS_whole_masked = base._apply_binary_mask_bg(ref_img, mask_ctx["mS"], bg_value)
        xL = base._crop_right(xL_whole_masked, hemi_shape)
        xR_masked_raw = base._crop_left(xR_whole_masked, hemi_shape)
        xR = base._flip_lr(xR_masked_raw)
        xS = base._crop_sub(xS_whole_masked, sub_shape)

        if hemi_scale_factor is None:
            local_sf = base._estimate_scale_factor_from_batch(
                ae=hemi_ae,
                x=torch.cat([xL, xR], dim=0),
                amp_enabled=amp_enabled,
            )
            hemi_scale_factor = _dist_mean_float(local_sf, device=device, world_size=world_size)
            if rank == 0:
                print(f"[scale] hemi_scale_factor auto={hemi_scale_factor:.6f}")
        if sub_scale_factor is None:
            local_sf = base._estimate_scale_factor_from_batch(ae=sub_ae, x=xS, amp_enabled=amp_enabled)
            sub_scale_factor = _dist_mean_float(local_sf, device=device, world_size=world_size)
            if rank == 0:
                print(f"[scale] sub_scale_factor auto={sub_scale_factor:.6f}")

    optimizer = torch.optim.AdamW(ddp_whole_unet.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler("cuda", enabled=amp_enabled)

    global_step = 0
    history = {
        "train_loss": [],
        "simple_eval": [],
        "eval": [],
        "loss_curve": [],
        "epoch_loss_curve": [],
    }

    if resume_ckpt:
        global_step, _, extra = base._load_ckpt_into_unet(
            ddp_whole_unet.module,
            resume_ckpt,
            device,
            opt=optimizer,
            scaler=scaler,
        )
        if isinstance(extra, dict):
            loaded_history = extra.get("history")
            if isinstance(loaded_history, dict):
                history = loaded_history
            if "whole_scale_factor" in extra:
                whole_scale_factor = float(extra["whole_scale_factor"])
            if "hemi_scale_factor" in extra:
                hemi_scale_factor = float(extra["hemi_scale_factor"])
            if "sub_scale_factor" in extra:
                sub_scale_factor = float(extra["sub_scale_factor"])
        if rank == 0:
            print(
                "[resume] "
                f"global_step={global_step}, "
                f"sf_whole={whole_scale_factor:.6f}, sf_hemi={hemi_scale_factor:.6f}, sf_sub={sub_scale_factor:.6f}"
            )

    if not isinstance(history.get("train_loss"), list):
        history["train_loss"] = []
    if not isinstance(history.get("simple_eval"), list):
        history["simple_eval"] = []
    if not isinstance(history.get("eval"), list):
        history["eval"] = []
    if not isinstance(history.get("loss_curve"), list):
        history["loss_curve"] = []
    if not isinstance(history.get("epoch_loss_curve"), list):
        history["epoch_loss_curve"] = []

    if global_step >= max_steps:
        if rank == 0:
            print(f"[resume] global_step ({global_step}) >= max_steps ({max_steps}), nothing to train.")
        return

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
    dist.barrier(device_ids=[local_rank])
    ddp_whole_unet.train()

    steps_per_epoch = max(1, int(len(train_loader)))

    def _update_epoch_loss_curve():
        raw = history.get("loss_curve", [])
        epoch_curve = []
        for i in range(0, len(raw), steps_per_epoch):
            chunk = raw[i : i + steps_per_epoch]
            if chunk:
                epoch_curve.append(float(sum(chunk) / len(chunk)))
        history["epoch_loss_curve"] = epoch_curve

    current_epoch = int(max(0, global_step - 1) // max(1, steps_per_epoch))
    train_sampler.set_epoch(current_epoch)
    data_iter = iter(train_loader)

    running = 0.0
    running_n = 0
    progress = tqdm(
        total=max_steps - global_step,
        ncols=130,
        file=sys.__stdout__,
        disable=rank != 0,
    )

    for step in range(global_step, max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            current_epoch += 1
            train_sampler.set_epoch(current_epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
            z_whole0 = whole_ae.encode_stage_2_inputs(images) * whole_scale_factor

        noise = torch.randn_like(z_whole0)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
        z_t = scheduler.add_noise(original_samples=z_whole0, noise=noise, timesteps=timesteps)
        with autocast(device_type="cuda", enabled=amp_enabled):
            eps_pred = ddp_whole_unet(z_t, timesteps=timesteps)
            loss_std = F.mse_loss(eps_pred.float(), noise.float())

        do_aux = (aux_every <= 1) or (step % aux_every == 0)
        if do_aux:
            t_aux = random.randint(int(taux_min), int(taux_max))

            with torch.no_grad():
                payload = base.build_expert_composite_branch(
                    whole_unet=ddp_whole_unet.module,
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

            active = timesteps <= int(t_aux)
            if bool(active.any().item()):
                z_pred_tm1 = base._predict_prev_from_eps_ddpm(
                    scheduler=scheduler,
                    z_t=z_t,
                    eps_pred=eps_pred,
                    timesteps=timesteps,
                )
                t_prev = torch.clamp(timesteps - 1, min=0)
                noise_aux = torch.randn_like(z_coarse0)
                z_coarse_tm1 = scheduler.add_noise(
                    original_samples=z_coarse0,
                    noise=noise_aux,
                    timesteps=t_prev,
                )
                z_whole_tm1 = scheduler.add_noise(
                    original_samples=z_whole0,
                    noise=noise_aux,
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

        loss_mean = _dist_mean_value(loss, world_size=world_size)
        std_mean = _dist_mean_value(loss_std, world_size=world_size)
        aux_mean = _dist_mean_value(loss_aux, world_size=world_size)
        seam_mean = _dist_mean_value(loss_seam, world_size=world_size)

        running += loss_mean
        running_n += 1
        avg_loss = running / max(1, running_n)

        if rank == 0:
            history["loss_curve"].append(loss_mean)
            epoch = int(max(0, global_step - 1) // max(1, steps_per_epoch))
            progress.set_description(f"step {global_step}/{max_steps} (ep~{epoch})")
            progress.set_postfix(
                {
                    "loss": f"{avg_loss:.6f}",
                    "std": f"{std_mean:.5f}",
                    "aux": f"{aux_mean:.5f}",
                    "seam": f"{seam_mean:.5f}",
                    "taux": "-" if t_aux < 0 else int(t_aux),
                    "aux_step": int(do_aux),
                    "sf": f"{whole_scale_factor:.4f}",
                }
            )
            progress.update(1)

            if global_step % 200 == 0:
                history["train_loss"].append(
                    {
                        "step": global_step,
                        "loss": avg_loss,
                        "loss_std": std_mean,
                        "loss_aux": aux_mean,
                        "loss_seam": seam_mean,
                    }
                )

            if (last_every > 0) and (global_step % last_every == 0):
                _update_epoch_loss_curve()
                base._save_ckpt(
                    os.path.join(outdir, "UNET_last.pt"),
                    ddp_whole_unet.module,
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
                base.plot_unet_loss(
                    history["epoch_loss_curve"],
                    title=f"UNET Epoch-Average Loss_step{global_step}",
                    outdir=outdir,
                    filename="UNET_loss.png",
                )

            if (ckpt_every > 0) and (global_step % ckpt_every == 0):
                _update_epoch_loss_curve()
                ckpt_path = os.path.join(outdir, f"UNET_step{global_step:09d}.pt")
                base._save_ckpt(
                    ckpt_path,
                    ddp_whole_unet.module,
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

        full_eval_trigger = (eval_every > 0) and (global_step % eval_every == 0)
        simple_eval_trigger = (
            (not full_eval_trigger)
            and (simple_eval_every > 0)
            and (global_step % simple_eval_every == 0)
        )

        if full_eval_trigger:
            if rank == 0 and val_loader is not None and len(val_loader) > 0:
                metrics = base.eval_taux_fast(
                    whole_unet=ddp_whole_unet.module,
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
                ddp_whole_unet.train()
            dist.barrier(device_ids=[local_rank])
        elif simple_eval_trigger:
            if rank == 0 and val_loader is not None and len(val_loader) > 0:
                metrics = base.eval_ldm_loss_only(
                    whole_unet=ddp_whole_unet.module,
                    whole_ae=whole_ae,
                    val_loader=val_loader,
                    device=device,
                    whole_scale_factor=whole_scale_factor,
                    torch_autocast=torch_autocast,
                    val_batches=simple_eval_val_batches,
                )
                history["simple_eval"].append({"step": global_step, **metrics})
                ddp_whole_unet.train()
            dist.barrier(device_ids=[local_rank])

    if rank == 0:
        progress.close()
        _update_epoch_loss_curve()
        base._save_ckpt(
            os.path.join(outdir, "UNET_last.pt"),
            ddp_whole_unet.module,
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
        base.plot_unet_loss(
            history["epoch_loss_curve"],
            title=f"UNET Epoch-Average Loss_step{global_step}",
            outdir=outdir,
            filename="UNET_loss.png",
        )
    dist.barrier(device_ids=[local_rank])


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file with argument defaults.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="DDP trainer for train_3d_brain_ldm_aux_taux_addition.py.",
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
    ap.add_argument("--seed", type=int, default=1017)

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

    # Compose controls
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
    ap.add_argument(
        "--ddp_find_unused_parameters",
        default=True,
        help="Enable DDP unused-parameter detection for conditional/sparse-grad branches.",
    )

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
    if str(args.scaffold_mode) != "forward_noise":
        print(f"Training scaffold_mode defaults to forward-noised for noise coupling.")
        args.scaffold_mode = "forward_noise"
        args.eval_scaffold_mode = args.scaffold_mode    
    if str(args.eval_scaffold_mode).strip() != "" and str(args.eval_scaffold_mode) not in valid_scaffold:
        ap.error("Require eval_scaffold_mode in {'sample','forward_noise'} (or empty).")
    if float(args.inject_alpha) < 0.0:
        ap.error("Require inject_alpha >= 0.")
    if float(args.lambda_seam) < 0.0:
        ap.error("Require lambda_seam >= 0.")

    rank = -1
    local_rank = -1
    world_size = -1

    try:
        rank, world_size, local_rank = setup_ddp()
        torch_autocast = parse_bool(args.torch_autocast)

        if rank == 0:
            run_name = (
                f"{args.out_prefix}_{args.out_postfix}" if args.out_prefix else f"{args.out_postfix}"
            )
        else:
            run_name = ""
        run_name = broadcast_str(run_name, src=0)

        experiment_dir = os.path.join(args.outdir, run_name)
        if rank == 0:
            os.makedirs(experiment_dir, exist_ok=True)
        dist.barrier(device_ids=[local_rank])
        args.outdir = experiment_dir

        resume_ckpt = args.resume_ckpt
        if resume_ckpt and os.path.isdir(resume_ckpt):
            resume_ckpt = os.path.join(resume_ckpt, "UNET_last.pt")
        if resume_ckpt and not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")

        if rank == 0:
            cfg = vars(args).copy()
            cfg["resolved_resume_ckpt"] = resume_ckpt
            with open(os.path.join(args.outdir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, sort_keys=True)
            print(f"\nOutput dir: {args.outdir}\n")

        spacing = base._parse_float_tuple(args.spacing)
        size = base._parse_int_tuple(args.size)
        hemi_size = base._parse_int_tuple(args.hemi_size)
        sub_size = base._parse_int_tuple(args.sub_size)

        device = torch.device(f"cuda:{local_rank}")
        base.seed_all(int(args.seed) + rank)
        random.seed(int(args.seed) + rank)

        n_samples = None if str(args.n_samples).upper() == "ALL" else int(args.n_samples)
        train_transforms = base.build_image_transforms(spacing=spacing, whole_size=size)
        train_loader, val_loader, _, train_sampler = make_dataloaders_from_csv_ddp(
            csv_path=args.csv,
            image_key=args.whole_key,
            rank=rank,
            conditions=("age", "sex", "group", "condition"),
            train_transforms=train_transforms,
            n_samples=n_samples,
            data_split_json_path=args.data_split_json_path,
            batch_size=args.batch,
            num_workers=args.workers,
            seed=int(args.seed),
        )

        if rank == 0:
            print(f"Number of training samples: {len(train_loader.dataset)}")
            if val_loader is not None:
                print(f"Number of validation samples: {len(val_loader.dataset)}")

        template_dir = args.template_dir
        lhemi_mask_path = base._resolve_path(template_dir, args.lhemi_template_name)
        rhemi_mask_path = base._resolve_path(template_dir, args.rhemi_template_name)
        sub_mask_path = base._resolve_path(template_dir, args.sub_template_name)
        whole_shape = tuple(int(v) for v in next(iter(train_loader))["image"].shape[-3:])
        mask_ctx = base.load_template_masks(
            lhemi_mask_path=lhemi_mask_path,
            rhemi_mask_path=rhemi_mask_path,
            sub_mask_path=sub_mask_path,
            spacing=spacing,
            whole_shape=whole_shape,
            device=device,
        )

        # Build models.
        whole_ae = base.AutoencoderKL(
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
        whole_unet = base.DiffusionModelUNet(
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

        hemi_ae = base.AutoencoderKL(
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
        hemi_unet = base.DiffusionModelUNet(
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

        sub_ae = base.AutoencoderKL(
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
        sub_unet = base.DiffusionModelUNet(
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
        base._load_ckpt_into_ae(whole_ae, args.whole_ae_ckpt, device)
        if args.whole_unet_ckpt:
            base._load_model_weights_shape_safe(whole_unet, args.whole_unet_ckpt, device)

        base._load_ckpt_into_ae(hemi_ae, args.hemi_ae_ckpt, device)
        base._load_model_weights_shape_safe(hemi_unet, args.hemi_unet_ckpt, device)
        if args.hemi_conditioner_ckpt:
            base._load_conditioner_weights_from_ckpt(hemi_conditioner, args.hemi_conditioner_ckpt, device)
        else:
            base._load_conditioner_weights_from_ckpt(hemi_conditioner, args.hemi_unet_ckpt, device)

        base._load_ckpt_into_ae(sub_ae, args.sub_ae_ckpt, device)
        base._load_model_weights_shape_safe(sub_unet, args.sub_unet_ckpt, device)

        # Scale factors: args override > ckpt extra > auto estimate in train loop.
        whole_sf = args.whole_scale_factor if args.whole_scale_factor > 0 else None
        hemi_sf = args.hemi_scale_factor if args.hemi_scale_factor > 0 else None
        sub_sf = args.sub_scale_factor if args.sub_scale_factor > 0 else None
        if whole_sf is None:
            for p in [args.whole_unet_ckpt, resume_ckpt]:
                if p:
                    whole_sf = base._read_scale_factor_from_ckpt(p, device)
                    if whole_sf is not None:
                        break
        if hemi_sf is None:
            hemi_sf = base._read_scale_factor_from_ckpt(args.hemi_unet_ckpt, device)
        if sub_sf is None:
            sub_sf = base._read_scale_factor_from_ckpt(args.sub_unet_ckpt, device)

        ddp_find_unused = parse_bool(args.ddp_find_unused_parameters)
        if rank == 0:
            print(f"DDP find_unused_parameters={ddp_find_unused}")

        ddp_whole_unet = DDP(
            whole_unet,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
        )

        train_ldm_aux_taux_ddp(
            ddp_whole_unet=ddp_whole_unet,
            whole_ae=whole_ae,
            hemi_ae=hemi_ae,
            hemi_unet=hemi_unet,
            hemi_conditioner=hemi_conditioner,
            sub_ae=sub_ae,
            sub_unet=sub_unet,
            train_loader=train_loader,
            train_sampler=train_sampler,
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
            compose_flip_lhemi=parse_bool(args.compose_flip_lhemi),
            compose_flip_rhemi=parse_bool(args.compose_flip_rhemi),
            compose_valid_thresh=float(args.compose_valid_thresh),
            bg_value=float(args.bg_value),
            inject_alpha=float(args.inject_alpha),
            torch_autocast=torch_autocast,
            device=device,
            outdir=args.outdir,
            ckpt_every=int(args.ckpt_every),
            last_every=int(args.last_every),
            eval_every=int(args.eval_every),
            simple_eval_every=int(args.simple_eval_every),
            eval_n=int(args.eval_n),
            eval_val_batches=int(args.eval_val_batches),
            simple_eval_val_batches=int(args.simple_eval_val_batches),
            resume_ckpt=resume_ckpt,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
