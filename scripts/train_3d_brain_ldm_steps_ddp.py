"""
DDP trainer for train_3d_brain_ldm_steps.py (step-based LDM stage).

Example (4 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_3d_brain_ldm_steps_ddp.py \
  --config configs/whole_brain_LDM_spacing1p5.json
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
from monai import transforms
from monai.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler

# Import single-GPU script as base utilities.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_3d_brain_ldm_steps as base  # noqa: E402


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


def make_dataloaders_from_csv_ddp(
    csv_path: str,
    rank: int,
    conditions=None,
    train_transforms=None,
    n_samples="ALL",
    data_split_json_path: str = "data/patient_splits_image_ids_75_10_15.json",
    batch_size: int = 1,
    num_workers: int = 8,
    seed: int = 1017,
):
    if conditions is None:
        conditions = ["age", "sex", "vol", "group"]

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
    train_added = 0

    for _, row in df.iterrows():
        image_id = row["imageID"]
        if image_id not in image_ids:
            continue

        sample = {"image": row["image"]}
        for c in conditions:
            sample[c] = row[c]

        split = image_ids[image_id]
        if split == "train":
            train_data.append(sample)
            train_added += 1
            if n_samples != "ALL" and train_added >= n_samples:
                break
        elif split == "val":
            val_data.append(sample)
        elif split == "test":
            test_data.append(sample)

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=train_transforms)
    test_ds = Dataset(data=test_data, transform=train_transforms)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Eval runs on rank-0 only to avoid duplicating sample generation/checkpoints.
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


def train_ldm_steps_ddp(
    *,
    ddp_unet: DDP,
    train_loader,
    train_sampler: DistributedSampler,
    val_loader,
    autoencoder,
    max_steps: int,
    lr: float = 1e-4,
    scale_factor: Optional[float] = None,
    torch_autocast: bool = True,
    device=torch.device("cuda"),
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
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    amp_enabled = bool(torch_autocast and device.type == "cuda")

    if scale_factor is None:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
            batch0 = next(iter(train_loader))
            z = autoencoder.encode_stage_2_inputs(batch0["image"].to(device))
            local_std = torch.std(z).detach()
        std_tensor = local_std.to(device=device, dtype=torch.float64)
        dist.all_reduce(std_tensor, op=dist.ReduceOp.SUM)
        std_tensor = std_tensor / max(1, world_size)
        scale_factor = float(1.0 / max(std_tensor.item(), 1e-8))
        if rank == 0:
            print(f"[LDM-DDP] scale_factor set to {scale_factor}")
    inferer.scale_factor = scale_factor

    optimizer = torch.optim.AdamW(ddp_unet.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    global_step = 0
    history: Dict[str, Any] = {
        "train_loss": [],
        "simple_eval": [],
        "eval": [],
        "loss_curve": [],
        "epoch_loss_curve": [],
    }

    if resume_ckpt:
        global_step, _, extra = base._load_ckpt_into_unet(
            ddp_unet.module,
            resume_ckpt,
            device,
            opt=optimizer,
            scaler=scaler,
        )
        if isinstance(extra, dict):
            loaded_history = extra.get("history")
            if isinstance(loaded_history, dict):
                history = loaded_history
            if "scale_factor" in extra:
                scale_factor = float(extra["scale_factor"])
                inferer.scale_factor = scale_factor
        if rank == 0:
            print(f"[resume] global_step={global_step}, scale_factor={scale_factor}")

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
    ddp_unet.train()

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
        ncols=110,
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

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(images) * scale_factor

        noise = torch.randn_like(z)
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            noise_pred = inferer(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=ddp_unet,
                noise=noise,
                timesteps=timesteps,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step = step + 1

        loss_tensor = loss.detach().to(dtype=torch.float64)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_mean = float(loss_tensor.item() / max(1, world_size))

        running += loss_mean
        running_n += 1
        avg_loss = running / max(1, running_n)

        if rank == 0:
            history["loss_curve"].append(loss_mean)
            epoch = int(max(0, global_step - 1) // max(1, steps_per_epoch))
            progress.set_description(f"step {global_step}/{max_steps} (ep~{epoch})")
            progress.set_postfix({"loss": f"{avg_loss:.6f}", "sf": f"{scale_factor:.4f}"})
            progress.update(1)
            if global_step % 200 == 0:
                history["train_loss"].append({"step": global_step, "loss": avg_loss})

            if (last_every > 0) and (global_step % last_every == 0):
                _update_epoch_loss_curve()
                base._save_ckpt(
                    os.path.join(outdir, "UNET_last.pt"),
                    ddp_unet.module,
                    optimizer,
                    scaler,
                    global_step=global_step,
                    epoch=epoch,
                    extra={"scale_factor": float(scale_factor), "history": history},
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
                    ddp_unet.module,
                    optimizer,
                    scaler,
                    global_step=global_step,
                    epoch=epoch,
                    extra={"scale_factor": float(scale_factor), "history": history},
                )

        full_eval_trigger = (eval_every > 0) and (global_step % eval_every == 0)
        simple_eval_trigger = (
            (not full_eval_trigger)
            and (simple_eval_every > 0)
            and (global_step % simple_eval_every == 0)
        )

        if full_eval_trigger:
            if rank == 0 and val_loader is not None:
                metrics = base.eval_ldm_fast(
                    unet=ddp_unet.module,
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
                ddp_unet.train()
            dist.barrier(device_ids=[local_rank])
        elif simple_eval_trigger:
            if rank == 0 and val_loader is not None:
                metrics = base.eval_ldm_loss_only(
                    unet=ddp_unet.module,
                    autoencoder=autoencoder,
                    val_loader=val_loader,
                    device=device,
                    scale_factor=scale_factor,
                    torch_autocast=torch_autocast,
                    val_batches=simple_eval_val_batches,
                )
                history["simple_eval"].append({"step": global_step, **metrics})
                ddp_unet.train()
            dist.barrier(device_ids=[local_rank])

    if rank == 0:
        progress.close()
        _update_epoch_loss_curve()
        base._save_ckpt(
            os.path.join(outdir, "UNET_last.pt"),
            ddp_unet.module,
            optimizer,
            scaler,
            global_step=global_step,
            epoch=int(global_step // max(1, steps_per_epoch)),
            extra={"scale_factor": float(scale_factor), "history": history},
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
        description="DDP wrapper for train_3d_brain_ldm_steps.py.",
        parents=[pre_ap],
    )
    ap.add_argument("--csv", default="", help="Path to CSV with columns: image[/path], sex, age, vol, [target_label]")
    ap.add_argument(
        "--data_split_json_path",
        default="data/patient_splits_image_ids_75_10_15.json",
        help="JSON file with train/val/test imageID splits.",
    )
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True, help="Use torch autocast to accelerate: True or False.")
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--train_val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1017)

    ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="both")

    ap.add_argument("--ae_latent_ch", type=int, default=8)
    ap.add_argument("--ae_num_channels", default="64,128,256")
    ap.add_argument(
        "--ae_attention_levels",
        default="0,0,0",
        help="Comma-separated binary flags for attention at each AE level (e.g., 0,0,1)",
    )
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt")

    ap.add_argument("--max_steps", type=int, default=120_000, help="Total number of optimizer steps for step-based LDM training.")
    ap.add_argument("--ckpt_every", type=int, default=10_000, help="Save packaged UNet checkpoint every N steps.")
    ap.add_argument("--last_every", type=int, default=1_000, help="Overwrite UNET_last.pt every N steps for easy resume.")
    ap.add_argument("--eval_every", type=int, default=10_000, help="Run full fast eval + sample generation every N steps.")
    ap.add_argument("--simple_eval_every", type=int, default=2_000, help="Run cheap val-loss-only eval every N steps (excluding full-eval steps).")
    ap.add_argument("--eval_n", type=int, default=8, help="Number of generated 3D volumes during full fast eval.")
    ap.add_argument("--ddim_steps", type=int, default=50, help="DDIM denoising steps for fast eval sampling.")
    ap.add_argument("--eval_val_batches", type=int, default=8, help="Number of val batches for full fast eval.")
    ap.add_argument("--simple_eval_val_batches", type=int, default=4, help="Number of val batches for simple eval.")
    ap.add_argument("--resume_ckpt", default="", help="Path to a packaged UNET_step*.pt to resume")

    ap.add_argument("--ldm_epochs", type=int, default=150)
    ap.add_argument("--ldm_lr", type=float, default=1e-4)
    ap.add_argument("--ldm_use_cond", default="False", help="Use [part_vol_norm,sex,age] conditioning")
    ap.add_argument("--ldm_num_channels", default="256,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_ckpt", default="", help="Resume UNet weights (optional)")
    ap.add_argument("--ldm_sample_every", type=int, default=25, help="Synthesize samples every N epochs")

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
            for k, v in list(cfg.items()):
                if isinstance(v, tuple):
                    cfg[k] = list(v)
            with open(os.path.join(args.outdir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, sort_keys=True)
            print(f"\nOutput dir: {args.outdir}\n")

        spacing = tuple(float(x) for x in args.spacing.split(","))
        size = tuple(int(x) for x in args.size.split(","))
        channel = 0
        keys = ["image"]
        train_transforms = transforms.Compose(
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

        device = torch.device(f"cuda:{local_rank}")
        base.seed_all(args.seed + rank)
        random.seed(args.seed + rank)

        try:
            n_samples = int(args.n_samples)
        except Exception:
            n_samples = args.n_samples

        train_loader, val_loader, test_loader, train_sampler = make_dataloaders_from_csv_ddp(
            args.csv,
            rank=rank,
            conditions=["age", "sex", "vol", "group"],
            train_transforms=train_transforms,
            n_samples=n_samples,
            data_split_json_path=args.data_split_json_path,
            batch_size=args.batch,
            num_workers=args.workers,
            seed=args.seed,
        )

        if rank == 0:
            print(f"Number of training samples: {len(train_loader.dataset)}")
            if val_loader is not None:
                print(f"Number of validation samples: {len(val_loader.dataset)}")
            if test_loader is not None:
                print(f"Number of test samples: {len(test_loader.dataset)}")

        ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
        ae_latent_ch = int(args.ae_latent_ch)
        ae_attention_levels = tuple(bool(int(x)) for x in args.ae_attention_levels.split(","))

        autoencoder = base.AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=ae_num_channels,
            latent_channels=ae_latent_ch,
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-06,
            attention_levels=ae_attention_levels,
        ).to(device)
        base._load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)

        ldm_num_channels = tuple(int(x) for x in args.ldm_num_channels.split(","))
        ldm_num_head_channels = tuple(int(x) for x in args.ldm_num_head_channels.split(","))
        unet = base.DiffusionModelUNet(
            spatial_dims=3,
            in_channels=ae_latent_ch,
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
        ).to(device)

        scale_factor = None
        if args.ldm_ckpt:
            base._load_ckpt_into_unet(unet, args.ldm_ckpt, device)
            try:
                ckpt = torch.load(args.ldm_ckpt, map_location=device, weights_only=False)
                scale_factor = float(ckpt["extra"]["scale_factor"])
                if rank == 0:
                    print(f"Loading scale factor = {scale_factor}")
            except Exception:
                scale_factor = None

        # Some UNet branches are not always active per-step; enable unused-param detection
        # to avoid "Expected to have finished reduction..." errors in DDP.
        ddp_unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        if args.stage in {"ldm", "both"}:
            train_ldm_steps_ddp(
                ddp_unet=ddp_unet,
                train_loader=train_loader,
                train_sampler=train_sampler,
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
