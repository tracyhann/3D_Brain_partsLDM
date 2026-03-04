from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

# Reuse architecture/checkpoint/data-loader utilities from segm-LDM training script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_3d_brain_ldm_segm as base  # noqa: E402


def _resolve_ckpt_path(path: str, default_name: str = "UNET_last.pt") -> str:
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


def _run_name_from_ckpt_path(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if p.is_dir():
        return p.name
    if p.parent.name:
        return p.parent.name
    return p.stem


def _extract_image_ids(batch: dict, batch_size: int, fallback_start: int) -> list[str]:
    v = batch.get("imageID", None)
    if v is None:
        return [f"test_{fallback_start + i:05d}" for i in range(batch_size)]
    if isinstance(v, (list, tuple)):
        out = [str(v[i]) if i < len(v) else f"test_{fallback_start + i:05d}" for i in range(batch_size)]
        return out
    if torch.is_tensor(v):
        if v.ndim == 0:
            return [str(v.item()) for _ in range(batch_size)]
        flat = v.reshape(-1)
        out = [str(flat[i].item()) if i < flat.numel() else f"test_{fallback_start + i:05d}" for i in range(batch_size)]
        return out
    return [str(v) for _ in range(batch_size)]


def _extract_affine_from_batched_image(image: torch.Tensor, idx: int) -> np.ndarray:
    affine = np.eye(4, dtype=np.float32)
    if not hasattr(image, "affine"):
        return affine
    aff = image.affine
    if isinstance(aff, torch.Tensor):
        aff = aff.detach().cpu().numpy()
    aff = np.asarray(aff)
    if aff.ndim == 3:
        if idx < aff.shape[0]:
            aff = aff[idx]
        elif aff.shape[0] > 0:
            aff = aff[0]
    if aff.shape == (4, 4):
        affine = aff.astype(np.float32)
    return affine


def _extract_source_path(batch: dict, idx: int) -> str:
    meta = batch.get("image_meta_dict", None)
    if isinstance(meta, dict) and "filename_or_obj" in meta:
        v = meta["filename_or_obj"]
        if isinstance(v, (list, tuple)):
            if idx < len(v):
                return str(v[idx])
            if len(v) > 0:
                return str(v[0])
        return str(v)
    return ""


def _step_prev_sample(step_out):
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, (tuple, list)):
        return step_out[0]
    if isinstance(step_out, dict) and "prev_sample" in step_out:
        return step_out["prev_sample"]
    return step_out


def _scan_existing_outputs(outdir: str, save_prefix: str) -> dict[int, dict[str, Any]]:
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


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Inference for segmentation-mask conditioned whole-brain LDM.",
        parents=[pre_ap],
    )

    # Data
    ap.add_argument("--csv", default="data/processed_parts/whole_brain+3parts+masks_0206.csv")
    ap.add_argument("--data_split_json_path", default="data/patient_splits_image_ids_75_10_15.json")
    ap.add_argument("--image_key", default="whole_brain")
    ap.add_argument("--whole_mask_key", default="whole_brain_mask")
    ap.add_argument("--lhemi_mask_key", default="lhemi_mask")
    ap.add_argument("--rhemi_mask_key", default="rhemi_mask")
    ap.add_argument("--sub_mask_key", default="sub_mask")
    ap.add_argument("--spacing", default="1.5,1.5,1.5")
    ap.add_argument("--size", default="128,128,128")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--torch_autocast", default=True)
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--seed", type=int, default=1017)
    ap.add_argument(
        "--conditions",
        default="age,sex,group,condition,imageID",
        help="CSV fields kept in loader samples; include imageID for output naming.",
    )

    # Architecture + checkpoints
    ap.add_argument("--ae_latent_ch", type=int, default=8)
    ap.add_argument("--ae_num_channels", default="64,128,256")
    ap.add_argument("--ae_attention_levels", default="0,0,0")
    ap.add_argument("--ae_ckpt", default="", help="Required: pretrained AE checkpoint.")
    ap.add_argument("--ldm_num_channels", default="256,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_ckpt", default="", help="Required UNet checkpoint or run dir.")
    ap.add_argument("--resume_ckpt", default="", help="Optional override for --ldm_ckpt.")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument(
        "--scale_factor",
        type=float,
        default=-1.0,
        help="Override latent scale factor. If <=0, uses checkpoint extra or first test batch estimate.",
    )

    # Output controls
    ap.add_argument("--samples_root", default="samples")
    ap.add_argument("--save_prefix", default="sample")
    ap.add_argument("--max_test_samples", type=int, default=-1)
    ap.add_argument("--save_real_pair", default=False)

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
    if not str(args.ae_ckpt).strip():
        ap.error("--ae_ckpt is required.")
    if int(args.batch) < 1:
        ap.error("Require batch >= 1.")
    if int(args.workers) < 0:
        ap.error("Require workers >= 0.")
    if int(args.ddim_steps) < 1 or int(args.ddim_steps) > 1000:
        ap.error("Require 1 <= ddim_steps <= 1000.")

    requested_ckpt = args.resume_ckpt if str(args.resume_ckpt).strip() else args.ldm_ckpt
    if not str(requested_ckpt).strip():
        ap.error("Either --ldm_ckpt or --resume_ckpt is required.")
    resolved_ldm_ckpt = _resolve_ckpt_path(str(requested_ckpt), default_name="UNET_last.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch_autocast = base.str2bool(args.torch_autocast)
    amp_enabled = bool(torch_autocast and device.type == "cuda")
    base.seed_all(int(args.seed))

    run_ckpt_name = _run_name_from_ckpt_path(resolved_ldm_ckpt)
    outdir = os.path.join(args.samples_root, run_ckpt_name)
    os.makedirs(outdir, exist_ok=True)

    conditions = [x.strip() for x in str(args.conditions).split(",") if x.strip()]
    if "imageID" not in conditions:
        conditions.append("imageID")

    spacing = tuple(float(x) for x in str(args.spacing).split(","))
    whole_size = tuple(int(x) for x in str(args.size).split(","))
    n_samples = None if str(args.n_samples).upper() == "ALL" else int(args.n_samples)

    test_transforms = base.build_transforms(
        image_key="image",
        whole_mask_key="whole_brain_mask",
        lhemi_mask_key="lhemi_mask",
        rhemi_mask_key="rhemi_mask",
        sub_mask_key="sub_mask",
        spacing=spacing,
        whole_size=whole_size,
    )

    _, _, test_loader = base.make_dataloaders_from_csv(
        csv_path=args.csv,
        image_key=args.image_key,
        whole_mask_key=args.whole_mask_key,
        lhemi_mask_key=args.lhemi_mask_key,
        rhemi_mask_key=args.rhemi_mask_key,
        sub_mask_key=args.sub_mask_key,
        conditions=tuple(conditions),
        train_transforms=test_transforms,
        n_samples=n_samples,
        data_split_json_path=args.data_split_json_path,
        batch_size=int(args.batch),
        num_workers=int(args.workers),
    )
    if len(test_loader.dataset) < 1:
        raise RuntimeError("Test split is empty; cannot run inference.")

    total = len(test_loader.dataset)
    if int(args.max_test_samples) > 0:
        total = min(total, int(args.max_test_samples))
    if total < 1:
        raise RuntimeError("No samples requested after applying --max_test_samples.")
    if total != 291 and int(args.max_test_samples) < 0:
        print(f"Warning: expected 291 test samples, got {total}.")

    ae_num_channels = tuple(int(x) for x in str(args.ae_num_channels).split(","))
    ae_attention_levels = tuple(bool(int(x)) for x in str(args.ae_attention_levels).split(","))
    ae_latent_ch = int(args.ae_latent_ch)
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=ae_num_channels,
        latent_channels=ae_latent_ch,
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=ae_attention_levels,
    ).to(device)
    base._load_ckpt_into_ae(autoencoder, str(args.ae_ckpt), device)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    ldm_num_channels = tuple(int(x) for x in str(args.ldm_num_channels).split(","))
    ldm_num_head_channels = tuple(int(x) for x in str(args.ldm_num_head_channels).split(","))
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
    ).to(device)
    _, _, extra = base._load_ckpt_into_unet(unet, resolved_ldm_ckpt, device)
    unet.eval()

    first_batch = next(iter(test_loader))
    scale_factor: Optional[float]
    if float(args.scale_factor) > 0:
        scale_factor = float(args.scale_factor)
        print(f"Using scale_factor from CLI: {scale_factor:.6f}")
    elif isinstance(extra, dict) and ("scale_factor" in extra):
        scale_factor = float(extra["scale_factor"])
        print(f"Using scale_factor from checkpoint extra: {scale_factor:.6f}")
    else:
        with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
            z_ref = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))
        z_std = max(float(torch.std(z_ref).item()), 1e-8)
        scale_factor = float(1.0 / z_std)
        print(f"Estimated scale_factor from first test batch: {scale_factor:.6f}")

    infer_cfg = vars(args).copy()
    infer_cfg["resolved_ldm_ckpt"] = resolved_ldm_ckpt
    infer_cfg["run_ckpt_name"] = run_ckpt_name
    infer_cfg["resolved_scale_factor"] = float(scale_factor)
    with open(os.path.join(outdir, "infer_args.json"), "w", encoding="utf-8") as f:
        json.dump(infer_cfg, f, indent=2, sort_keys=True)

    ddim = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=False,
    )
    ddim.set_timesteps(num_inference_steps=int(args.ddim_steps))

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
    batch_size_cfg = int(args.batch)
    for i, batch in enumerate(tqdm(test_loader, desc="Segm-LDM inference", ncols=120)):
        batch_start = i * batch_size_cfg
        if batch_start >= total:
            break

        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
            z_ref = autoencoder.encode_stage_2_inputs(images) * float(scale_factor)
        cond_lat = base.build_mask_condition_latents(
            batch=batch,
            device=device,
            latent_shape=tuple(z_ref.shape[-3:]),
        )

        z = torch.randn_like(z_ref)
        for t in ddim.timesteps:
            t_int = int(t.item()) if hasattr(t, "item") else int(t)
            t_batch = torch.full((z.shape[0],), t_int, device=device, dtype=torch.long)
            with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
                model_input = torch.cat([z, cond_lat], dim=1)
                eps = unet(model_input, timesteps=t_batch)
            z = _step_prev_sample(ddim.step(model_output=eps, timestep=t_int, sample=z))

        with torch.no_grad(), autocast(device_type="cuda", enabled=amp_enabled):
            x_gen = autoencoder.decode_stage_2_outputs(z / float(scale_factor))
        x_gen_cpu = x_gen.detach().cpu()
        image_cpu = images.detach().cpu()

        bsz = int(x_gen_cpu.shape[0])
        ids = _extract_image_ids(batch, bsz, fallback_start=batch_start)
        for bi in range(bsz):
            test_index = batch_start + bi
            if test_index >= total:
                break
            if test_index < resume_from:
                continue

            image_id = ids[bi]
            safe_id = _safe_component(image_id)
            out_name = f"{args.save_prefix}_{test_index:0{index_width}d}_{safe_id}.nii.gz"
            out_path = os.path.join(outdir, out_name)
            if os.path.exists(out_path):
                skipped_existing += 1
                if test_index not in manifest_by_index:
                    manifest_by_index[test_index] = {
                        "test_index": int(test_index),
                        "imageID": str(image_id),
                        "output_file": out_name,
                        "resume_stub": True,
                    }
                continue

            affine = _extract_affine_from_batched_image(batch["image"], bi)
            nib.save(
                nib.Nifti1Image(
                    x_gen_cpu[bi, 0].float().numpy().astype(np.float32),
                    affine,
                ),
                out_path,
            )

            real_name = ""
            if base.str2bool(args.save_real_pair):
                real_name = f"real_{test_index:0{index_width}d}_{safe_id}.nii.gz"
                real_path = os.path.join(outdir, real_name)
                nib.save(
                    nib.Nifti1Image(
                        image_cpu[bi, 0].float().numpy().astype(np.float32),
                        affine,
                    ),
                    real_path,
                )

            manifest_by_index[test_index] = {
                "test_index": int(test_index),
                "imageID": str(image_id),
                "output_file": out_name,
                "real_pair_file": real_name,
                "real_source_file": _extract_source_path(batch, bi),
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


if __name__ == "__main__":
    main()

