import argparse
import json
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pprint import pprint

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from generative.networks.schedulers import DDPMScheduler
from monai.bundle import ConfigParser
from monai.utils import first, set_determinism

from morphldm.autoencoderkl import (
    AutoencoderKLConditionalTemplateRegistration,
    AutoencoderKLTemplateRegistration,
)
from morphldm.data_utils import (
    build_train_transforms,
    make_dataloaders_from_csv,
    parse_n_samples,
)
from morphldm.inferer import LatentDiffusionInferer


def _rescale_to_01(images: torch.Tensor) -> torch.Tensor:
    # Match autoencoder/diffusion training preprocessing (input in [-1, 1]).
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def _extract_reference_affine(image_tensor) -> np.ndarray:
    affine = None
    if hasattr(image_tensor, "affine"):
        affine = image_tensor.affine
    if affine is None:
        return np.eye(4, dtype=np.float32)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    affine = np.asarray(affine)
    if affine.ndim == 3:
        affine = affine[0]
    if affine.shape != (4, 4):
        return np.eye(4, dtype=np.float32)
    return affine.astype(np.float32)


def _resolve_path(path_value: str, base_dir: str | None = None) -> str:
    path = os.path.expanduser(path_value)
    if os.path.isabs(path):
        return path
    if base_dir is None:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def _parse_age_value(value, image_id: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid age value for imageID={image_id}: {value!r}") from exc


def _parse_sex_value(value, image_id: str) -> float:
    if isinstance(value, str):
        key = value.strip().lower()
        mapping = {
            "m": 1.0,
            "male": 1.0,
            "man": 1.0,
            "f": 0.0,
            "female": 0.0,
            "woman": 0.0,
        }
        if key in mapping:
            return mapping[key]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sex value for imageID={image_id}: {value!r}") from exc


def _safe_id_for_filename(value: str) -> str:
    text = str(value)
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)
    return safe if safe else "unknown"


def _find_first_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"CSV missing required {label} column. Tried: {list(candidates)}")


def _transform_age_value(age: float, transform: str) -> float:
    if transform == "none":
        return float(age)
    if transform == "divide_by_100":
        return float(age) / 100.0
    if transform == "auto":
        # Training metadata in this project is normalized (e.g., age ~0.62..0.90).
        # If a CSV provides raw age in years (e.g., 61..90), normalize by 100.
        return float(age) / 100.0 if float(age) > 2.0 else float(age)
    raise ValueError(f"Unknown age transform: {transform}")


def _get_first_present_key(mapping: dict, candidates: tuple[str, ...], label: str, image_id: str):
    for key in candidates:
        if key in mapping:
            return mapping[key]
    raise ValueError(
        f"Metadata for imageID={image_id} missing required {label} key. Tried: {list(candidates)}"
    )


def _load_test_split_conditions(args, config_dir: str):
    if args.dataset_type != "CSV":
        raise ValueError(
            f"condition_source='test_split' requires dataset_type='CSV', got {args.dataset_type!r}"
        )
    if not hasattr(args, "csv_path") or not hasattr(args, "data_split_json_path"):
        raise ValueError(
            "condition_source='test_split' requires csv_path and data_split_json_path in config."
        )

    condition_metadata_raw = str(getattr(args, "condition_metadata_path", "")).strip()
    if condition_metadata_raw:
        metadata_path = _resolve_path(condition_metadata_raw, base_dir=config_dir)
    else:
        # Backward compatibility: legacy key name.
        condition_csv_raw = str(getattr(args, "condition_csv_path", "")).strip()
        if condition_csv_raw:
            metadata_path = _resolve_path(condition_csv_raw, base_dir=config_dir)
        else:
            metadata_path = _resolve_path(str(args.csv_path), base_dir=config_dir)
    split_path = _resolve_path(str(args.data_split_json_path), base_dir=config_dir)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Condition metadata file not found: {metadata_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split JSON file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)
    if "test" not in split_dict:
        raise ValueError(f"Split JSON has no 'test' key: {split_path}")

    test_ids = [str(v) for v in split_dict["test"]]
    if not test_ids:
        raise ValueError(f"'test' split is empty in: {split_path}")

    metadata_ext = os.path.splitext(metadata_path)[1].lower()
    row_by_id = {}
    metadata_kind = None
    age_col = None
    sex_col = None
    image_col = None
    if metadata_ext == ".json":
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        if not isinstance(metadata_dict, dict):
            raise ValueError(
                f"JSON metadata must be an object keyed by imageID, got {type(metadata_dict).__name__}"
            )
        row_by_id = {str(k): v for k, v in metadata_dict.items() if isinstance(v, dict)}
        metadata_kind = "json"
    else:
        df = pd.read_csv(metadata_path)
        image_id_col = _find_first_column(
            df,
            ("imageID", "Image Data ID", "ImageID", "image_id", "imageId"),
            label="imageID",
        )
        age_col = _find_first_column(df, ("age", "Age"), label="age")
        sex_col = _find_first_column(df, ("sex", "Sex"), label="sex")
        image_col = "image" if "image" in df.columns else None
        for _, row in df.iterrows():
            row_by_id.setdefault(str(row[image_id_col]), row)
        metadata_kind = "csv"

    missing_ids = [image_id for image_id in test_ids if image_id not in row_by_id]
    if missing_ids:
        preview = missing_ids[:10]
        raise ValueError(
            f"Found {len(missing_ids)} test IDs missing in condition metadata. First missing IDs: {preview}"
        )

    age_transform = str(getattr(args, "condition_age_transform", "auto"))
    condition_records = []
    for image_id in test_ids:
        row = row_by_id[image_id]
        if metadata_kind == "json":
            raw_age_value = _get_first_present_key(row, ("age", "Age"), "age", image_id=image_id)
            raw_sex_value = _get_first_present_key(row, ("sex", "Sex"), "sex", image_id=image_id)
            source_image = row.get("image", row.get("Image"))
        else:
            raw_age_value = row[age_col]
            raw_sex_value = row[sex_col]
            source_image = row[image_col] if image_col is not None else None
        raw_age = _parse_age_value(raw_age_value, image_id=image_id)
        condition_records.append(
            {
                "imageID": image_id,
                "age": _transform_age_value(raw_age, transform=age_transform),
                "age_raw": raw_age,
                "sex": _parse_sex_value(raw_sex_value, image_id=image_id),
                "source_image": source_image,
            }
        )

    return condition_records, metadata_path, split_path


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def get_data(args):
    if args.dataset_type == "T1All":
        from stai_utils.datasets.dataset_utils import T1All

        dataset = T1All(
            args.img_size,
            args.num_workers,
            age_normalization=args.age_normalization,
            rank=0,
            world_size=1,
            spacing=args.spacing,
            sample_balanced_age_for_training=args.sample_balanced_age_for_training,
        )
        train_loader, val_loader = dataset.get_dataloaders(
            1, debug_one_sample=getattr(args, "debug_one_sample", False)
        )
        return train_loader, val_loader

    if args.dataset_type == "CSV":
        required_keys = ("csv_path", "data_split_json_path")
        missing = [key for key in required_keys if not hasattr(args, key)]
        if missing:
            raise ValueError(f"Missing config keys for CSV dataset_type: {missing}")

        train_transforms = build_train_transforms(
            spacing=args.spacing,
            channel=int(getattr(args, "channel", 0)),
            pad_k=int(getattr(args, "pad_k", 32)),
        )
        train_loader, val_loader, _ = make_dataloaders_from_csv(
            csv_path=args.csv_path,
            data_split_json_path=args.data_split_json_path,
            conditions=getattr(args, "conditions", ("age", "sex", "vol", "group")),
            train_transforms=train_transforms,
            n_samples=parse_n_samples(getattr(args, "n_samples", None)),
            batch_size=1,
            num_workers=args.num_workers,
            data_root=getattr(args, "data_root", None),
        )
        return train_loader, val_loader

    raise ValueError(f"Unknown dataset type: {args.dataset_type}")


def _is_template_registration_autoencoder(autoencoder: torch.nn.Module) -> bool:
    return isinstance(
        autoencoder,
        (AutoencoderKLTemplateRegistration, AutoencoderKLConditionalTemplateRegistration),
    )


def _build_template(autoencoder: torch.nn.Module, metadata: torch.Tensor | None) -> torch.Tensor | None:
    if isinstance(autoencoder, AutoencoderKLConditionalTemplateRegistration):
        if metadata is None:
            raise ValueError(
                "Conditional template autoencoder requires metadata for inference. "
                "Provide --age and --sex, or use data-driven calibration."
            )
        return autoencoder.get_template_image(metadata)

    if isinstance(autoencoder, AutoencoderKLTemplateRegistration):
        return autoencoder.get_template_image()

    return None


def _encode_latent(
    autoencoder: torch.nn.Module,
    images: torch.Tensor,
    template: torch.Tensor | None,
) -> torch.Tensor:
    if _is_template_registration_autoencoder(autoencoder):
        return autoencoder.encode_stage_2_inputs(images, template)
    return autoencoder.encode_stage_2_inputs(images)


def _load_state_dict(ckpt_path: str, map_location="cpu") -> tuple[dict, int | None]:
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"], int(state.get("epoch")) if state.get("epoch") is not None else None
    if isinstance(state, dict):
        return state, None
    raise ValueError(f"Unexpected checkpoint format for {ckpt_path}")


def _build_scheduler(noise_scheduler_cfg: dict) -> DDPMScheduler:
    if noise_scheduler_cfg["schedule"] == "cosine":
        return DDPMScheduler(
            num_train_timesteps=noise_scheduler_cfg["num_train_timesteps"],
            schedule=noise_scheduler_cfg["schedule"],
            clip_sample=noise_scheduler_cfg["clip_sample"],
        )
    return DDPMScheduler(
        num_train_timesteps=noise_scheduler_cfg["num_train_timesteps"],
        schedule=noise_scheduler_cfg["schedule"],
        beta_start=noise_scheduler_cfg["beta_start"],
        beta_end=noise_scheduler_cfg["beta_end"],
        clip_sample=noise_scheduler_cfg["clip_sample"],
    )


def _compute_scale_and_latent_shape(args, autoencoder, device, sample_metadata):
    if args.scale_factor is not None:
        scale_factor = float(args.scale_factor)
    else:
        scale_factor = None

    reference_affine = np.eye(4, dtype=np.float32)

    # Prefer data-driven calibration (same as training), but fall back to a synthetic tensor if unavailable.
    latent_shape = None
    if args.skip_data_calibration:
        data_error = "skipped by --skip-data-calibration"
    else:
        data_error = None
        try:
            train_loader, _ = get_data(args)
            batch = first(train_loader)
            reference_affine = _extract_reference_affine(batch["image"])

            images = batch["image"].float().to(device)
            images = _rescale_to_01(images)

            if args.diffusion_def.get("with_conditioning", False):
                age = batch["age"][None].float().to(device)
                sex = batch["sex"][None].float().to(device)
                metadata = torch.cat([age, sex], dim=-1)
            else:
                metadata = None

            template = _build_template(autoencoder, metadata)
            z = _encode_latent(autoencoder, images, template)
            latent_shape = list(z.shape[2:])
            if scale_factor is None:
                scale_factor = float((1.0 / torch.std(z)).detach().cpu().item())
        except Exception as exc:  # noqa: BLE001
            data_error = str(exc)

    if latent_shape is None:
        img_size = [int(v) for v in args.img_size]
        with torch.no_grad():
            dummy = torch.zeros((1, 1, *img_size), device=device, dtype=torch.float32)
            template = _build_template(autoencoder, sample_metadata)
            z = _encode_latent(autoencoder, dummy, template)
            latent_shape = list(z.shape[2:])
        if scale_factor is None:
            scale_factor = 1.0

    return {
        "scale_factor": float(scale_factor),
        "autoencoder_latent_shape": latent_shape,
        "reference_affine": reference_affine,
        "data_calibration_error": data_error,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MorphLDM diffusion inference")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./environment_config.json",
        help="Environment json with base paths.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config_spacing1p5.json",
        help="Training config json used for this run.",
    )
    parser.add_argument(
        "--diffusion-ckpt",
        default=None,
        help="Path to diffusion UNet checkpoint (.pt). Defaults to <run>/diffuion/diffusion_unet_best.pt.",
    )
    parser.add_argument(
        "--autoencoder-ckpt",
        default=None,
        help="Path to autoencoder checkpoint (.pt). Defaults to autoencoder_ckpt_path or autoencoder_<name>.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save generated NIfTI files. Defaults to <run>/diffuion/inference_<timestamp>/.",
    )
    parser.add_argument(
        "--inference-config",
        default="./inference_config.json",
        help=(
            "Optional JSON file for inference-only args "
            "(e.g., num_samples, age, sex, ckpt paths, device)."
        ),
    )
    parser.add_argument(
        "--condition-source",
        choices=("manual", "test_split"),
        default="manual",
        help=(
            "manual: use --age/--sex for all samples. "
            "test_split: use age/sex from all IDs in split['test'] (1-to-1)."
        ),
    )
    parser.add_argument(
        "--condition-metadata-path",
        default=None,
        help=(
            "Metadata table path for test_split conditioning (.json or .csv). "
            "If unset, falls back to condition_csv_path, then csv_path."
        ),
    )
    parser.add_argument(
        "--condition-age-transform",
        choices=("auto", "none", "divide_by_100"),
        default="auto",
        help=(
            "Transform applied to age values loaded from condition metadata in test_split mode. "
            "'auto' divides by 100 when age looks unnormalized (age>2)."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Sampling steps. Defaults to NoiseScheduler.num_train_timesteps.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="How many samples to generate per forward pass. Lower this to avoid CUDA OOM.",
    )
    parser.add_argument(
        "--max-empty-retries",
        type=int,
        default=8,
        help="Retries with new noise when a generated sample is near-empty (std <= empty_std_threshold).",
    )
    parser.add_argument(
        "--empty-std-threshold",
        type=float,
        default=1e-6,
        help="Std threshold used to treat a generated sample as empty.",
    )
    parser.add_argument(
        "--age",
        type=float,
        default=0.75,
        help="Conditioning age value in the same normalized scale used in training.",
    )
    parser.add_argument(
        "--sex",
        type=float,
        default=0.0,
        help="Conditioning sex value in the same encoding used in training.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Optional override for latent scale factor. If omitted, it is calibrated from the training data.",
    )
    parser.add_argument(
        "--skip-data-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip dataset loading for scale-factor calibration and latent shape checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show diffusion progress bar.",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use AMP during inference. Defaults to config value.",
    )

    # First pass: only to discover file paths provided by CLI.
    bootstrap_args = parser.parse_args()

    env_dict = json.load(open(bootstrap_args.environment_file, "r", encoding="utf-8"))
    config_dict = json.load(open(bootstrap_args.config_file, "r", encoding="utf-8"))

    merged_defaults = {}
    merged_defaults.update(env_dict)
    merged_defaults.update(config_dict)

    inference_config_path = _resolve_path(
        bootstrap_args.inference_config,
        base_dir=os.path.dirname(os.path.abspath(bootstrap_args.config_file)),
    )
    if os.path.exists(inference_config_path):
        with open(inference_config_path, "r", encoding="utf-8") as f:
            inference_dict = json.load(f)
        if not isinstance(inference_dict, dict):
            raise ValueError(
                f"Inference config must be a JSON object, got {type(inference_dict).__name__}"
            )
        merged_defaults.update(inference_dict)

    # Preserve chosen control files and reparse CLI so explicit CLI flags always win.
    merged_defaults.update(
        {
            "environment_file": bootstrap_args.environment_file,
            "config_file": bootstrap_args.config_file,
            "inference_config": bootstrap_args.inference_config,
        }
    )
    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    args.use_amp = bool(getattr(args, "use_amp", False))
    return args


def main():
    args = parse_args()
    set_determinism(seed=int(args.seed))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    args.model_dir = os.path.join(args.base_model_dir, args.run_name)
    args.autoencoder_dir = os.path.join(args.model_dir, "autoencoder")
    args.diffusion_dir = os.path.join(args.model_dir, "diffuion")

    config_dir = os.path.dirname(os.path.abspath(args.config_file))

    if args.autoencoder_ckpt:
        autoencoder_ckpt_path = _resolve_path(args.autoencoder_ckpt, base_dir=config_dir)
    else:
        custom_autoencoder_ckpt = str(getattr(args, "autoencoder_ckpt_path", "")).strip()
        if custom_autoencoder_ckpt:
            autoencoder_ckpt_path = _resolve_path(custom_autoencoder_ckpt, base_dir=config_dir)
        else:
            autoencoder_ckpt_path = os.path.join(
                args.autoencoder_dir,
                f"autoencoder_{args.autoencoder_ckpt_name}.pt",
            )

    if args.diffusion_ckpt:
        diffusion_ckpt_path = _resolve_path(args.diffusion_ckpt, base_dir=config_dir)
    else:
        diffusion_ckpt_path = os.path.join(args.diffusion_dir, "diffusion_unet_best.pt")

    if not os.path.exists(autoencoder_ckpt_path):
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {autoencoder_ckpt_path}")
    if not os.path.exists(diffusion_ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_ckpt_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        output_dir = _resolve_path(args.output_dir)
    else:
        output_dir = os.path.join(args.diffusion_dir, f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    autoencoder_state_dict, _ = _load_state_dict(autoencoder_ckpt_path, map_location="cpu")
    autoencoder_state_dict.pop("template_image", None)
    autoencoder.load_state_dict(autoencoder_state_dict)
    autoencoder.eval()

    unet = define_instance(args, "diffusion_def").to(device)
    unet_state_dict, diffusion_epoch = _load_state_dict(diffusion_ckpt_path, map_location="cpu")
    unet.load_state_dict(unet_state_dict)
    unet.eval()

    condition_records = None
    metadata_path_for_conditions = None
    split_path_for_conditions = None
    with_conditioning = bool(args.diffusion_def.get("with_conditioning", False))
    if with_conditioning:
        if args.condition_source == "test_split":
            condition_records, metadata_path_for_conditions, split_path_for_conditions = _load_test_split_conditions(
                args=args,
                config_dir=config_dir,
            )
            num_samples = len(condition_records)
            age_values = torch.tensor(
                [record["age"] for record in condition_records],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(1)
            sex_values = torch.tensor(
                [record["sex"] for record in condition_records],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(1)
        else:
            num_samples = int(args.num_samples)
            age_values = torch.full(
                (num_samples, 1), float(args.age), device=device, dtype=torch.float32
            )
            sex_values = torch.full(
                (num_samples, 1), float(args.sex), device=device, dtype=torch.float32
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        age = age_values
        sex = sex_values
        condition = torch.cat([age, sex], dim=-1).unsqueeze(1)
        template_metadata = condition[:, 0]
    else:
        num_samples = int(args.num_samples)
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        condition = None
        template_metadata = None

    calibration = _compute_scale_and_latent_shape(
        args=args,
        autoencoder=autoencoder,
        device=device,
        sample_metadata=template_metadata[:1] if template_metadata is not None else None,
    )
    scale_factor = calibration["scale_factor"]
    autoencoder_latent_shape = calibration["autoencoder_latent_shape"]
    reference_affine = calibration["reference_affine"]

    configured_latent_shape = [int(v) for v in getattr(args, "latent_shape", autoencoder_latent_shape)]

    scheduler = _build_scheduler(args.NoiseScheduler)
    if args.num_inference_steps is not None:
        scheduler.set_timesteps(int(args.num_inference_steps))

    if tuple(configured_latent_shape) == tuple(autoencoder_latent_shape):
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)
    else:
        inferer = LatentDiffusionInferer(
            scheduler=scheduler,
            scale_factor=scale_factor,
            ldm_latent_shape=tuple(configured_latent_shape),
            autoencoder_latent_shape=tuple(autoencoder_latent_shape),
        )

    sample_batch_size = int(args.sample_batch_size)
    if sample_batch_size <= 0:
        raise ValueError(f"sample_batch_size must be > 0, got {sample_batch_size}")
    sample_batch_size = min(sample_batch_size, int(num_samples))
    max_empty_retries = int(args.max_empty_retries)
    if max_empty_retries < 0:
        raise ValueError(f"max_empty_retries must be >= 0, got {max_empty_retries}")
    empty_std_threshold = float(args.empty_std_threshold)
    if empty_std_threshold < 0.0:
        raise ValueError(f"empty_std_threshold must be >= 0, got {empty_std_threshold}")

    amp_enabled = bool(args.use_amp and device.type == "cuda")

    sample_manifest = []
    with torch.no_grad():
        for batch_start in range(0, int(num_samples), sample_batch_size):
            batch_end = min(batch_start + sample_batch_size, int(num_samples))
            batch_n = batch_end - batch_start

            noise_shape = [batch_n, int(args.latent_channels)] + configured_latent_shape
            input_noise = torch.randn(noise_shape, device=device, dtype=torch.float32)

            batch_condition = condition[batch_start:batch_end] if condition is not None else None
            batch_template_metadata = (
                template_metadata[batch_start:batch_end]
                if template_metadata is not None
                else None
            )
            batch_template = _build_template(autoencoder, batch_template_metadata)

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                synthetic_images = inferer.sample(
                    input_noise=input_noise,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    scheduler=scheduler,
                    conditioning=batch_condition,
                    template=batch_template,
                    verbose=bool(args.verbose),
                )

            synthetic_images = synthetic_images.detach().float().cpu()
            if synthetic_images.ndim != 5 or synthetic_images.shape[1] < 1:
                raise ValueError(
                    f"Expected output shape [B, C, D, H, W], got {tuple(synthetic_images.shape)}"
                )

            for local_idx in range(synthetic_images.shape[0]):
                sample_idx = batch_start + local_idx
                sample_img = synthetic_images[local_idx : local_idx + 1]
                retries_used = 0
                if max_empty_retries > 0:
                    sample_std = float(sample_img[0, 0].std().item())
                    while sample_std <= empty_std_threshold and retries_used < max_empty_retries:
                        retries_used += 1
                        retry_noise_shape = [1, int(args.latent_channels)] + configured_latent_shape
                        retry_noise = torch.randn(retry_noise_shape, device=device, dtype=torch.float32)
                        retry_condition = (
                            batch_condition[local_idx : local_idx + 1]
                            if batch_condition is not None
                            else None
                        )
                        retry_template_metadata = (
                            batch_template_metadata[local_idx : local_idx + 1]
                            if batch_template_metadata is not None
                            else None
                        )
                        retry_template = _build_template(autoencoder, retry_template_metadata)
                        retry_autocast_ctx = (
                            torch.amp.autocast("cuda", enabled=amp_enabled)
                            if device.type == "cuda"
                            else nullcontext()
                        )
                        with retry_autocast_ctx:
                            retry_sample = inferer.sample(
                                input_noise=retry_noise,
                                autoencoder_model=autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                                conditioning=retry_condition,
                                template=retry_template,
                                verbose=False,
                            )
                        sample_img = retry_sample.detach().float().cpu()
                        sample_std = float(sample_img[0, 0].std().item())

                image_np = sample_img[0, 0].numpy()
                if condition_records is not None:
                    image_id = condition_records[sample_idx]["imageID"]
                    out_name = f"sample_{sample_idx:03d}_{_safe_id_for_filename(image_id)}.nii.gz"
                else:
                    out_name = f"sample_{sample_idx:03d}.nii.gz"
                nib.save(nib.Nifti1Image(image_np, reference_affine), os.path.join(output_dir, out_name))
                row = {
                    "sample_index": int(sample_idx),
                    "output_file": out_name,
                    "retries_used": int(retries_used),
                    "sample_std": float(sample_img[0, 0].std().item()),
                }
                if condition_records is not None:
                    row.update(condition_records[sample_idx])
                sample_manifest.append(row)

    with open(os.path.join(output_dir, "sample_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(sample_manifest, f, indent=2)

    run_summary = {
        "event": "inference_complete",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "inference_config": _resolve_path(
            args.inference_config,
            base_dir=os.path.dirname(os.path.abspath(args.config_file)),
        ),
        "diffusion_ckpt": diffusion_ckpt_path,
        "autoencoder_ckpt": autoencoder_ckpt_path,
        "diffusion_ckpt_epoch": diffusion_epoch,
        "num_samples": int(num_samples),
        "conditioning": {
            "enabled": with_conditioning,
            "source": args.condition_source if with_conditioning else None,
            "age": float(args.age) if (with_conditioning and args.condition_source == "manual") else None,
            "sex": float(args.sex) if (with_conditioning and args.condition_source == "manual") else None,
            "age_transform": args.condition_age_transform if with_conditioning else None,
            "matched_test_count": len(condition_records) if condition_records is not None else None,
            "matched_test_split_json": split_path_for_conditions,
            "matched_test_metadata_path": metadata_path_for_conditions,
        },
        "latent_channels": int(args.latent_channels),
        "ldm_latent_shape": configured_latent_shape,
        "autoencoder_latent_shape": autoencoder_latent_shape,
        "scale_factor": float(scale_factor),
        "num_inference_steps": int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(args.NoiseScheduler["num_train_timesteps"]),
        "sample_batch_size": int(sample_batch_size),
        "max_empty_retries": int(max_empty_retries),
        "empty_std_threshold": float(empty_std_threshold),
        "output_dir": output_dir,
        "sample_manifest": os.path.join(output_dir, "sample_manifest.json"),
        "data_calibration_error": calibration["data_calibration_error"],
    }
    with open(os.path.join(output_dir, "inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    pprint(run_summary)


if __name__ == "__main__":
    main()
