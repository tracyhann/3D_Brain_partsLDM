import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pprint import pprint
import torch
import torch.nn.functional as F
from generative.networks.schedulers import DDPMScheduler
from monai.utils import first, set_determinism
from monai.bundle import ConfigParser
from torch.amp import GradScaler
import numpy as np
import nibabel as nib

from morphldm.inferer import LatentDiffusionInferer
from morphldm.data_utils import (
    build_train_transforms,
    make_dataloaders_from_csv,
    parse_n_samples,
)


def _to_float_dict(metrics):
    return {k: float(torch.as_tensor(v).detach().cpu().item()) for k, v in metrics.items()}


def _to_jsonable_dict(data):
    try:
        return json.loads(json.dumps(data))
    except TypeError:
        out = {}
        for key, value in data.items():
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = str(value)
        return out


def _rescale_to_01(images: torch.Tensor) -> torch.Tensor:
    # Match AE training preprocessing: assume source intensities are in [-1, 1].
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def init_logger(args, run_name, log_dir, metrics_filename):
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, metrics_filename)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    use_wandb = bool(getattr(args, "use_wandb", False))
    wandb_module = None
    if use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "use_wandb is true but wandb is not installed. "
                "Set use_wandb=false to keep local logging only."
            ) from exc
        wandb_module = wandb
        wandb_module.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
        )
        wandb_module.save(__file__)
        print("Logging backend: wandb + local jsonl")
    else:
        print("Logging backend: local jsonl only (wandb disabled)")
    with open(metrics_path, "w", encoding="utf-8") as f:
        header = {
            "event": "run_start",
            "run_id": run_id,
            "run_name": run_name,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "config": _to_jsonable_dict(vars(args)),
        }
        f.write(json.dumps(header) + "\n")
    print(f"Local metrics log: {metrics_path}")
    return {
        "use_wandb": use_wandb,
        "wandb": wandb_module,
        "metrics_path": metrics_path,
        "run_id": run_id,
        "run_name": run_name,
        "record_index": 0,
    }


def log_metrics(logger, step, metrics, phase=None, epoch=None):
    metrics = _to_float_dict(metrics)
    logger["record_index"] += 1
    payload = {
        "event": "metric",
        "run_id": logger["run_id"],
        "record_index": int(logger["record_index"]),
        "step": int(step),
    }
    if phase is not None:
        payload["phase"] = str(phase)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    payload.update(metrics)
    with open(logger["metrics_path"], "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    if logger["use_wandb"]:
        logger["wandb"].log(metrics, step=step)


def log_slice_image(logger, epoch, axis, image, out_dir, global_step=None):
    os.makedirs(out_dir, exist_ok=True)
    step_value = int(global_step) if global_step is not None else int(epoch)
    save_path = os.path.join(
        out_dir,
        f"epoch_{int(epoch):04d}_step_{step_value:06d}_axis_{int(axis)}.npy",
    )
    np.save(save_path, image)
    if logger["use_wandb"]:
        logger["wandb"].log(
            {f"val/image/syn_axis_{axis}": logger["wandb"].Image(image)},
            step=step_value,
        )


def extract_reference_affine(image_tensor) -> np.ndarray:
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


def save_sample_nifti(
    synthetic_images: torch.Tensor,
    out_dir: str,
    epoch: int,
    global_step: int,
    reference_affine: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    vols = synthetic_images.detach().float().cpu()
    if vols.ndim != 5 or vols.shape[1] < 1:
        raise ValueError(f"Expected synthetic_images shape [B,C,D,H,W], got {tuple(vols.shape)}")
    for idx in range(vols.shape[0]):
        arr = vols[idx, 0].numpy()
        img = nib.Nifti1Image(arr, reference_affine)
        out_name = f"epoch_{int(epoch):04d}_step_{int(global_step):06d}_sample_{int(idx):02d}.nii.gz"
        nib.save(img, os.path.join(out_dir, out_name))


def close_logger(logger):
    with open(logger["metrics_path"], "a", encoding="utf-8") as f:
        footer = {
            "event": "run_end",
            "run_id": logger["run_id"],
            "run_name": logger["run_name"],
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "records": int(logger["record_index"]),
        }
        f.write(json.dumps(footer) + "\n")
    if logger["use_wandb"]:
        logger["wandb"].finish()


def visualize_one_slice_in_3d_image(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """
    image = image.cpu().detach().numpy()
    # draw image
    center = image.shape[axis] // 2
    if axis == 0:
        draw_img = image[center, :, :]
    elif axis == 1:
        draw_img = image[:, center, :]
    elif axis == 2:
        draw_img = image[:, :, center]
    else:
        raise ValueError("axis should be in [0,1,2]")
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img

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
            args.diffusion_train["batch_size"], debug_one_sample=args.debug_one_sample
        )
        return train_loader, val_loader

    if args.dataset_type == "CSV":
        required_keys = ("csv_path", "data_split_json_path")
        missing = [key for key in required_keys if not hasattr(args, key)]
        if missing:
            raise ValueError(
                f"Missing config keys for CSV dataset_type: {missing}"
            )
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
            batch_size=args.diffusion_train["batch_size"],
            num_workers=args.num_workers,
            data_root=getattr(args, "data_root", None),
        )
        return train_loader, val_loader

    raise ValueError(f"Unknown dataset type: {args.dataset_type}")


def train_one_epoch(train_loader, unet, autoencoder, inferer, optimizer, noise_shape, args, scaler=None):
    unet.train()

    train_recon_epoch_loss = 0
    steps_run = 0
    for step, batch in enumerate(train_loader):
        if step == args.train_steps_per_epoch:
            break
        steps_run += 1
        if step % 10 == 0:
            print("Step:", step)

        images = batch["image"].to(args.device)
        images = _rescale_to_01(images)
        if args.diffusion_def["with_conditioning"]:
            age = batch["age"][None].float().to(args.device)
            sex = batch["sex"][None].float().to(args.device)
            condition = torch.cat([age, sex], dim=-1).unsqueeze(1)  # for seq_len
        else:
            condition = None

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with torch.amp.autocast("cuda", enabled=args.use_amp):
            # Generate random noise
            noise = torch.randn(noise_shape, dtype=images.dtype).to(args.device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            with torch.no_grad():
                if args.autoencoder_def["_target_"] in [
                    "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                    "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
                ]:
                    template = autoencoder.get_template_image(condition).detach()
                else:
                    template = None

            noise_pred = inferer(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps,
                condition=condition,
                template=template,
                plot_img=args.debug_mode and step == 0,
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())
            train_recon_epoch_loss += loss.item()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    if steps_run == 0:
        raise ValueError("No training steps were run in this epoch.")
    train_recon_epoch_loss = train_recon_epoch_loss / steps_run
    return train_recon_epoch_loss, steps_run


def eval_one_epoch(val_loader, unet, autoencoder, inferer, noise_shape, args):
    autoencoder.eval()
    unet.eval()
    val_recon_epoch_loss = 0
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=args.use_amp):
            for step, batch in enumerate(val_loader):
                if step == args.val_steps_per_epoch:
                    break
                images = batch["image"].to(args.device)
                images = _rescale_to_01(images)

                if args.diffusion_def["with_conditioning"]:
                    age = batch["age"][None].float().to(args.device)
                    sex = batch["sex"][None].float().to(args.device)
                    condition = torch.cat([age, sex], dim=-1).unsqueeze(1)  # for seq_len
                else:
                    condition = None

                noise = torch.randn(noise_shape, dtype=images.dtype).to(args.device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                if args.autoencoder_def["_target_"] in [
                    "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                    "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
                ]:
                    template = autoencoder.get_template_image(condition)
                else:
                    template = None

                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=condition,
                    template=template,
                )

                val_loss = F.mse_loss(noise_pred.float(), noise.float())
                val_recon_epoch_loss += val_loss.item()
    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
    return val_recon_epoch_loss


def synthesize_example_image(
    unet,
    autoencoder,
    inferer,
    scheduler,
    noise_shape,
    args,
    sample_batch=None,
):
    # Generate random noise
    noise = torch.randn(noise_shape).to(args.device)

    if args.diffusion_def.get("with_conditioning", False):
        if sample_batch is not None and ("age" in sample_batch) and ("sex" in sample_batch):
            age = torch.as_tensor(sample_batch["age"]).float().view(-1, 1).to(args.device)[:1]
            sex = torch.as_tensor(sample_batch["sex"]).float().view(-1, 1).to(args.device)[:1]
        else:
            # Fallback in normalized metadata space used by this project.
            age = torch.tensor([[0.75]], device=args.device).float()
            sex = torch.tensor([[0.0]], device=args.device).float()
        condition = torch.cat([age, sex], dim=-1).unsqueeze(1)  # [B,1,2]
    else:
        condition = None
    if args.autoencoder_def["_target_"] in [
        "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
        "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
    ]:
        template = autoencoder.get_template_image(condition[:, 0]) if condition is not None else None
    else:
        template = None
    synthetic_images = inferer.sample(
        input_noise=noise[0:1, ...],
        autoencoder_model=autoencoder,
        diffusion_model=unet,
        scheduler=scheduler,
        conditioning=condition,
        template=template,
    )
    return synthetic_images


def recon_example_image(x, autoencoder, template=None):
    if template is not None:
        z = autoencoder.encode_stage_2_inputs(x, template)
        return autoencoder.decode_stage_2_outputs(z, template)
    else:
        z = autoencoder.encode_stage_2_inputs(x)
        return autoencoder.decode_stage_2_outputs(z)


def parse_args():
    parser = argparse.ArgumentParser(description="MorphLDM Diffusion Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("--debug-mode", action=argparse.BooleanOptionalAction)
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    return args


def main():
    args = parse_args()
    pprint(vars(args))

    args.device = 0
    torch.cuda.set_device(args.device)
    print(f"Using {args.device}")

    set_determinism(42)

    args.model_dir = os.path.join(args.base_model_dir, args.run_name)
    args.autoencoder_dir = os.path.join(args.model_dir, "autoencoder")
    args.diffusion_dir = os.path.join(args.model_dir, "diffuion")
    logger = init_logger(
        args=args,
        run_name=args.run_name.replace("__auto__", "__diff__"),
        log_dir=args.diffusion_dir,
        metrics_filename="metrics_diffusion.jsonl",
    )
    os.makedirs(args.diffusion_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_data(args)

    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(args.device)
    custom_autoencoder_ckpt = str(getattr(args, "autoencoder_ckpt_path", "")).strip()
    if custom_autoencoder_ckpt:
        autoencoder_path = os.path.expanduser(custom_autoencoder_ckpt)
        if not os.path.isabs(autoencoder_path):
            autoencoder_path = os.path.abspath(autoencoder_path)
    else:
        autoencoder_path = os.path.join(
            args.autoencoder_dir,
            f"autoencoder_{args.autoencoder_ckpt_name}.pt",
        )
    if not os.path.exists(autoencoder_path):
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {autoencoder_path}")
    autoencoder_state_dict = torch.load(autoencoder_path, map_location="cpu")
    autoencoder_state_dict.pop("template_image", None)
    autoencoder.load_state_dict(autoencoder_state_dict)
    print(f"Load trained autoencoder from {autoencoder_path}")

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=args.use_amp):
            check_data = first(train_loader)
            reference_affine = extract_reference_affine(check_data["image"])
            check_image = check_data["image"].float().to(args.device)
            check_image = _rescale_to_01(check_image)
            check_age = check_data["age"][None].float().to(args.device)
            check_sex = check_data["sex"][None].float().to(args.device)
            if args.autoencoder_def["_target_"] in [
                "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
            ]:
                check_metadata = torch.cat([check_age, check_sex], dim=-1)
                check_template = autoencoder.get_template_image(check_metadata)
                z = autoencoder.encode_stage_2_inputs(check_image, check_template)
                recon_images = recon_example_image(check_image, autoencoder, check_template)
            else:
                z = autoencoder.encode_stage_2_inputs(check_image)
                recon_images = recon_example_image(check_image, autoencoder)
            print(f"Latent feature shape {z.shape}")

    scale_factor = 1 / torch.std(z)
    print(f"scale_factor: {scale_factor}")
    if z.shape[1] != args.latent_channels:
        raise ValueError(
            f"Latent channels mismatch: config latent_channels={args.latent_channels}, "
            f"autoencoder latent channels={z.shape[1]}"
        )

    autoencoder_latent_shape = list(z.shape[2:])
    configured_latent_shape = getattr(args, "latent_shape", None)
    if configured_latent_shape is None:
        ldm_latent_shape = autoencoder_latent_shape
    else:
        ldm_latent_shape = [int(v) for v in configured_latent_shape]
    if len(ldm_latent_shape) != len(autoencoder_latent_shape):
        raise ValueError(
            f"Configured latent_shape={ldm_latent_shape} does not match latent rank {len(autoencoder_latent_shape)}"
        )
    noise_shape = [check_data["image"].shape[0], args.latent_channels] + ldm_latent_shape
    print(f"Autoencoder latent shape: {autoencoder_latent_shape}")
    print(f"Diffusion latent shape: {ldm_latent_shape}")

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_def").to(args.device)

    trained_diffusion_path_best = os.path.join(args.diffusion_dir, "diffusion_unet_best.pt")

    if args.NoiseScheduler["schedule"] == "cosine":
        scheduler = DDPMScheduler(
            num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
            schedule=args.NoiseScheduler["schedule"],
            clip_sample=args.NoiseScheduler["clip_sample"],
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
            schedule=args.NoiseScheduler["schedule"],
            beta_start=args.NoiseScheduler["beta_start"],
            beta_end=args.NoiseScheduler["beta_end"],
            clip_sample=args.NoiseScheduler["clip_sample"],
        )

    # We define the inferer using the scale factor:
    if tuple(ldm_latent_shape) == tuple(autoencoder_latent_shape):
        inferer = LatentDiffusionInferer(
            scheduler,
            scale_factor=scale_factor,
        )
    else:
        inferer = LatentDiffusionInferer(
            scheduler,
            scale_factor=scale_factor,
            ldm_latent_shape=tuple(ldm_latent_shape),
            autoencoder_latent_shape=tuple(autoencoder_latent_shape),
        )

    # Step 3: training config
    optimizer_diff = torch.optim.AdamW(
        unet.parameters(),
        lr=args.diffusion_train["lr"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_diff, milestones=args.diffusion_train["lr_scheduler_milestones"], gamma=0.1
    )

    n_epochs = args.diffusion_train["n_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    sample_every_steps = int(args.diffusion_train.get("sample_every_steps", 0))
    save_sample_slices = args.diffusion_train.get("save_sample_slices", False)
    if isinstance(save_sample_slices, str):
        save_sample_slices = save_sample_slices.strip().lower() in ("1", "true", "t", "yes", "y")
    save_sample_nifti_flag = args.diffusion_train.get("save_sample_nifti", True)
    if isinstance(save_sample_nifti_flag, str):
        save_sample_nifti_flag = save_sample_nifti_flag.strip().lower() in ("1", "true", "t", "yes", "y")
    save_every_epoch = args.diffusion_train.get("save_every_epoch", True)
    if isinstance(save_every_epoch, str):
        save_every_epoch = save_every_epoch.strip().lower() in ("1", "true", "t", "yes", "y")
    save_ckpt_every = int(args.diffusion_train.get("save_ckpt_every", 25))
    if save_ckpt_every <= 0:
        save_ckpt_every = 1
    autoencoder.eval()
    scaler = GradScaler("cuda")
    best_val_recon_epoch_loss = 100.0
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        print("Epoch: ", epoch)
        train_epoch_loss, steps_this_epoch = train_one_epoch(
            train_loader,
            unet,
            autoencoder,
            inferer,
            optimizer_diff,
            noise_shape,
            args,
            scaler=scaler,
        )
        prev_global_step = global_step
        global_step += steps_this_epoch

        # write to wandb
        log_metrics(
            logger,
            step=epoch,
            metrics={
                "train/diffusion_loss": train_epoch_loss,
                "train/lr": optimizer_diff.param_groups[0]["lr"],
            },
            phase="train",
            epoch=epoch,
        )
        lr_scheduler.step()

        if epoch % val_interval == 0:
            val_epoch_loss = eval_one_epoch(
                val_loader,
                unet,
                autoencoder,
                inferer,
                noise_shape,
                args,
            )

            # write to local log and optional wandb
            trained_diffusion_path_epoch = os.path.join(args.diffusion_dir, f"diffusion_unet_{epoch}.pt")
            log_metrics(
                logger,
                step=epoch,
                metrics={
                    "val/diffusion_loss": val_epoch_loss,
                },
                phase="val",
                epoch=epoch,
            )
            print(f"Epoch {epoch} val_diffusion_loss: {val_epoch_loss}")

            # save last model
            ckpt_dict = {
                "state_dict": unet.state_dict(),
                "epoch": epoch,
            }
            should_save_epoch_ckpt = bool(save_every_epoch) or (epoch % save_ckpt_every == 0) or (epoch == n_epochs)
            if should_save_epoch_ckpt:
                torch.save(ckpt_dict, trained_diffusion_path_epoch)

            # save best model
            if val_epoch_loss < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_epoch_loss
                torch.save(ckpt_dict, trained_diffusion_path_best)
                print("Got best val noise pred loss.")
                print("Save trained latent diffusion model to", trained_diffusion_path_best)

            # visualize synthesized image
            if sample_every_steps > 0:
                should_sample = (global_step // sample_every_steps) > (
                    prev_global_step // sample_every_steps
                )
            else:
                # Backward-compatible behavior: sample on every validation epoch.
                should_sample = True

            if should_sample and (bool(save_sample_slices) or bool(save_sample_nifti_flag)):
                sample_batch = first(val_loader)
                synthetic_images = synthesize_example_image(
                    unet,
                    autoencoder,
                    inferer,
                    scheduler,
                    noise_shape,
                    args,
                    sample_batch=sample_batch,
                )

                if bool(save_sample_slices):
                    for axis in range(3):
                        synthetic_img = visualize_one_slice_in_3d_image(
                            synthetic_images[0, 0, ...], axis
                        )
                        log_slice_image(
                            logger=logger,
                            epoch=epoch,
                            axis=axis,
                            image=synthetic_img,
                            out_dir=os.path.join(args.diffusion_dir, "sample_slices"),
                            global_step=global_step,
                        )
                if bool(save_sample_nifti_flag):
                    save_sample_nifti(
                        synthetic_images=synthetic_images,
                        out_dir=os.path.join(args.diffusion_dir, "sample_volumes"),
                        epoch=epoch,
                        global_step=global_step,
                        reference_affine=reference_affine,
                    )
    close_logger(logger)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
