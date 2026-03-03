import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pprint import pprint
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from monai.utils import set_determinism
from monai.bundle import ConfigParser
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator

import morphldm.layers as reg_layers
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
        wandb_module.init(project=args.wandb_project_name, name=run_name, config=vars(args))
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


def save_reconstruction_plot(images, reconstruction, out_path, title):
    if images is None or reconstruction is None:
        return
    if images.ndim != 5 or reconstruction.ndim != 5:
        return

    img = images[0, 0].detach().float().cpu().numpy()
    rec = reconstruction[0, 0].detach().float().cpu().numpy()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    for ax in axs.ravel():
        ax.axis("off")

    vmin, vmax = 0.0, 1.0
    axs[0, 0].imshow(img[..., img.shape[2] // 2], cmap="gray", vmin=vmin, vmax=vmax)
    axs[0, 1].imshow(img[:, img.shape[1] // 2, ...], cmap="gray", vmin=vmin, vmax=vmax)
    axs[0, 2].imshow(img[img.shape[0] // 2, ...], cmap="gray", vmin=vmin, vmax=vmax)

    axs[1, 0].imshow(rec[..., rec.shape[2] // 2], cmap="gray", vmin=vmin, vmax=vmax)
    axs[1, 1].imshow(rec[:, rec.shape[1] // 2, ...], cmap="gray", vmin=vmin, vmax=vmax)
    axs[1, 2].imshow(rec[rec.shape[0] // 2, ...], cmap="gray", vmin=vmin, vmax=vmax)

    axs[0, 1].set_title(f"{title}\nOriginal", fontsize=12)
    axs[1, 1].set_title("Reconstruction", fontsize=12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_autoencoder_resume_ckpt(
    path,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch,
    best_val_recon_epoch_loss,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "state_dict": autoencoder.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict() if optimizer_g is not None else None,
        "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
        "best_val_recon_epoch_loss": float(best_val_recon_epoch_loss),
    }
    torch.save(payload, path)


def _load_autoencoder_resume_ckpt(
    path,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    device,
):
    """
    Returns: (start_epoch, best_val_recon_epoch_loss)
    """
    map_location = device
    if isinstance(device, int):
        map_location = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    is_packaged = (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and (
            "optimizer_g" in ckpt
            or "optimizer_d" in ckpt
            or "discriminator_state_dict" in ckpt
            or "epoch" in ckpt
        )
    )

    if not is_packaged:
        # Backward-compat: raw state_dict checkpoints (weights only).
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = autoencoder.load_state_dict(state, strict=False)
        print(
            "[resume] loaded weights-only autoencoder checkpoint: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

        # Try loading a matching discriminator checkpoint in the same folder.
        inferred_epoch = 0
        base = os.path.basename(path)
        m = re.search(r"autoencoder_(\d+)\.pt$", base)
        if m:
            inferred_epoch = int(m.group(1))

        disc_path = os.path.join(
            os.path.dirname(path),
            base.replace("autoencoder", "discriminator", 1),
        )
        if os.path.exists(disc_path):
            disc_state = torch.load(disc_path, map_location=map_location, weights_only=False)
            disc_state = disc_state.get("state_dict", disc_state) if isinstance(disc_state, dict) else disc_state
            d_missing, d_unexpected = discriminator.load_state_dict(disc_state, strict=False)
            print(
                "[resume] loaded paired discriminator checkpoint: "
                f"path={disc_path} missing={len(d_missing)} unexpected={len(d_unexpected)}"
            )
        else:
            print(f"[resume] paired discriminator checkpoint not found: {disc_path}")

        return inferred_epoch, 100.0

    missing, unexpected = autoencoder.load_state_dict(ckpt["state_dict"], strict=False)
    print(
        "[resume] loaded packaged autoencoder checkpoint: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    if "discriminator_state_dict" in ckpt and ckpt["discriminator_state_dict"] is not None:
        d_missing, d_unexpected = discriminator.load_state_dict(
            ckpt["discriminator_state_dict"], strict=False
        )
        print(
            "[resume] loaded discriminator checkpoint: "
            f"missing={len(d_missing)} unexpected={len(d_unexpected)}"
        )

    if optimizer_g is not None and ckpt.get("optimizer_g") is not None:
        try:
            optimizer_g.load_state_dict(ckpt["optimizer_g"])
        except Exception as exc:
            print(f"[resume] optimizer_g state incompatible, skipping: {exc}")

    if optimizer_d is not None and ckpt.get("optimizer_d") is not None:
        try:
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        except Exception as exc:
            print(f"[resume] optimizer_d state incompatible, skipping: {exc}")

    start_epoch = int(ckpt.get("epoch", 0))
    best_val_recon_epoch_loss = float(ckpt.get("best_val_recon_epoch_loss", 100.0))
    return start_epoch, best_val_recon_epoch_loss


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


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
            args.autoencoder_train["batch_size"],
        )
        print(f"Number of batches in train_loader: {len(train_loader)}")
        print(f"Number of batches in val_loader: {len(val_loader)}")
        return train_loader, val_loader, dataset

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
            batch_size=args.autoencoder_train["batch_size"],
            num_workers=args.num_workers,
            data_root=getattr(args, "data_root", None),
        )
        print(f"Number of batches in train_loader: {len(train_loader)}")
        print(f"Number of batches in val_loader: {len(val_loader)}")
        return train_loader, val_loader, None

    raise ValueError(f"Unknown dataset type: {args.dataset_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="MorphLDM autoencoder training")
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
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "--resume-autoencoder-ckpt",
        default="",
        help=(
            "Optional path to packaged autoencoder checkpoint (autoencoder_last.pt) "
            "to resume full AE training state."
        ),
    )
    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    return args


def aggregate_dicts(dicts):
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return {k: sum(v) / len(v) for k, v in result.items()}


def train_one_epoch(
    train_loader,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    intensity_loss,
    loss_perceptual,
    adv_loss,
    args,
):
    autoencoder.train()
    discriminator.train()

    res = []

    autoencoder_warm_up_n_epochs = args.autoencoder_train["warm_up_n_epochs"]
    adv_weight = args.autoencoder_train["adv_weight"]
    for step, batch in enumerate(train_loader):
        if step == args.train_steps_per_epoch:
            break
        if step % 10 == 0:
            print("step: ", step)

        images = batch["image"].to(args.device).as_tensor()
        images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0) # clip to [0,1] after scaling to avoid potential issues with adversarial loss or perceptual loss
        age = batch["age"].float().to(args.device).view(-1, 1)  # [B,1]
        sex = batch["sex"].float().to(args.device).view(-1, 1)  # [B,1]
        condition = torch.cat([age, sex], dim=1)                # [B,2]

        del batch

        # train Generator part
        optimizer_g.zero_grad(set_to_none=True)
        if args.autoencoder_def["_target_"] in [
            "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
            "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
        ]:
            reconstruction, z_mu, z_sigma, z, displacement_field = autoencoder(
                images, condition
            )
            kl_loss = KL_loss(z_mu, z_sigma)
            recons_loss = intensity_loss(reconstruction.float(), images.float())
            # p_loss = loss_perceptual(reconstruction.float(), images_fixed.float())
            p_loss = torch.tensor(0.0)
            displace_loss = F.mse_loss(
                displacement_field, torch.zeros_like(displacement_field)
            )
            grad_loss = reg_layers.Grad(loss_mult=1.0)(None, displacement_field)
            loss_g = (
                recons_loss
                + args.autoencoder_train["kl_weight"] * kl_loss
                # + perceptual_weight * p_loss
                + args.autoencoder_train["displace_weight"] * displace_loss
                + args.autoencoder_train["grad_weight"] * grad_loss
            )

            train_metrics = {
                "train/recon_loss_iter": recons_loss.item(),
                "train/kl_loss_iter": kl_loss.item(),
                "train/perceptual_loss_iter": p_loss.item(),
                "train/displace_loss_iter": displace_loss.item(),
                "train/grad_loss_iter": grad_loss.item(),
            }
        else:
            reconstruction, z_mu, z_sigma, z = autoencoder(images)
            recons_loss = intensity_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = (
                recons_loss
                + args.autoencoder_train["kl_weight"] * kl_loss
                + args.autoencoder_train["perceptual_weight"] * p_loss
            )

            train_metrics = {
                "train/recon_loss_iter": recons_loss.item(),
                "train/kl_loss_iter": kl_loss.item(),
                "train/perceptual_loss_iter": p_loss.item(),
            }

        if adv_weight > 0 and args.curr_epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            loss_g = loss_g + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # train Discriminator part
        if adv_weight > 0 and args.curr_epoch > autoencoder_warm_up_n_epochs:
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(
                logits_fake, target_is_real=False, for_discriminator=True
            )
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(
                logits_real, target_is_real=True, for_discriminator=True
            )
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

            train_metrics.update(
                {
                    "train/adv_loss_iter": generator_loss.item(),
                    "train/fake_loss_iter": loss_d_fake.item(),
                    "train/real_loss_iter": loss_d_real.item(),
                }
            )

        # Convert metrics to numpy
        train_metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in train_metrics.items()
        }
        res.append(train_metrics)
    return aggregate_dicts(res), images, reconstruction


def eval_one_epoch(val_loader, autoencoder, intensity_loss, loss_perceptual, args):
    autoencoder.eval()

    val_epoch_loss = 0
    val_recon_epoch_loss = 0
    val_grad_epoch_loss = 0
    val_kl_epoch_loss = 0
    for step, batch in enumerate(val_loader):
        images = batch["image"].to(args.device).as_tensor()
        images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0) # clip to [0,1] after scaling to avoid potential issues with adversarial loss or perceptual loss
        age = batch["age"].float().to(args.device).view(-1, 1)  # [B,1]
        sex = batch["sex"].float().to(args.device).view(-1, 1)  # [B,1]
        condition = torch.cat([age, sex], dim=1)                # [B,2]

        #print("images", images.shape, images.min().item(), images.max().item(), images.mean().item())
        #print("condition", condition.shape, condition[0].detach().cpu().numpy())



        if step == args.val_steps_per_epoch:
            break

        with torch.no_grad():
            if args.autoencoder_def["_target_"] in [
                "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
            ]:
                reconstruction, z_mu, z_sigma, z, displacement_field = autoencoder(
                    images, condition
                )
                '''
                template = autoencoder.get_template_image(condition)
                print("template", template.shape, template.min().item(), template.max().item(), template.mean().item())
                print("disp", displacement_field.shape,
                    displacement_field.min().item(), displacement_field.max().item(),
                    displacement_field.abs().mean().item())
                print("recon", reconstruction.shape, reconstruction.min().item(), reconstruction.max().item(), reconstruction.mean().item())
        '''
                kl_loss = KL_loss(z_mu, z_sigma)
                recons_loss = intensity_loss(reconstruction.float(), images.float())
                # p_loss = loss_perceptual(reconstruction.float(), images_fixed.float())
                p_loss = torch.tensor(0.0)
                displace_loss = F.mse_loss(
                    displacement_field, torch.zeros_like(displacement_field)
                )
                grad_loss = reg_layers.Grad(loss_mult=1.0)(None, displacement_field)
                loss_g = (
                    recons_loss
                    + args.autoencoder_train["kl_weight"] * kl_loss
                    + args.autoencoder_train["perceptual_weight"] * p_loss
                    + args.autoencoder_train["displace_weight"] * displace_loss
                    + args.autoencoder_train["grad_weight"] * grad_loss
                )
            else:
                reconstruction, z_mu, z_sigma, z = autoencoder(images)
                recons_loss = intensity_loss(reconstruction, images)
                kl_loss = KL_loss(z_mu, z_sigma)
                p_loss = loss_perceptual(reconstruction.float(), images.float())
                grad_loss = torch.tensor(0.0).to(args.device)
                loss_g = (
                    recons_loss
                    + args.autoencoder_train["kl_weight"] * kl_loss
                    + args.autoencoder_train["perceptual_weight"] * p_loss
                )

        val_epoch_loss += loss_g.item()
        val_recon_epoch_loss += recons_loss.item()
        val_grad_epoch_loss += grad_loss.item()
        val_kl_epoch_loss += kl_loss.item()

    val_epoch_loss = val_epoch_loss / (step + 1)
    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
    val_grad_epoch_loss = val_grad_epoch_loss / (step + 1)
    val_kl_epoch_loss = val_kl_epoch_loss / (step + 1)
    return (
        val_epoch_loss,
        val_recon_epoch_loss,
        val_grad_epoch_loss,
        val_kl_epoch_loss,
        images,
        reconstruction,
    )


def main():
    args = parse_args()
    pprint(vars(args))

    args.device = 0
    torch.cuda.set_device(args.device)
    print(f"Using device {args.device}")

    set_determinism(42)

    # Data
    train_loader, val_loader, dataset = get_data(args)
    check_batch = next(iter(train_loader))
    actual_shape = tuple(int(v) for v in check_batch["image"].shape[2:])
    expected_shape = tuple(int(v) for v in args.autoencoder_def["template_shape"][1:])
    if actual_shape != expected_shape:
        raise ValueError(
            f"Data shape {actual_shape} != template_shape {expected_shape}. "
            "Adjust config spacing/template_shape for consistency."
        )

    # Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(args.device)
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(args.device)

    args.model_dir = os.path.join(args.base_model_dir, args.run_name)
    args.autoencoder_dir = os.path.join(args.model_dir, "autoencoder")
    args.diffusion_dir = os.path.join(args.model_dir, "diffuion")

    # Ensure directories exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.autoencoder_dir, exist_ok=True)
    os.makedirs(args.diffusion_dir, exist_ok=True)
    logger = init_logger(
        args=args,
        run_name=args.run_name,
        log_dir=args.autoencoder_dir,
        metrics_filename="metrics_autoencoder.jsonl",
    )
    trained_g_path_best = os.path.join(args.autoencoder_dir, "autoencoder_best.pt")
    trained_d_path_best = os.path.join(args.autoencoder_dir, "discriminator_best.pt")
    trained_resume_last_path = os.path.join(args.autoencoder_dir, "autoencoder_last.pt")

    # Losses
    if (
        "recon_loss" in args.autoencoder_train
        and args.autoencoder_train["recon_loss"] == "l2"
    ):
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(
        spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
    )
    loss_perceptual.to(args.device)
    adv_weight = args.autoencoder_train["adv_weight"]

    # Optimizers
    optimizer_g = torch.optim.Adam(
        params=autoencoder.parameters(), lr=args.autoencoder_train["lr"]
    )
    optimizer_d = None
    if adv_weight > 0:
        optimizer_d = torch.optim.Adam(
            params=discriminator.parameters(), lr=args.autoencoder_train["lr"]
        )

    # Training
    n_epochs = args.autoencoder_train["n_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    save_recon_every = int(args.autoencoder_train.get("save_recon_every", val_interval))
    save_ckpt_every = int(args.autoencoder_train.get("save_ckpt_every", val_interval))
    recon_dir = os.path.join(args.autoencoder_dir, "reconstructions")
    best_val_recon_epoch_loss = 100.0
    start_epoch = 0

    resume_ckpt = str(getattr(args, "resume_autoencoder_ckpt", "") or "").strip()
    if resume_ckpt:
        if os.path.isdir(resume_ckpt):
            resume_ckpt = os.path.join(resume_ckpt, "autoencoder_last.pt")
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        start_epoch, best_val_recon_epoch_loss = _load_autoencoder_resume_ckpt(
            resume_ckpt,
            autoencoder=autoencoder,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=args.device,
        )
        print(
            "[resume] "
            f"start_epoch={start_epoch}, "
            f"best_val_recon_epoch_loss={best_val_recon_epoch_loss:.6f}"
        )

    if start_epoch >= n_epochs:
        print(
            f"[resume] start_epoch ({start_epoch}) >= n_epochs ({n_epochs}), "
            "nothing to train."
        )
        close_logger(logger)
        return

    for epoch in range(start_epoch + 1, n_epochs + 1):
        args.curr_epoch = epoch
        print("Epoch:", epoch)
        train_metrics, images, reconstruction = train_one_epoch(
            train_loader,
            autoencoder,
            discriminator,
            optimizer_g,
            optimizer_d,
            intensity_loss,
            loss_perceptual,
            adv_loss,
            args,
        )
        log_metrics(
            logger,
            step=epoch,
            metrics=train_metrics,
            phase="train",
            epoch=epoch,
        )
        if epoch % save_recon_every == 0:
            save_reconstruction_plot(
                images=images,
                reconstruction=reconstruction,
                out_path=os.path.join(recon_dir, f"train_epoch_{epoch:04d}.png"),
                title=f"Epoch {epoch} Train",
            )

        # validation
        if epoch % val_interval == 0:
            (
                val_epoch_loss,
                val_recon_epoch_loss,
                val_grad_epoch_loss,
                val_kl_epoch_loss,
                images,
                reconstruction,
            ) = eval_one_epoch(
                val_loader, autoencoder, intensity_loss, loss_perceptual, args
            )

            # save last model
            print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")

            trained_g_path_epoch = os.path.join(
                args.autoencoder_dir, f"autoencoder_{epoch}.pt"
            )
            trained_d_path_epoch = os.path.join(
                args.autoencoder_dir, f"discriminator_{epoch}.pt"
            )

            if epoch % save_ckpt_every == 0:
                torch.save(autoencoder.state_dict(), trained_g_path_epoch)
                torch.save(discriminator.state_dict(), trained_d_path_epoch)
            # save best model
            if val_recon_epoch_loss < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_recon_epoch_loss
                torch.save(autoencoder.state_dict(), trained_g_path_best)
                torch.save(discriminator.state_dict(), trained_d_path_best)
                print("Got best val recon loss.")
                print("Save trained autoencoder to", trained_g_path_best)
                print("Save trained discriminator to", trained_d_path_best)

            # write val loss for each epoch into wandb
            val_metrics = {
                "val/loss": val_epoch_loss,
                "val/recon_loss": val_recon_epoch_loss,
                "val/grad_loss": val_grad_epoch_loss,
                "val/kl_loss": val_kl_epoch_loss,
            }
            log_metrics(
                logger,
                step=epoch,
                metrics=val_metrics,
                phase="val",
                epoch=epoch,
            )
            if epoch % save_recon_every == 0:
                save_reconstruction_plot(
                    images=images,
                    reconstruction=reconstruction,
                    out_path=os.path.join(recon_dir, f"val_epoch_{epoch:04d}.png"),
                    title=f"Epoch {epoch} Val",
                )
        _save_autoencoder_resume_ckpt(
            trained_resume_last_path,
            autoencoder=autoencoder,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            epoch=epoch,
            best_val_recon_epoch_loss=best_val_recon_epoch_loss,
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
