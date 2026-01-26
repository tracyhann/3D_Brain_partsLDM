import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn import L1Loss
from tqdm import tqdm
import copy

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandSpatialCropd, RandFlipd, EnsureTyped, CropForegroundd, SpatialPadd
)
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader

from eval_utils import psnr, ssim_3d, fid_from_features, plot_recon_loss, plot_adversarial_loss, plot_reconstructions, sample_ldm, plot_unet_loss

from glob import glob
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import nibabel as nib
import numpy as np
import random
import argparse

# Avoid shared-memory exhaustion in DataLoader workers (e.g., limited /dev/shm)
mp.set_sharing_strategy("file_system")

def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # determinism (good for debugging; may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(1017)


def make_dataloaders_from_csv(csv_path, conditions = ['age', 'sex', 'vol'], train_transforms = None,
                              n_samples = None, train_val_split = 0.1, batch_size = 1, num_workers=8, seed = 1017):
    #Make a list of dicts. Keys must match your transforms.
    #images = sorted(glob("data/ADNI_turboprepout_whole_brain/*.nii.gz")) 
    #labels = sorted(glob("/data/ct/labelsTr/*.nii.gz"))
    #conds = pd.read_csv('data/whole_brain_data.csv').to_dict()

    df = pd.read_csv(csv_path)
    data = []
    for i, row in df.iterrows():
        if i == n_samples:
            break
        sample = {}
        sample['image'] = row['image']
        for c in conditions:
            sample[c] = row[c]
        data.append(sample)
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data)*train_val_split)
    train_data, val_data = data[:-split], data[-split:]
    train_ds = Dataset(data=train_data, transform=train_transforms)
    print(f'Transformed data shape: {train_ds[0]["image"].shape}')
    print(f"Number of training samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ds = Dataset(data=val_data, transform=train_transforms)
    print(f"Number of validation samples: {len(val_ds)}")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader



def check_train_data(train_loader, idx = 0, title = 'Train data example', save_path = None):
    # Plot axial, coronal and sagittal slices of a training sample
    check_data = first(train_loader)

    img = check_data["image"][idx, 0]
    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
    plt.title(title)
    if not save_path == None:
        plt.savefig(save_path)
    plt.close('all')





def _load_ckpt_into_ae(ae, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    missing, unexpected = ae.load_state_dict(state, strict=False)


def _save_ckpt(path: str, unet: torch.nn.Module, opt, scaler, epoch: int, global_step: int,
               ema: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": unet.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    if ema is not None:
        payload["ema"] = {k: v.cpu() for k, v in ema.shadow.items()}
    torch.save(payload, path)


def _load_ckpt_into_unet(unet: torch.nn.Module, ckpt_path: str, device) -> Tuple[int, int]:
    """Loads either a raw state_dict or a packaged checkpoint. Returns (epoch, global_step) if present."""
    sd = torch.load(ckpt_path, map_location=device)
    if "state_dict" in sd:
        missing, unexpected = unet.load_state_dict(sd["state_dict"], strict=False)
        print(f"[LDM] resumed UNet (packaged): missing={len(missing)} unexpected={len(unexpected)}")
        return int(sd.get("epoch", 0)), 0 #int(sd.get("global_step", 0))
    else:
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        print(f"[LDM] resumed UNet (raw): missing={len(missing)} unexpected={len(unexpected)}")
        return 0, 0


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def train_ae(autoencoder, train_loader, val_loader = None, val_interval = 1, ae_epochs = 100, 
             adv_weight = 0.01, perceptual_weight = 0.001, kl_weight = 1e-6, 
             lr=1e-4, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), outdir = 'ckpts'):

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)


    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)



    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), lr=lr)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=lr)



    n_epochs = ae_epochs
    autoencoder_warm_up_n_epochs = int(0.2*n_epochs) # default: 20% of training epochs
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_ssim_train_list, epoch_psnr_train_list = [], []
    val_recon_epoch_loss_list = []
    epoch_ssim_val_list, epoch_psnr_val_list = [], []
    intermediary_images = []
    n_example_images = 4

    ae_best = None
    ae_best_loss = float('inf')

    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        ssim_train, psnr_train = 0, 0
        ssim_val, psnr_val = 0, 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)  # choose only one of Brats channels

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            #reconstruction = torch.clamp(reconstruction, -1.0, 1.0)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss
            psnr_train += psnr(reconstruction.float(), images.float())
            ssim_train += ssim_3d(reconstruction.float(), images.float())

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                    "SSIM_train": ssim_train / (step + 1),
                    "PSNR_train": psnr_train / (step + 1),

                }
            )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
        epoch_ssim_train_list.append(np.mean(ssim_train / (step + 1)))
        epoch_psnr_train_list.append(np.mean(psnr_train / (step + 1)))

        #print(epoch_gen_loss_list)

        plot_recon_loss(epoch_recon_loss_list, epoch_ssim_train_list, epoch_psnr_train_list, title = f'Train Reconstruction Loss Curve_ep{epoch+1}', outdir = outdir, filename = 'AE_train_recon_loss.png')
        plot_adversarial_loss(epoch_gen_loss_list, epoch_disc_loss_list, title = f'Adversarial Training Curves_ep{epoch+1}', outdir = outdir, filename = 'AE_disc_loss.png')
        plot_reconstructions(batch, reconstruction, idx = 0, channel=0, title = f'Train Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_train_recons.png')

        if val_loader is not None and (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            val_epoch_loss = 0
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=70)
            val_progress_bar.set_description(f"Val Epoch {epoch}")
            with torch.no_grad():
                for val_step, val_batch in val_progress_bar:
                    val_images = val_batch["image"].to(device)
                    val_reconstruction, _, _ = autoencoder(val_images)
                    val_recons_loss = l1_loss(val_reconstruction.float(), val_images.float())
                    val_epoch_loss += val_recons_loss.item()
                    psnr_val += psnr(val_reconstruction.float(), val_images.float())
                    ssim_val += ssim_3d(val_reconstruction.float(), val_images.float())
                    val_progress_bar.set_postfix(
                        {
                            "val_recons_loss": val_epoch_loss / (val_step + 1),
                            "SSIM_val": ssim_val / (val_step + 1),
                            "PSNR_val": psnr_val / (val_step + 1),
                        }
                    )
            val_recon_epoch_loss_list.append(val_epoch_loss / (val_step + 1))
            epoch_ssim_val_list.append(ssim_val / (val_step + 1))
            epoch_psnr_val_list.append(np.mean(psnr_val / (val_step + 1)))
            plot_recon_loss(val_recon_epoch_loss_list, epoch_ssim_val_list, epoch_psnr_val_list, title = f'Val Reconstruction Loss Curve_ep{epoch+1}', outdir = outdir, filename = 'AE_val_recon_loss.png')

            if val_epoch_loss / (val_step + 1) < ae_best_loss:
                print('Updating best AE checkpoint...')
                ae_best_loss = val_epoch_loss / (val_step + 1)
                ae_best = copy.deepcopy(autoencoder)
                _save_ckpt(os.path.join(outdir, "AE_best.pt"), autoencoder, opt=optimizer_g, scaler=None, epoch=epoch+1, global_step=None, 
                            extra={"opt_d": optimizer_d, "train_rec_loss": epoch_recon_loss_list, "train_disc_loss": epoch_disc_loss_list, 
                                   "train_gen_loss": epoch_gen_loss_list, "train_ssim": epoch_ssim_train_list, "train_psnr": epoch_psnr_train_list,
                                   "val_rec_loss": val_recon_epoch_loss_list, "val_ssim": epoch_ssim_val_list, "val_psnr": epoch_psnr_val_list})
                plot_reconstructions(val_batch, val_reconstruction, idx = 0, channel=0, title = f'Val Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_val_recons.png')
            if epoch + 1 == n_epochs:
                print('Final epoch, saving last AE checkpoint and val reconstructions...')
                plot_reconstructions(val_batch, val_reconstruction, idx = 0, channel=0, title = f'Val Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_last_val_recons.png')
        _save_ckpt(os.path.join(outdir, "AE_last.pt"), autoencoder, opt=optimizer_g, scaler=None, epoch=epoch+1, global_step=None, 
                        extra={"opt_d": optimizer_d, "train_rec_loss": epoch_recon_loss_list, "train_disc_loss": epoch_disc_loss_list, 
                               "train_gen_loss": epoch_gen_loss_list, "train_ssim": epoch_ssim_train_list, "train_psnr": epoch_psnr_train_list,
                               "val_rec_loss": val_recon_epoch_loss_list, "val_ssim": epoch_ssim_val_list, "val_psnr": epoch_psnr_val_list})
    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()
    print('Reutrning the last and best AEs.')
    return autoencoder, ae_best


'''
# 5) Iterate
for batch in loader:
    imgs, segs = batch["image"], batch["label"]
    # ... forward pass ...
'''



def train_ldm(unet, train_loader, autoencoder, ldm_epochs = 150,
              lr=1e-4, scale_factor=None, torch_autocast = True,
              device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
              outdir = 'ckpts', sample_every = 25):

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

    with torch.no_grad(), torch.autocast("cuda", enabled=torch_autocast):
        #with autocast(device_type = 'cuda', enabled=(device.type == "cuda")):
                check_data = first(train_loader)
                z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
                #print('AUTOENCODER_Z\n', z)
    if scale_factor == None:
        scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")


    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=lr)


    n_epochs = ldm_epochs
    epoch_loss_list = []
    autoencoder.eval()
    scaler = GradScaler()

    unet_best = None
    unet_best_loss = float('inf')

    for p in autoencoder.parameters(): p.requires_grad = False
    for epoch in range(n_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer_diff.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = autoencoder.encode_stage_2_inputs(images) * scale_factor

            with autocast(device_type = 'cuda', enabled=torch_autocast):
                # Generate random noise
                noise = torch.randn_like(z).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                noise_pred = inferer(
                    inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
                )
                #print('NOISE_PRED\n', noise_pred)

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if epoch_loss / (step + 1) < unet_best_loss:
            print('Updating best UNet + Conditioner checkpoint...')
            unet_best_loss = epoch_loss / (step + 1)
            unet_best = copy.deepcopy(unet)
            _save_ckpt(os.path.join(outdir, "UNET_best.pt"), unet, opt=optimizer_diff, scaler=scaler, epoch=epoch+1, global_step=None, 
                extra={"train_loss": epoch_loss_list, "scale_factor": float(scale_factor)})
            sample_ldm(train_loader, autoencoder, unet, scheduler, inferer, device, idx = 0, channel = 0, 
                       outdir = outdir, filename = f'synthetic_ep{epoch+1}_BEST')

        if (epoch + 1) % sample_every == 0:
            sample_ldm(train_loader, autoencoder, unet, scheduler, inferer, device, idx = 0, channel = 0, 
                       outdir = outdir, filename = f'synthetic_ep{epoch+1}')
            
        _save_ckpt(os.path.join(outdir, "UNET_last.pt"), unet, opt=optimizer_diff, scaler=scaler, epoch=epoch+1, global_step=None, 
                    extra={"train_loss": epoch_loss_list, "scale_factor": float(scale_factor)})
        plot_unet_loss(epoch_loss_list, title = f'UNET Loss Curves_ep{epoch+1}', outdir = outdir, filename = 'UNET_loss.png')





# ------------
# CLI wrapper
# ------------
def main():
    ap = argparse.ArgumentParser(description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.")
    ap.add_argument("--csv", required=True, help="Path to CSV with columns: image[/path], sex, age, vol, [target_label]")
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True, help='Use torch autocast to accelerate: True or False.')
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--train_val_split", type=float, default=0.1)


    ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="both")


    # AE config
    ap.add_argument("--ae_epochs", type=int, default=50)
    ap.add_argument("--ae_lr", type=float, default=1e-4)
    ap.add_argument("--ae_latent_ch", type=int, default=3)
    ap.add_argument("--ae_kl", type=float, default=1e-6)
    ap.add_argument("--ae_adv_weight", type=float, default=0.01)
    ap.add_argument("--ae_perceptual_weight", type=float, default=0.001)
    ap.add_argument("--ae_kl_weight", type=float, default=1e-6)
    ap.add_argument("--ae_num_channels", default="64,128,256,512")
    #ap.add_argument("--ae_factors", default="1,2,2,2")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt (optional)")
    #ap.add_argument("--ae_decoder_only", action="store_true", help="Fine-tune decoder only")


    # LDM config
    ap.add_argument("--ldm_epochs", type=int, default=150)
    ap.add_argument("--ldm_lr", type=float, default=1e-4)
    ap.add_argument("--ldm_use_cond", default="False", help="Use [part_vol_norm,sex,age] conditioning")
    ap.add_argument("--ldm_num_channels", default="128,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_ckpt", default="", help="Resume UNet weights (optional)")
    ap.add_argument("--ldm_sample_every", type=int, default=25, help="Synthesize samples every N epochs")


    ap.add_argument("--outdir", default="ckpts")
    ap.add_argument("--out_prefix", default="")


    args = ap.parse_args()

    from datetime import datetime
    import json
    experiment_dir = os.path.join(args.outdir, f"run_{args.out_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    args.outdir = experiment_dir
    print(f"\n✅ Output dir: {args.outdir}\n")

    # argparse Namespace -> dict
    cfg = vars(args).copy()

    # Make non-JSON types serializable (e.g., tuples)
    for k,v in list(cfg.items()):
        if isinstance(v, tuple): cfg[k] = list(v)
    with open(os.path.join(args.outdir, 'args.json'), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size    = tuple(int(x)   for x in args.size.split(","))

    channel = 0  # 0 = Flair
    keys = ['image']
    train_transforms = transforms.Compose(
        [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        transforms.EnsureTyped(keys=keys),
        transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
        transforms.CropForegroundd(keys=keys, source_key='image'),
        transforms.DivisiblePadd(keys=keys, k=32, mode="constant",constant_values=-1.0),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️ Using {device}\n")
    
    seed = 1017
    g = torch.Generator(device=device).manual_seed(seed)
    random.seed(1017)
    seed_all(1017)

    try:
        n_samples = int(args.n_samples)
    except:
        n_samples = args.n_samples
    train_loader, val_loader = make_dataloaders_from_csv(args.csv, conditions = ['age', 'sex', 'vol'], train_transforms = train_transforms,
                                                         n_samples=n_samples, train_val_split = args.train_val_split, batch_size = args.batch, 
                                                         num_workers=args.workers, seed = 1017)

    ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
    ae_latent_ch = int(args.ae_latent_ch)


    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1, 
        num_channels=ae_num_channels, # default: (64, 128, 256, 512)
        latent_channels=ae_latent_ch, # default: 3
        num_res_blocks=2, # default: 2
        norm_num_groups=32, # default: 32
        norm_eps=1e-06,
        attention_levels=(False, False, True, True),
    )
    autoencoder.to(device)

    if args.ae_ckpt != "":
        _load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)

    if args.stage in ["ae", "both"]:
        last_ae, best_ae = train_ae(autoencoder, train_loader, val_loader, val_interval = 1, ae_epochs = args.ae_epochs, 
                                    adv_weight = args.ae_adv_weight, perceptual_weight = args.ae_perceptual_weight, kl_weight = args.ae_kl_weight, 
                                    lr=args.ae_lr, device = device, outdir = args.outdir)
        if best_ae:
            autoencoder = best_ae
    

    ldm_num_channels = tuple(int(x) for x in args.ldm_num_channels.split(","))
    ldm_num_head_channels = tuple(int(x) for x in args.ldm_num_head_channels.split(","))
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3, # default: 3
        out_channels=3, # default: 3
        num_res_blocks=2,
        num_channels=ldm_num_channels, # default: (128, 256, 512)
        attention_levels=(False, True, True),
        num_head_channels=ldm_num_head_channels, # default: (0, 64, 64)
        norm_num_groups=32, norm_eps=1e-6,
        resblock_updown=True,
        upcast_attention=True,
        use_flash_attention=False
    )
    unet.to(device)

    scale_factor = None

    if args.ldm_ckpt != "":
        _load_ckpt_into_unet(unet, args.ldm_ckpt, device)
        ckpt = torch.load(args.ldm_ckpt, map_location=device)
        try:
            ckpt = torch.load(args.ldm_ckpt, map_location=device)
            scale_factor = float(ckpt['extra']['scale_factor'])
            print('Loading scale factor = ', scale_factor)
        except:
            scale_factor = None

    if args.stage in ["ldm", "both"]:
        train_ldm(unet, train_loader, autoencoder, ldm_epochs = args.ldm_epochs,
              lr=args.ldm_lr, scale_factor=scale_factor, torch_autocast=args.torch_autocast,
              device = device, outdir = args.outdir, sample_every = args.ldm_sample_every)


if __name__ == "__main__":
   main()
