Dockerfile contains the conda environment 'monai' that can run this project. Mostly, the environment includes MONAI and MONAI-generative tools.

pip install cu128 if running on sm120 GPU

`cd <PROJECT_DIR>`

sudo docker run --gpus all -it   -v "$PWD":/workspace   -w /workspace   h8w108/3dbrain:parts_ldm_20251013  bash

#  If encountering conda init issue
## load conda’s bash hook for this shell
source /opt/conda/etc/profile.d/conda.sh  \
  || eval "$(/opt/conda/bin/conda shell.bash hook)"

`conda activate monai`

# SynthSeg parts definitions
https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg



# Training
## Original pretrained weights
models/autoencoder.pt
models/model.pt

## Finetuned model weights 
ckpts/ae_best.pt
ckpts/ldm_last.pt

## args
ap = argparse.ArgumentParser(description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.")
ap.add_argument("--csv", required=True, help="Path to CSV with columns: image[/path], sex, age, [seg/seg_path], [target_label]")
ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
ap.add_argument("--size", default="160,224,160", help="Canvas D,H,W (e.g., 160,224,160)")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--val_frac", type=float, default=0.1)


ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="both")


### AE config
ap.add_argument("--ae_epochs", type=int, default=50)
ap.add_argument("--ae_lr", type=float, default=2e-4)
ap.add_argument("--ae_latent_ch", type=int, default=3)
ap.add_argument("--ae_kl", type=float, default=1e-6)
ap.add_argument("--ae_lpips_w", type=float, default=0.1)
ap.add_argument("--ae_channels", default="64,128,128,128")
ap.add_argument("--ae_factors", default="1,2,2,2")
ap.add_argument("--ae_finetune_ckpt", default="", help="Path to pretrained AE .pt (optional)")
ap.add_argument("--ae_decoder_only", action="store_true", help="Fine-tune decoder only")


### LDM config
ap.add_argument("--ldm_epochs", type=int, default=100)
ap.add_argument("--ldm_lr", type=float, default=1e-4)
ap.add_argument("--ldm_use_cond", action="store_true", help="Use [part_vol_norm,sex,age] conditioning")
ap.add_argument("--ldm_channels", default="128,256,512")
ap.add_argument("--ldm_resume", default="", help="Resume UNet weights (optional)")


ap.add_argument("--outdir", default="ckpts")

## Train 3D AE (-- stage ae); train full AE by removing --ae_decoder_only

python scripts/train_3d_ldm_from_parts.py \
  --csv data/cerebral_data.csv \
  --stage ae \
  --ae_finetune_ckpt models/autoencoder.pt --ae_decoder_only \
  --ldm_use_cond \
  --batch 1 --workers 0 \
  --spacing 2,2,2 --size 128,160,128 

## Train 3D LDM (-- stage ldm)
python scripts/train_3d_ldm_from_parts_pretrained.py \
  --csv data/cerebral_data.csv \
  --stage ldm \
  --ae_finetune_ckpt ckpts/run_lulin_1022/ae_best.pt --ae_decoder_only \
  --batch 1 --workers 0 \
  --spacing 2,2,2 --size 128,160,128 \
  --ldm_channels 256,512,768 \
  --ldm_use_cond True \
  --ldm_resume models/model.pt

## Train full model (-- stage both)
python scripts/train_3d_ldm_from_parts.py \
  --csv data/cerebral_data.csv \
  --stage both \
  --ae_finetune_ckpt models/autoencoder.pt  --ae_decoder_only\
  --ldm_use_cond \
  --batch 1 --workers 0 \
  --spacing 2,2,2 --size 128,160,128 --ae_epochs 1

## To avoid OOM issue
python scripts/train_3d_ldm_from_parts.py \
  --csv data/cerebral_WM_data.csv \
  --stage both \
  --ae_finetune_ckpt models/autoencoder.pt --ae_decoder_only \
  --ldm_use_cond \
  --batch 1 --workers 0 \
  --spacing 2,2,2 --size 128,160,128


# Inference
## AE inference
python scripts/ae_recon_infer.py \
  --csv data/cerebral_data.csv \
  --ae_ckpt models/autoencoder.pt \
  --spacing 2,2,2 \
  --size 128,160,128 \
  --latent_ch 3 \
  --ae_channels 64,128,128,128 \
  --outdir ckpts/ae_recons \
  --max_save 8

## LDM inference
python scripts/3d_ldm_infer.py
--ae_ckpt ckpts/ae_best.pt
--unet_ckpt ckpts/ldm_last.pt
--size 160,224,160
--latent_ch 3
--ae_channels 64,128,128,128
--unet_channels 128,256,512
--steps 100 --eta 0.0 --num 4
--outdir samples_ldm

python scripts/3d_ldm_infer.py \
  --ae_ckpt ckpts/run_lulin_1022/ae_best.pt \
  --unet_ckpt ckpts/run_lulin_1022/ldm_last.pt \
  --size 160,224,160 \
  --latent_ch 3 \
  --use_cond \
  --ae_channels 64,128,128,128 \
  --unet_channels 128,256,512 \
  --steps 50 --eta 0.0 --num 4 \
  --outdir output/myscript/1022

python scripts/3d_ldm_infer_pretrained.py \
  --ae_ckpt ckpts/run_lulin_1022/ae_best.pt \
  --unet_ckpt ckpts/run_20251028_071951/ldm_best.pt \
  --size 160,224,160 \
  --latent_ch 3 \
  --use_cond \
  --ae_channels 64,128,128,128 \
  --unet_channels 256,512,768 \
  --steps 50 --eta 0.0 --num 4 \
  --outdir output/myscript/1030bs1

## Conditional LDM Example: vol_norm=0.40, sex=1 (male), age_norm=0.35, guidance scale 3.0
python scripts/3d_ldm_infer.py \
  --ae_ckpt ckpts/run_lulin_1022/ae_best.pt \
  --unet_ckpt ckpts/run_lulin_1022/ldm_last.pt \
  --size 160,224,160 \
  --latent_ch 3 \
  --ae_channels 64,128,128,128 \
  --unet_channels 128,256,512 \
  --steps 250 --eta 0.0 --num 4 \
  --use_cond --cond 0.75,1,0.28 --cfg_scale 1.0 \
  --outdir samples_ldm_cond


