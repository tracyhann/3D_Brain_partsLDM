This README contains logs of experiments for training brain 3D-LDM. - Tracy 11/16/2025 updates.

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
We will train our own models. The model configurations of pretrained models are not the most compatible for this project.

## Finetuned model weights 
Copy above.

## args
<pre>
ap = argparse.ArgumentParser(description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.")
ap.add_argument("--csv", required=True, help="Path to CSV with columns: image[/path], sex, age, [seg/seg_path], [target_label]")
ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
ap.add_argument("--size", default="160,224,160", help="Canvas D,H,W (e.g., 160,224,160)")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--val_frac", type=float, default=0.1)


ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="both")
</pre>


### AE config
<pre>
ap.add_argument("--ae_epochs", type=int, default=50)
ap.add_argument("--ae_lr", type=float, default=1e-4)
ap.add_argument("--ae_latent_ch", type=int, default=3)
ap.add_argument("--ae_kl", type=float, default=1e-6)
ap.add_argument("--ae_adv_weight", type=float, default=0.01)
ap.add_argument("--ae_perceptual_weight", type=float, default=0.001)
ap.add_argument("--ae_kl_weight", type=float, default=1e-6)
ap.add_argument("--ae_num_channels", default="64,128,256,512")
ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt (optional)")
</pre>


### LDM config
<pre>
ap.add_argument("--ldm_epochs", type=int, default=150)
ap.add_argument("--ldm_lr", type=float, default=1e-4)
ap.add_argument("--ldm_use_cond", default="False", help="Use [part_vol_norm,sex,age] conditioning")
ap.add_argument("--ldm_num_channels", default="128,256,512")
ap.add_argument("--ldm_num_head_channels", default="0,64,64")
ap.add_argument("--ldm_ckpt", default="", help="Resume UNet weights (optional)")
ap.add_argument("--ldm_sample_every", type=int, default=25, help="Synthesize samples every N epochs")


ap.add_argument("--outdir", default="ckpts")
</pre>


## Best case scenario if VRAM allows: 
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/whole_brain_data_1114.csv \
  --spacing 1,1,1 \
  --size 192,224,192 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>

## On a single 5090 GPU (VRAM < 30G):
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/whole_brain_data_1114.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_ckpt ckpts/run_20251117_190756/AE_last.pt \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
python scripts/train_3d_brain_ldm.py \
  --csv data/whole_brain_data_1114.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_ckpt ckpts/run_20251117_190756/AE_last.pt \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>

## On limited VRAM, can also try:
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/whole_brain_data_1114.csv \
  --spacing 2,2,2 \
  --size 128,160,128 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/whole_brain_data_1114.csv \
  --spacing 1,1,1 \
  --size 160,192,160 \
  --batch 1 \
  --workers 8 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>



## Experiments

### 11/16/25

Without conditions; whole brain and parts generation.

Download the datasets from Hugging Face, place them under the /data directory in this project.

<pre>
./data
├── ADNI_turboprepout_cerebellum_1114
├── ADNI_turboprepout_cerebral_1114
├── ADNI_turboprepout_left_hemi_1114
├── ADNI_turboprepout_right_hemi_1114
├── ADNI_turboprepout_whole_brain_1114
├── cerebellum_data_1114.csv
├── cerebral_data_1114.csv
├── left_hemi_data_1114.csv
├── right_hemi_data_1114.csv
├── whole_brain_data_1114.csv
└── ... (other folders if there exists any, irrelevant)
</pre>

- [Whole Brain](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_whole_brain_1114)
- [Left Hemisphere (excluding cerebellum)](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_left_hemi_1114)
- [Right Hemisphere (excluding cerebellum)](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_right_hemi_1114)
- [Cerebellum](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_cerebellum_1114)
- [Cerebral](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_cerebral_1114)

#### Whole Brain:
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/whole_brain_data_1114.csv \
  --spacing 1,1,1 \
  --size 192,224,192 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>

#### Left Hemi:
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/left_hemi_data_1114.csv \
  --spacing 1,1,1 \
  --size 192,224,192 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>

#### Right Hemi:
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/right_hemi_data_1114.csv \
  --spacing 1,1,1 \
  --size 192,224,192 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>

#### Cerebellum:
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/cerebellum_data_1114.csv \
  --spacing 1,1,1 \
  --size 192,224,192 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>

#### Cerebral:
<pre>
python scripts/train_3d_brain_ldm.py \
  --csv data/cerebral_data_1114.csv \
  --spacing 1,1,1 \
  --size 192,224,192 \
  --batch 1 \
  --workers 4 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25
</pre>


### 11/17/25 Tracy's experiments

#### 
python scripts/train_3d_brain_ldm.py \
  --csv data/left_hemi_data_1114.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25

python scripts/train_3d_brain_ldm.py \
  --csv data/cerebellum_data_1114.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/brain_parts_runs_20251118/run_whole_brain_recon_20251117_190756/AE_best.pt \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25 \
  --out_prefix cerebellum_from_whole_brain_AE


python scripts/3dgen_eval.py \
  --csv data/whole_brain_data_1114.csv \
  --generate_n_samples 300 \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/run_20251117_190756/AE_best.pt \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_ckpt ckpts/run_20251118_030349/UNET_last.pt


python scripts/3dgen_eval.py \
  --csv data/whole_brain_data_1114.csv \
  --generate_n_samples 0 \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 2 \
  --workers 0 \
  --train_val_split 0.1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/run_20251117_190756/AE_best.pt \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_ckpt ckpts/run_20251118_030349/UNET_last.pt


python scripts/3dgen_eval.py \
  --csv data/whole_brain_data_1114.csv \
  --generate_n_samples 0 \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 2 \
  --workers 0 \
  --train_val_split 0.1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/brain_parts_runs_20251118/run_20251118_154755/AE_best.pt \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_ckpt ckpts/brain_parts_runs_20251118/run_20251118_154755/UNET_last.pt


### 11/23 Tracy's experiment

python scripts/train_3d_brain_ldm.py \
  --csv data/cerebellum_data_1114.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/brain_parts_runs_20251118/run_whole_brain_recon_20251117_190756/AE_best.pt \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25 \
  --out_prefix cerebellum_from_whole_brain_AE

python scripts/train_3d_brain_ldm.py \
  --csv data/cerebellum_data_1114.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_epochs 0 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/run_cerebellum_from_whole_brain_AE_20251123_231609/AE_best.pt \
  --ldm_epochs 150 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 25 \
  --out_prefix cerebellum_from_whole_brain_AE


### 11/28 Tracy's on conditional ldm

python scripts/train_3d_brain_ldm_cond.py \
  --csv data/cond_data_5parts_exp.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_epochs 1 \
  --ae_ckpt ckpts/run_cond_5parts_gen_exp_20251129_144532/AE_best.pt \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 1000 \
  --ldm_ckpt ckpts/run_cond_5parts_gen_exp_20251129_031258/UNET_last.pt \
  --ldm_num_parts 5 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 200 \
  --out_prefix cond_5parts_gen_exp


python scripts/train_3d_brain_ldm_cond.py \
  --csv data/cond_data_2parts_exp.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 500 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 1000 \
  --ldm_num_parts 2 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 200 \
  --out_prefix cond_2parts_gen_exp


python scripts/train_3d_brain_ldm_cond.py \
  --csv data/cond_data_3parts_exp_1201.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 500 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 1000 \
  --ldm_num_parts 2 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 200 \
  --out_prefix cond_3parts_gen_exp