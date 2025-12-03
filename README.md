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


### 12/03/25

With conditions; whole brain and parts generation.

Download the datasets from Hugging Face, place them under the /data directory in this project.

<pre>
./data
├── ADNI_turboprepout_lhemi_1201
├── ADNI_turboprepout_lhemi_1201
├── ADNI_turboprepout_cerebellum_1201
├── cond_data_3parts_1201.csv
└── ... (other folders if there exists any, irrelevant)
</pre>

- [Left Hemisphere](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_lhemi_1201)
- [Right Hemisphere](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_rhemi_1201)
- [Cerebellum](https://huggingface.co/tracyhan816/3D_Brain_partsLDM/tree/main/data/ADNI_turboprepout_cerebellum_1201)

#### 3 parts condition LDM
Adjust the visible GPUs as needed 

<pre>
CUDA_VISIBLE_DEVICES=0, \
python scripts/train_3d_brain_ldm_cond.py \
  --csv data/cond_data_3parts_1201.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 1000 \
  --ldm_num_parts 3 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 100 \
  --out_prefix cond_3parts_gen
</pre>
