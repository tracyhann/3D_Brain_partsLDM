`cd <PROJECT_DIR>`
<pre>
docker run --gpus all -it   -v "$PWD":/workspace   -w /workspace   h8w108/3dbrain:parts_ldm_20251013  bash
</pre>

###  If encountering conda init issue
Load conda’s bash hook for this shell
<pre>
source /opt/conda/etc/profile.d/conda.sh  \
  || eval "$(/opt/conda/bin/conda shell.bash hook)"
</pre>

# Inference & Eval

## 01/14/26
The script will generate 300 samples, saved in nifti files, and run evaluations.
### Fill in path to ckpts for both AE and LDM; make sure data csv path is correct as well.
Data csvs:
- lhemi_0107.csv
- rhemi_0107.csv
- whole_brain_0107.csv
- cerebellum_0107.csv

Part names: {lhemi, rhemi, whole_brain, cerebellum}

For cLDM:
<pre>
  python scripts/3dgen_eval.py \
  --csv data/{PART_NAME}_0107.csv \
  --generate_n_samples 300 \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 2 \
  --workers 0 \
  --train_val_split 0.1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/{FOLDER}/AE_best.pt \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_ckpt ckpts/{FOLDER}/UNET_last.pt \
  --outdir samples/brain_parts_runs_20260112/{PART_NAME}
</pre>

For LDM:
<pre>
  python scripts/3dgen_eval.py \
  --csv data/{PART_NAME}_0107.csv \
  --generate_n_samples 300 \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 2 \
  --workers 0 \
  --train_val_split 0.1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/{FOLDER}/AE_best.pt \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_ckpt ckpts/{FOLDER}/UNET_last.pt \
  --outdir samples/brain_parts_runs_20260110/{PART_NAME}
</pre>

# Experiments

## 01/08/26

### Download dataset from huggingface
https://huggingface.co/datasets/tracyhan816/ADNI_subset

... place the dataset under /data -> `data/turboprep_out_1114`

### Preprocess data
`python data_prep/prep_data.py`

### 3D LDM baseline experiments
#### Whole brain
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/whole_brain_0107.csv \
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
  --ldm_sample_every 10 \
  --out_prefix whole_brain_LDM_fixed_scale_0107
</pre>

#### Left hemi
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/lhemi_0107.csv \
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
  --ldm_sample_every 10 \
  --out_prefix lhemi_LDM_fixed_scale_0107
</pre>

#### Right hemi
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/rhemi_0107.csv \
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
  --ldm_sample_every 10 \
  --out_prefix rhemi_LDM_fixed_scale_0107
</pre>

#### Cerebellum
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
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
  --ldm_sample_every 10 \
  --out_prefix cerebellum_LDM_fixed_scale_0107
</pre>


### 3D cLDM experiments
#### Whole brain
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/whole_brain_0107.csv \
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
  --ldm_sample_every 10 \
  --out_prefix whole_brain_cLDM_fixed_scale_0107
</pre>

#### Left hemi
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/lhemi_0107.csv \
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
  --ldm_sample_every 10 \
  --out_prefix lhemi_cLDM_fixed_scale_0107
</pre>

#### Right hemi
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/rhemi_0107.csv \
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
  --ldm_sample_every 10 \
  --out_prefix rhemi_cLDM_fixed_scale_0107
</pre>

#### Cerebellum
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-5 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix cerebellum_cLDM_fixed_scale_0107
</pre>

## 01/12/26
#### Cerebellum; 450 samples; AE and UNET @ lr=1e-5
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples ALL \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-5 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-5 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_LDM_fixed_scale_0107_1e-5_450
</pre>

#### Cerebellum; 100 samples; AE and UNET @ lr=1e-5
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples 100 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-5 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-5 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_LDM_fixed_scale_0107_1e-5_100
</pre>

#### Cerebellum; 450 samples; AE and UNET @ lr=1e-6
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples ALL \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-6 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-6 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_LDM_fixed_scale_0107_1e-6_450
</pre>

#### Cerebellum; 100 samples; AE and UNET @ lr=1e-6
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples 100 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-6 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-6 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_LDM_fixed_scale_0107_1e-6_100
</pre>

## 01/13/26
#### Cerebellum; 100 samples; AE @ lr=1e-4 UNET @ lr=1e-6
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples 100 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-6 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_0107_ldm1e-6_100
</pre>

<pre>
python scripts/train_3d_brain_ldm_crop.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples 100 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_0107_bothcrop_100
</pre>

<pre>
python scripts/train_3d_brain_ldm_fgcrop.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples 100 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_0107_fgcrop_100
</pre>

<pre>
python scripts/train_3d_brain_ldm_ctcrop.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples 100 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_0107_ctcrop_100
</pre>

## 01/14/2023 
### Updates: Cerebellum AE @ lr = 1e-4 and UNET @ lr = 1e-6 worked on 100 samples
<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples ALL \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-6 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_LDM_fixed_scale_0107_1e-6_450
</pre>

<pre>
python scripts/train_3d_brain_ldm_.py \
  --csv data/cerebellum_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --n_samples ALL \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-5 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 1 \
  --out_prefix cerebellum_LDM_fixed_scale_0107_1e-5_450
</pre>
