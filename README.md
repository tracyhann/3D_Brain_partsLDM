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

## 01/14/2026 
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
  --out_prefix cerebellum_LDM_fixed_scale_0107_ldm1e-6_450
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
  --out_prefix cerebellum_LDM_fixed_scale_0107_ldm1e-5_450
</pre>

## 01/15/2026

### Train fusion model from coarse compose
<pre>
python scripts/train_3d_brain_ldm_from_coarse.py \
  --csv data/whole_brain+3parts+masks_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --n_samples ALL \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_epochs 1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/run_whole_brain_LDM_fixed_scale_0107_20260110_225454/AE_best.pt \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix whole_brain_coarse_uncond_CLAMP_1e-4_450
</pre>

<pre>
python scripts/train_3d_brain_ldm_from_coarse.py \
  --csv data/whole_brain+3parts+masks_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --n_samples ALL \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_epochs 1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/run_whole_brain_LDM_fixed_scale_0107_20260110_225454/AE_best.pt \
  --ldm_epochs 150 \
  --ldm_lr 1e-5 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix whole_brain_coarse_uncond_CLAMP_1e-5_450
</pre>

<pre>
python scripts/train_3d_brain_ldm_from_coarse.py \
  --csv data/whole_brain+3parts+masks_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --n_samples ALL \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_epochs 1 \
  --ae_num_channels 64,128,256,512 \
  --ae_ckpt ckpts/run_whole_brain_LDM_fixed_scale_0107_20260110_225454/AE_best.pt \
  --ldm_epochs 150 \
  --ldm_lr 1e-6 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix whole_brain_coarse_uncond_CLAMP_1e-6_450
</pre>

## 01/17/26
### 3D cLDM experiments
- Right hemispheres struggled to be learned in the previous experiments, but left hemisphere were fine.

#### Right hemi; UNet lr=1e-6
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/rhemi_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_epochs 1 \
  --ae_ckpt ckpts/run_rhemi_cLDM_fixed_scale_0107_20260112_063405/AE_best.pt \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-6 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix rhemi_cLDM_ldm1e-6_0107
</pre>

#### Right hemi; UNet lr = 1e-5
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/rhemi_0107.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage ldm \
  --ae_epochs 1 \
  --ae_ckpt ckpts/run_rhemi_cLDM_fixed_scale_0107_20260112_063405/AE_best.pt \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-5 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix rhemi_cLDM_ldm1e-5_0107
</pre>


## 01/20/26
### Mirrored LDM (CONCAT + CROSSATTN conditioning)
- The brain is overall symmetrical with subtle differences giving rise to functional specialization.
#### Generate mirroring brain data 
<pre>
  python data_prep/prep_data.py \
  --part left,right,right_mirror,whole_brain \
  --postfix 0120
</pre>
#### Generate data csv file.
<pre>
  python data_prep/pair_mirrored_LR_csv.py
</pre>

#### Train LDM on paired images

- Train at spacing = `2,2,2`; lr = `1e-4`

<pre>
python scripts/train_3d_brain_mirror_ldm.py \
  --csv data/left_right_paired_0120.csv \
  --spacing 2,2,2 \
  --size 96,128,96 \
  --n_samples ALL \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix left_right_paired_450
</pre>

- Train at spacing = `1,1,1`; lr = `1e-4`
- If OOM, adjust spacing; can try `1.2,1.2,1.2`, etc.

<pre>
python scripts/train_3d_brain_mirror_ldm.py \
  --csv data/left_right_paired_0120.csv \
  --spacing 1,1,1 \
  --size 96,128,96 \
  --n_samples ALL \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix left_right_paired_450
</pre>

- Train at spacing = `1,1,1`; lr = `1e-5`

<pre>
python scripts/train_3d_brain_mirror_ldm.py \
  --csv data/left_right_paired_0120.csv \
  --spacing 1,1,1 \
  --size 96,128,96 \
  --n_samples ALL \
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
  --out_prefix left_right_paired_450
</pre>


## 01/22/26

### Generate sub part including brain stem, ventricles, and cerebellum
<pre>
  python data_prep/prep_data.py \
  --part sub,right_hemi_mirror,left_hemi,hemi \
  --postfix 0120
</pre>

- For all generative experiments below, feel free to adjust `--spacing` to as close as possible to `1,1,1` without causing OOMs.
  
### LDM with better transforms of data
#### LDM of subpart
<pre>
  python scripts/train_3d_brain_ldm_.py \
  --csv data/sub_0120.csv \
  --spacing 1.2,1.2,1.2 \
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
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix sub_LDM_0120_ldm1e-4_450
</pre>
#### LDM of hemispheres (left and right mirrored hemispheres)
<pre>
  python scripts/train_3d_brain_ldm_.py \
  --csv data/hemi_0120.csv \
  --spacing 1.5,1.5,1.5 \
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
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix hemi_LDM_0120_ldm1e-4_450
</pre>

### cLDM with better transforms of data
#### cLDM of subpart
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/sub_0120.csv \
  --spacing 1.2,1.2,1.2 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --n_samples ALL \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix sub_cLDM_0120_ldm1e-4_450
</pre>

#### cLDM of hemispheres (left and right mirrored hemispheres)
<pre>
python scripts/train_3d_brain_cond_ldm.py \
  --csv data/hemi_0120.csv \
  --spacing 1.5,1.5,1.5 \
  --size 96,128,96 \
  --batch 1 \
  --workers 0 \
  --train_val_split 0.1 \
  --n_samples ALL \
  --stage both \
  --ae_epochs 100 \
  --ae_num_channels 64,128,256,512 \
  --ldm_epochs 150 \
  --ldm_lr 1e-4 \
  --ldm_num_channels 128,256,512 \
  --ldm_num_head_channels 0,64,64 \
  --ldm_sample_every 10 \
  --out_prefix hemi_cLDM_0120_ldm1e-4_450
</pre>
