#
`cd <PROJECT_DIR>`
# Docker
https://hub.docker.com/r/h8w108/3dbrain

<pre>
docker run --gpus all -it   -v "$PWD":/workspace   -w /workspace   h8w108/3dbrain:parts_ldm_20251013  bash
</pre>

###  If encountering conda init issue
Load conda’s bash hook for this shell
<pre>
source /opt/conda/etc/profile.d/conda.sh  \
  || eval "$(/opt/conda/bin/conda shell.bash hook)"
</pre>
<pre>
conda activate monai
</pre>

## Data processing
### Step 1: Cropping and Resizing
<pre>
python data/turboprep_postproc.py 
  --root data/ADNI_0206/turboprep_out \
  --outdir data/ADNI_0206/turboprep_out \
  --n_samples ALL
</pre>

### Step 2: Generate part data and .csvs for training
- After post processing the turboprep output files, generate whole brain, part data and corresponding .csv files with conditions. These will be for training.
- `$OUTDIR` refers to the directory containing post-processed turboprep output files. Each normalized scan has shape (160, 192, 160).
- By default, this step will also generate the combined .csv of all parts and masks, and hemi.csv.
<pre>
python data_prep/prep_data.py \
  --root data/ADNI_0206/turboprep_out \
  --part whole_brain,left_hemi,right_hemi,right_hemi_mirror,sub \
  --outdir data/processed_parts \
  --postfix 0206 
</pre>


## Train whole_brain, hemi, and sub AEs

### AE for whole brain
<pre>
python scripts/train_3d_VAE.py \
  --csv data/processed_parts/whole_brain_0206.csv \
  --spacing 1.5,1.5,1.5 \
  --batch 1 \
  --n_samples ALL \
  --workers 0 \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --outdir ckpts/AE \
  --out_prefix whole_brain_AE \
  --out_postfix 0214
</pre>

### AE for hemispheres
<pre>
python scripts/train_3d_VAE.py \
  --csv data/processed_parts/hemi_0206.csv \
  --spacing 1.5,1.5,1.5 \
  --batch 1 \
  --n_samples ALL \
  --workers 0 \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --outdir ckpts/AE \
  --out_prefix hemi_AE \
  --out_postfix 0214
</pre>

### AE for sub
<pre>
python scripts/train_3d_VAE.py \
  --csv data/processed_parts/sub_0206.csv \
  --spacing 1.5,1.5,1.5 \
  --batch 1 \
  --n_samples ALL \
  --workers 0 \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_num_channels 64,128,256,512 \
  --outdir ckpts/AE \
  --out_prefix sub_AE \
  --out_postfix 0214
</pre>
