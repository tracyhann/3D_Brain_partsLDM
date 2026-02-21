
`cd <PROJECT_DIR>`


<details>
<summary><strong>Environment</strong></summary>

<details>
<summary><strong>Docker</strong></summary>

https://hub.docker.com/r/h8w108/3dbrain

<pre>
docker run --gpus all -it   -v "$PWD":/workspace   -w /workspace   h8w108/3dbrain:parts_ldm_20251013  bash
</pre>

### If encountering conda init issue
Load conda’s bash hook for this shell

<pre>
source /opt/conda/etc/profile.d/conda.sh  \
  || eval "$(/opt/conda/bin/conda shell.bash hook)"
</pre>

<pre>
conda activate monai
</pre>

</details>

<details>
<summary><strong>yaml</strong></summary>

Alternatively, you can build a conda environment using this yaml file:

https://github.com/lulinliu/MineLongTail/blob/main/monaifull.yml

</details>

</details>

<details>
<summary><strong>Data</strong></summary>

<details>
<summary><strong>Download data</strong></summary>

Download data to the `/data` dir. The directory should look like: `/data/ADNI_0206/*`.

### ADNI_0206
This directory contains ADNI NIfTI scans, input/output path lists, and the outputs from a turboprep run.

#### Contents
- Dataset dir structure.

<pre>
.
├── input_0206.txt
├── input.txt
├── MNI152_T1_1mm_brain.nii
├── output_0206.txt
├── output.txt
├── raw
    ├── XXX.nii
    └── ...
├── README.md
├── scripts
    ├── turboprep_postproc.py
    └── turboprep_preproc.py
└── turboprep_out
    └── ADNI_941_S_1311_MR_MPR__GradWarp_Br_20081026142330778_S56645_I123814
        ├── affine_transf.mat
        ├── normalized.nii.gz
        ├── mask.nii.gz
        └── segm.nii.gz
</pre>

- `raw/` original NIfTI inputs organized by subject/session folders.
- `turboprep_out/` per-scan output directories.
- `MNI152_T1_1mm_brain.nii` template volume file.
- `input.txt` input file paths for preprocessing. 1,735 total.
- `output.txt` output file paths for preprocessing.
- `input_0206.txt` runtime log of input file paths during preprocessing (identical to input.txt).
- `output_0206.txt` runtime log of output file paths during preprocessing (identical to output.txt).

- If you wish to check out dataset repo structure on Huggingface:

<pre>
python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
items = set()
for p in api.list_repo_files("nnuochen/ADNI", repo_type="dataset"):
    items.add(p.split("/", 1)[0])
for name in sorted(items):
    print(name)
PY
</pre>

#### Structure of `turboprep_out`
- Contains 1,735 preprocessed sessions.
- Each session contains:
    - affine_transf.mat
    - mask.nii.gz
    - normalized.nii.gz
    - segm.nii.gz

</details>

<details>
<summary><strong>Data processing</strong></summary>

### Step 1: Cropping and Resizing
- Normalize intensities to [-1, 1] where background is defined as -1. Normalization percentiles computed withn head masks.
- Foreground cropping and pad the volumes, masks, and segmentations to standardized shape: (160,192,160)
- Converting to parts: whole_brain (160,192,160), left_hemi/right_hemi/right_hemi_mirror (96,192,160), sub (160,128,96)

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

</details>

</details>


<details>
<summary><strong>Train AEs</strong></summary>

## AE for whole brain
<pre>
python scripts/train_3d_VAE.py \
  --csv data/processed_parts/whole_brain_0206.csv \
  --spacing 1.5,1.5,1.5 \
  --batch 2 \
  --n_samples ALL \
  --workers 0 \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_latent_ch 8 \
  --ae_num_channels 64,128,256 \
  --outdir ckpts/AE \
  --out_prefix whole_brain_AE \
  --out_postfix 0214
</pre>

## AE for hemispheres
<pre>
python scripts/train_3d_VAE.py \
  --csv data/processed_parts/hemi_0206.csv \
  --spacing 1.5,1.5,1.5 \
  --batch 2 \
  --n_samples ALL \
  --workers 0 \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_latent_ch 8 \
  --ae_num_channels 64,128,256 \
  --outdir ckpts/AE \
  --out_prefix hemi_AE \
  --out_postfix 0214
</pre>

## AE for cerebellar-brain-stem complex (sub)
<pre>
python scripts/train_3d_VAE.py \
  --csv data/processed_parts/sub_0206.csv \
  --spacing 1.5,1.5,1.5 \
  --batch 2 \
  --n_samples ALL \
  --workers 0 \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_epochs 100 \
  --ae_lr 1e-4 \
  --ae_latent_ch 8 \
  --ae_num_channels 64,128,256 \
  --outdir ckpts/AE \
  --out_prefix sub_AE \
  --out_postfix 0214
</pre>

</details>


<details>
<summary><strong>Train LDM UNets</strong></summary>

## LDM for whole brain

## LDM for hemispheres

## LDM for cerebellar-brain-stem complex (sub)


  
</details>
