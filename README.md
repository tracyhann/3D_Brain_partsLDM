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
