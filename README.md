
`cd <PROJECT_DIR>`

# To start
## Environment
<details>
<summary><strong>Details</strong></summary>

### Docker
https://hub.docker.com/r/h8w108/3dbrain

<pre>
docker run --gpus all -it   -v "$PWD":/workspace   -w /workspace   h8w108/3dbrain:parts_ldm_20251013  bash
</pre>

#### If encountering conda init issue
Load conda’s bash hook for this shell

<pre>
source /opt/conda/etc/profile.d/conda.sh  \
  || eval "$(/opt/conda/bin/conda shell.bash hook)"
</pre>

<pre>
conda activate monai
</pre>



### yaml

Alternatively, you can build a conda environment using this yaml file:

https://github.com/lulinliu/MineLongTail/blob/main/monaifull.yml

</details>

</details>


## Data
<details>
<summary><strong>Details</strong></summary>

- Download the entire `/data` dir from huggingface (about 37G):

```bash
cd <PROJECT_PATH>/3D_brain_partLDM
```

```bash
hf auth login
```
paste `$HF_TOKEN`

```bash
hf download tracyhan816/3D_brain_partLDM_data --repo-type dataset 
```

- Dataset info

<pre>
hf datasets info tracyhan816/3D_brain_partLDM_data
{
  "id": "tracyhan816/3D_brain_partLDM_data",
  "author": "tracyhan816",
  ...
  "private": true,
}
</pre>

- Inside project dir, release `data.tar.gz`

```bash
mv data data_ARCHIVE_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
mkdir -p data
tar -xzf data.tar.gz -C data --strip-components=1
```

</details>



## Ckpts
<details>
<summary><strong>Details</strong></summary>

- Inside project dir:
- Download the entire `/ckpts` dir from huggingface (about 10G): https://huggingface.co/tracyhan816/3D_Brain_partsLDM
- This becomes `/ckpts` dir under project dir: `./3D_Brain_partsLDM/ckpts`.

```bash
mv ckpts ckpts_ARCHIVE_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

# download the ckpts folder from the model repo into ./ckpts
hf download tracyhan816/3D_Brain_partsLDM \
  --repo-type model \
  --include "ckpts/**" \
  --local-dir . 
```

</details>


#### After the above steps, the project dir looks like this:

<pre>
.
├── ckpts
├── configs
├── data
├── data_prep
├── Dockerfile
├── environment.yml
├── README.md
├── scripts
├── train_*.sh
└── train_*.slurm
</pre>

#### When running the following cmds, replace `<PATH_TO_PROJECT>` 
# Ours

## Compositional fusion LDM for whole brain

<details>
<summary><strong>Details</strong></summary>

#### Estimated runtime: ~160 GPU hours on H100

### Single GPU:
```bash
python3 scripts/train_3d_brain_ldm_aux_taux.py   --config configs/whole_brain_aux_taux_spacing1p5.json
```

### Slurm:
##### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_ddp.slurm
```

</details>

# Baselines
## Baseline 3: Segmentation-mask-guided LDM (Med-DDPM style)
<details>
<summary><strong>Details</strong></summary>

#### *Estimated* runtime: ~80 GPU hours on H100

- The actual runtime of this experiment has not been tested.

#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_segmLDM_spacing1p5.json,MASTER_PORT=29630,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_segm_ddp.slurm
```

</details>




# Ablations
## Ablation 1: Soft part fusion LDM
<details>
<summary><strong>Details</strong></summary>

#### Estimated runtime: ~160 GPU hours on H100

### Single GPU:
```bash
python3 scripts/train_3d_brain_ldm_aux_taux_addition.py \
  --config configs/whole_brain_aux_taux_spacing1p5_ADDITION.json
```
### Slurm:
#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/configs/whole_brain_aux_taux_spacing1p5_ADDITION.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_addition_ddp.slurm
```


</details>


## Ablation 2: Context conditioning cLDM for fusion of parts
<details>
<summary><strong>Details</strong></summary>

#### *Estimated* runtime: ~50 + 160 GPU hours on H100 (2 steps)

- The actual runtime of this experiment has not been tested.
- The slurm job below will run 2 steps consecutively.

### Slurm:
#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM

job1=$(sbatch --parsable -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_maskLDM_spacing1p5.json,MASTER_PORT=29610,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_mask_ddp.slurm)

job2=$(sbatch --parsable --dependency=afterok:${job1} -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5_CONTEXT.json,MASTER_PORT=29620,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_context_ddp.slurm)

echo "step1=${job1}"
echo "step2=${job2} (afterok:${job1})"
```

</details>


## Ablation 3: No Aux
<details>
<summary><strong>Details</strong></summary>

#### Estimated runtime: ~160 GPU hours on H100

### Single GPU:
```bash
python3 scripts/train_3d_brain_ldm_aux_taux.py \
  --config configs/whole_brain_aux_taux_spacing1p5_NO_AUX.json
```
### Slurm:
#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5_NO_AUX.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_ddp.slurm
```


</details>



## Ablation 4: No shared AEs across hemispheres
<details>
<summary><strong>Details</strong></summary>

#### *Estimated* runtime: ~40 * 2 + 160 GPU hours on H100

- The actual runtime of this experiment has not been tested.
- Each hemi model takes < 100 hrs run time, followed by the fusion model.
- The following cmd runs all three steps consecutively (concurrently step 1, 2, followed by step 3).


### Full job (3 steps):
#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
NODES=12 GPUS_PER_NODE=8 NPROC_PER_NODE=8 MAX_RESTARTS=20 MAX_REQUEUE=10 ./submit_diffhemi_pipeline.sh
```

### To run jobs independently:

#### lhemi
```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/lhemi_AE_spacing1p5.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ae_lhemi.slurm
```
```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/lhemi_LDM_spacing1p5.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_lhemi.slurm
```


#### rhemi
```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/rhemi_AE_spacing1p5.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ae_rhemi.slurm
```
```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/rhemi_LDM_spacing1p5.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_rhemi.slurm
```

#### Fusion LDM

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5_DiffHEMI.json,MASTER_PORT=29640,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_DiffHEMI_ddp.slurm
```


</details>


# FOR LULIN

# Inference

## Ours

```bash
python scripts/infer_3d_brain_ldm_aux_taux.py   --config configs/infer/infer_whole_brain_aux_taux_spacing1p5.json   --erode_r 2
```

## Ablation: No Aux

```bash
python scripts/infer_3d_brain_ldm_aux_taux.py   --config configs/infer/infer_whole_brain_aux_taux_spacing1p5_NO_AUX.json   --erode_r 2
```

## Ablation: No INj
```bash
python3 scripts/infer_3d_brain_ldm_steps.py \
  --csv data/processed_parts/whole_brain_0206.csv \
  --data_split_json_path data/patient_splits_image_ids_75_10_15.json \
  --ae_ckpt ckpts/AE/whole_brain_AE_spacing1p5/AE_best.pt \
  --ldm_ckpt ckpts/UNET/whole_brain_aux_taux_UNET_spacing1p5/UNET_last.pt \
  --workers 0 \
  --spacing 1.5,1.5,1.5 \
  --size 128,128,128 \
  --conditions age,sex,group,imageID \
  --ddim_steps 50 \
  --num_samples -1 \
  --outdir samples \
  --outdir samples \
  --out_prefix whole_brain_aux_taux_NO_INJ \
  --out_postfix UNET_spacing1p5

```




## Segm (IGNORE)

```bash
python3 scripts/infer_3d_brain_ldm_segm.py \
  --config configs/infer/infer_whole_brain_segmLDM_spacing1p5.json
```


## SEGMENTATION

### Example cmd structure
#### REPLACE with your data volume dir. 
`-v /PATH/TO/PROJ`

```bash
docker run -it --rm  -v /home/ttt/Desktop/wenyan/3D_Brain_partsLDM/samples/whole_brain_aux_taux_UNET_spacing1p5:/data   pwesp/synthseg:py38
```

```bash
python /workspace/SynthSeg/scripts/commands/SynthSeg_predict.py \
--i /data \
--o /data/segm \
--v1 --threads 16 \
--fast --qc /data/qc --vol /data/vol
```

##### Also download these samples and place them under `/samples`
https://huggingface.co/datasets/tracyhan816/3D_Brain_partsLDM/tree/main

#### REAL 
```bash
docker run -it --rm  -v /PATH/TO/PROJ/3D_Brain_partsLDM/samples/real_test_samples:/data   pwesp/synthseg:py38
```
```bash
python /workspace/SynthSeg/scripts/commands/SynthSeg_predict.py \
--i /data \
--o /data/segm \
--v1 --threads 16 \
--fast --qc /data/qc --vol /data/vol
```



### DO THE ABOVE STEP FOR EACH FOLDER OF SAMPLES 

#### OURS
```bash
docker run -it --rm  -v /PATH/TO/PROJ/3D_Brain_partsLDM/samples/whole_brain_aux_taux_UNET_spacing1p5:/data   pwesp/synthseg:py38
```

```bash
python /workspace/SynthSeg/scripts/commands/SynthSeg_predict.py \
--i /data \
--o /data/segm \
--v1 --threads 16 \
--fast --qc /data/qc --vol /data/vol
```


#### ABLATION: NO AUX
```bash
docker run -it --rm  -v /PATH/TO/PROJ/3D_Brain_partsLDM/samples/whole_brain_aux_taux_NO_AUX_UNET_spacing1p5:/data   pwesp/synthseg:py38
```

```bash
python /workspace/SynthSeg/scripts/commands/SynthSeg_predict.py \
--i /data \
--o /data/segm \
--v1 --threads 16 \
--fast --qc /data/qc --vol /data/vol
```

#### ABLATION: NO INJECTION
```bash
docker run -it --rm  -v /PATH/TO/PROJ/3D_Brain_partsLDM/samples/whole_brain_aux_taux_NO_INJ_UNET_spacing1p5:/data   pwesp/synthseg:py38
```

```bash
python /workspace/SynthSeg/scripts/commands/SynthSeg_predict.py \
--i /data \
--o /data/segm \
--v1 --threads 16 \
--fast --qc /data/qc --vol /data/vol
```


#### BASELINE 1: LDM 
```bash
docker run -it --rm  -v /PATH/TO/PROJ/3D_Brain_partsLDM/samples/whole_brain_aux_taux_NO_AUX_UNET_spacing1p5:/data   pwesp/synthseg:py38
```

```bash
python /workspace/SynthSeg/scripts/commands/SynthSeg_predict.py \
--i /data \
--o /data/segm \
--v1 --threads 16 \
--fast --qc /data/qc --vol /data/vol
```


#### BASELINE 2: Segmentation-conditioned LDM
```bash
docker run -it --rm  -v /PATH/TO/PROJ/3D_Brain_partsLDM/samples/whole_brain_mask_UNET_spacing1p5_ddp:/data   pwesp/synthseg:py38
```

```bash
python /workspace/SynthSeg/scripts/commands/SynthSeg_predict.py \
--i /data \
--o /data/segm \
--v1 --threads 16 \
--fast --qc /data/qc --vol /data/vol
```


# Grab files
```bash
python3 keep_worst20.py
```
