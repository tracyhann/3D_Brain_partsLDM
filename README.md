
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

<pre>
hf datasets info tracyhan816/3D_brain_partLDM_data
{
  "id": "tracyhan816/3D_brain_partLDM_data",
  "author": "tracyhan816",
  ...
  "private": true,
}
</pre>

- You may use the `data.tar.gz` to replace the `/data` dir if conflicts arise. 
- Place `/data` dir under project dir: `./3D_Brain_partsLDM/data`.

</details>



## Ckpts
<details>
<summary><strong>Details</strong></summary>

- Download the entire `/ckpts` dir from huggingface (about 10G): https://huggingface.co/tracyhan816/3D_Brain_partsLDM
- Place `/ckpts` dir under project dir: `./3D_Brain_partsLDM/ckpts`.
- You may use the huggingface `/ckpts` to overwrite the existing `/ckpts` dir in the project dir if conflicts arise.

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

#### *Estimated* runtime: ~100 GPU hours on H100

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

#### *Estimated* runtime: ~100 + 160 GPU hours on H100 (2 steps)

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

#### *Estimated* runtime: ~100 * 2 + 160 GPU hours on H100

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
