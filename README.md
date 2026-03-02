
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
├── train_ldm_aux_taux_addition_ddp.sh
├── train_ldm_aux_taux_addition_ddp.slurm
├── train_ldm_aux_taux_ddp.sh
├── train_ldm_aux_taux_ddp.slurm
├── train_ldm_hemi.sh
├── train_ldm_hemi.slurm
├── train_ldm_sub.sh
├── train_ldm_sub.slurm
├── train_ldm_wholebrain.sh
└── train_ldm_wholebrain.slurm
</pre>


# Ours

## Compositional fusion LDM for whole brain

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

##### Launch 64-GPU DDP Training (8 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 8 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_ddp.slurm
```

# Ablations
## Ablation 3: Soft part fusion LDM
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

#### Launch 64-GPU DDP Training (8 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 8 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/configs/whole_brain_aux_taux_spacing1p5_ADDITION.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_addition_ddp.slurm
```

</details>

## Ablation 4: No Aux
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

#### Launch 64-GPU DDP Training (8 nodes × 8 GPUs)

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 8 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5_NO_AUX.json,MASTER_PORT=29600,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_ddp.slurm
```

</details>
