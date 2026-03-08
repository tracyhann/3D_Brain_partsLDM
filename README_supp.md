# FOR LULIN

# Ablations

## Ablation 1: Soft part fusion LDM
<details>
<summary><strong>Details</strong></summary>

#### Estimated runtime: ~160 GPU hours on H100

### Example command if using single GPU:
```bash
python3 scripts/train_3d_brain_ldm_aux_taux_addition.py \
  --config configs/whole_brain_aux_taux_spacing1p5_ADDITION.json
```

#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)
#### Modify GPUs and Nodes requirements as needed

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

#### If needed, download ckpt for mask LDM here:
https://huggingface.co/nnuochen/3D_Brain_partsLDM/tree/main/whole_brain_mask_UNET_spacing1p5_ddp

#### Train context conditioning cLDM for part fusion
#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)
#### Modify GPUs and Nodes requirements as needed

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5_CONTEXT.json,MASTER_PORT=29620,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_context_ddp.slurm)
```
</details>


## Ablation 4: No shared AEs across hemispheres
<details>
<summary><strong>Details</strong></summary>

#### *Estimated* runtime: ~40 * 2 + 160 GPU hours on H100

- The actual runtime of this experiment has not been tested.
- Each hemi model takes < 100 hrs run time, followed by the fusion model.
- The following cmd runs all three steps consecutively (concurrently step 1, 2, followed by step 3).


### Download ckpts:
#### lhemi ldm; place under `ckpts/UNET/`
`https://huggingface.co/nnuochen/3D_Brain_partsLDM/tree/main/lhemi_UNET_spacing1p5_ddp`
#### rhemi ldm; place under `ckpts/UNET/`
`https://huggingface.co/nnuochen/3D_Brain_partsLDM/tree/main/rhemi_UNET_spacing1p5_ddp`
#### Fusion LDM
#### Launch 96-GPU DDP Training (12 nodes × 8 GPUs)
#### Modify GPUs and Nodes requirements as needed

```bash
cd <PATH_TO_PROJECT>/3D_Brain_partsLDM
sbatch -N 12 --ntasks-per-node=1 --gpus-per-node=8 \
  --export=ALL,PROJECT_ROOT=<PATH_TO_PROJECT>/3D_Brain_partsLDM,NPROC_PER_NODE=8,CONFIG=configs/whole_brain_aux_taux_spacing1p5_DiffHEMI.json,MASTER_PORT=29640,MAX_RESTARTS=20,MAX_REQUEUE=10 \
  train_ldm_aux_taux_DiffHEMI_ddp.slurm
```

</details>
