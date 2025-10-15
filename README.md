Dockerfile contains the conda environment 'monai' that can run this project. Mostly, the environment includes MONAI and MONAI-generative tools.

pip install cu128 if running on sm120 GPU

`cd <PROJECT_DIR>`

sudo docker run --gpus all -it \
  -v "$PWD":/workspace \
  -w /workspace \
  3dbrain:latest bash

#  If encountering conda init issue
## load conda’s bash hook for this shell
source /opt/conda/etc/profile.d/conda.sh  \
  || eval "$(/opt/conda/bin/conda shell.bash hook)"

`conda activate monai`


## finetune from pre-trained weights
python scripts/train_3d_ldm_from_parts.py \
  --csv caud_puta_data.csv \
  --stage both \
  --ae_finetune_ckpt models/autoencoder.pt --ae_decoder_only \
  --ldm_use_cond \
  --batch 1 --workers 0



## To avoid OOM issue

python scripts/train_3d_ldm_from_parts.py \
  --csv caud_puta_data.csv \
  --stage both \
  --ae_finetune_ckpt models/autoencoder.pt --ae_decoder_only \
  --ldm_use_cond \
  --batch 1 --workers 0 \
  --spacing 2,2,2 --size 128,160,128
