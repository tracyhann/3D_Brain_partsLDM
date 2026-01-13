# Small + stable: CUDA runtime, not devel
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- System deps (lean) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates curl git tini \
 && rm -rf /var/lib/apt/lists/*

# --- Miniconda ---
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Use a current py311 Miniconda (py310 also fine if you prefer)
RUN wget -qO /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
 && rm -f /tmp/miniconda.sh \
 && conda clean -afy

# Faster solver
RUN conda install -y -n base conda-libmamba-solver \
 && conda config --system --set solver libmamba \
 && conda clean -afy

# --- Create env + install PyTorch (CUDA 12.1) + MONAI ---
ARG ENV_NAME=monai
RUN conda create -y -n $ENV_NAME python=3.11 \
 && conda run -n $ENV_NAME conda install -y -c pytorch -c nvidia \
      pytorch torchvision torchaudio pytorch-cuda=12.1 \
 && conda run -n $ENV_NAME pip install --no-cache-dir "monai[all]" \
      numpy pandas matplotlib monai-generative\
 && conda clean -afy

# Make the env default
ENV PATH=$CONDA_DIR/envs/$ENV_NAME/bin:$PATH

# --- Sanity check ---
RUN python - <<'PY'
import torch, monai
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime  :", torch.version.cuda)
print("PyTorch       :", torch.__version__)
print("MONAI         :", monai.__version__)
PY
