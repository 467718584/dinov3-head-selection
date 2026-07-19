# P-001 DINOv3 Head Selection — Reproducible Environment
# v28.4.2 camera-ready (2026-07-26)
#
# Build:    docker build -t dinov3-head-selection:28.4.2 .
# Run:      docker run --gpus all -v $(pwd):/work -it dinov3-head-selection:28.4.2
# Test:     docker run --gpus all dinov3-head-selection:28.4.2 pytest tests/

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

LABEL maintainer="Zhang Zhenyi <Z467718583@126.com>" \
      version="28.4.2" \
      description="DINOv3 Head Selection for Few-Shot Industrial Defect Detection (P-001)"

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        git \
        wget \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Working directory
WORKDIR /work

# Python deps (cached layer)
COPY requirements.txt /work/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# DINOv3 Meta-native checkpoints (downloaded at build time)
# NOTE: These are the Meta-released checkpoints, NOT HuggingFace gated versions.
# Total size ~3.4 GB for 4 backbones (ViT-S/16, ViT-B/16, ViT-L/16, ConvNeXt-T)
RUN mkdir -p /work/04_data/models/dinov3-jay && \
    cd /work/04_data/models/dinov3-jay && \
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
        https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16_pretrain_lvd1689m-101b2c18.pth \
        https://dl.fbaipublicfiles.com/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth

# Source code
COPY src/ /work/src/
COPY experiments/ /work/experiments/
COPY data/ /work/data/
COPY tests/ /work/tests/

# Run entrypoint
ENTRYPOINT ["python", "-m", "src.run_pemb_ablation_v3"]
CMD ["--help"]