# Track 04 — reproducible environment (CUDA 12.6, Python 3.10, torch 2.7.1).
#   docker build -t track04 .                         # downloads the checkpoints during the build
#   docker build --build-arg WEIGHTS=0 -t track04 .   # skip; mount a weights/ folder at run time instead
#   docker run --gpus all -v /path/to/videos:/data track04 /data/P01_11.MP4 /data/P01_11.json
# Needs the NVIDIA Container Toolkit on the host. A 4 GB GPU is enough.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python-is-python3 build-essential ffmpeg git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
COPY . /app
RUN bash setup.sh
ARG WEIGHTS=1
RUN if [ "$WEIGHTS" = "1" ]; then bash weights/download_weights.sh; fi
ENV HF_HUB_OFFLINE=1 TEAM_ID=K99
ENTRYPOINT ["/app/run.sh"]
