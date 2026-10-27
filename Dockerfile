ARG VARIANT=cuda
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30 AS base-cuda
FROM python:3.10-slim AS base-cpu

FROM base-${VARIANT} AS final

ARG VARIANT=cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/hf_cache
ENV WANDB_MODE=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

VOLUME ["/app/data", "/app/src/data", "/app/runs", "/app/save", "/hf_cache"]

CMD ["python", "-c", "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')"]