ARG VARIANT=cuda
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30 AS base-cuda
FROM python:3.10-slim AS base-cpu

FROM base-${VARIANT} AS final

ARG VARIANT=cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/hf_cache
ENV WANDB_MODE=disabled
# stack.md patch 15 (story 1.6): torch 2.1 expandable-segments allocator — kills the
# reserved-but-unallocated fragmentation that caused OOM at ~95% VRAM (see the
# calibration runs pre-mb16 vs pre-mb16-x); numerics-neutral (allocation layout only).
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# stack.md patch 13 (story 1.6, decision D-1): stage targets run with
# --user $(id -u):$(id -g) so mounted outputs are owned by the host user;
# the container user needs write access to the baked code tree for data
# staging (src/data/redial, src/data/redial_gen) and loguru sinks (src/log).
RUN chmod -R a+rwX /app/src

VOLUME ["/app/data", "/app/src/data", "/app/runs", "/app/save", "/hf_cache"]

CMD ["python", "-c", "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')"]