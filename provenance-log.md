# UniCRS Baseline Provenance Log

> CAP-5 artifact: every deviation from the official 2022 pin-set recorded for defensibility.

## Reference

Wang et al., "UniCRS: Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning", KDD 2022, arXiv:2206.09363

Official repo: https://github.com/RUCAIBox/UniCRS

## Environment

| Component | Official (2022) | Portable (self-run) | Notes |
|-----------|-----------------|---------------------|-------|
| OS | linux + CUDA 11.1 | Docker on Ubuntu 26.04 | Containerized |
| Python | 3.8.13 | 3.10 | Wheels unavailable for 3.8 |
| PyTorch | 1.8.1 + cu111 | 2.1.2 + cu121 | Driver 595.84 / CUDA 13.2 |
| transformers | 4.15.0 | 4.30.2 | API compatibility preserved |
| accelerate | 0.8.0 | 0.20.3 | Accelerate API updated |
| torch-geometric | 2.0.1 | 2.4.0 | PyG wheel index |
| torch-scatter | cu111 build | 2.1.2+pt21cu121 | PyG wheel index |
| torch-sparse | cu111 build | 0.6.18+pt21cu121 | PyG wheel index |

## Hardware

| Spec | Official | Self-run |
|------|----------|----------|
| GPU | 24 GB VRAM | RTX 3050 4 GB VRAM |
| RAM | - | 30 GiB |
| Disk | - | 312 GiB free |

## Batch Size Deviations (4 GB VRAM constraint)

| Stage | README batch | Self-run batch | grad_accum | Reason |
|-------|--------------|----------------|------------|--------|
| train_pre | 64 train, 128 eval | 16 train, 32 eval | 4 | 4 GB VRAM |
| train_conv | 8 train, 16 eval | 2 train, 32 eval | 4 | 4 GB VRAM |
| train_rec | 64 train, 64 eval | 16 train, 32 eval | 4 | 4 GB VRAM |
| infer_conv | 64 eval | 32 eval | - | 4 GB VRAM |

## Code Patches

| File | Line | Old | New | Reason |
|------|------|-----|-----|--------|
| Dockerfile | 2 | `FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base-cuda` | `FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30 AS base-cuda` | Story 1.3: 12.1.0 base is EOL; 12.1.1 pinned by digest for reproducible builds (torch cu121 wheels bundle their own CUDA/cuDNN) |
| Dockerfile | 12 | (absent) | `ENV WANDB_MODE=disabled` | Story 1.3: training scripts import wandb; runs must stay offline, never hit wandb servers |
| .dockerignore | new | (absent) | `data/`, `runs/`, `save/`, `hf_cache/`, `.git` | Story 1.3: build context must not bake datasets, outputs or git history into the image (AD-1: data is mounted, not copied) |
| Makefile | 13 | smoke imports torch, transformers, accelerate, torch_scatter | smoke imports torch, transformers, accelerate, torch_geometric, torch_scatter, torch_sparse, loguru, nltk, wandb, numpy, tqdm and prints versions + `cuda=True` | Story 1.3: smoke must cover every third-party module the training scripts use |
| requirements.lock | (whole file) | 23 top-level pins | full transitive freeze from the working container (+ loguru, nltk, wandb), `--extra-index-url` (cu121) and `--find-links` (pyg) kept, `+cu121` / `+pt21cu121` local versions kept | Story 1.3: locked, rebuildable environment |
| requirements-cpu.lock | (unchanged) | 23 top-level pins | kept minimal (not frozen) | Story 1.3 decision: CPU variant is unused for baseline stages; frozen only if a CPU run becomes load-bearing |

### Story 1.3 build record (2026-10-26, dogbench-gpu)

| Item | Value |
|------|-------|
| Base image | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30` |
| `requirements.lock` | full transitive freeze: 74 packages (incl. loguru 0.7.3, nltk 3.10.4, wandb 0.31.0); `--extra-index-url` (cu121) and `--find-links` (pyg) kept; `+cu121` / `+pt21cu121` local versions kept |
| Reproduction check | image rebuilt from the lock alone; `pip freeze` inside the rebuild is **byte-identical** to the captured environment (74/74) |
| Smoke (`make smoke`) | imports torch, transformers, accelerate, torch_geometric, torch_scatter, torch_sparse, loguru, nltk, wandb, numpy, tqdm; `torch=2.1.2+cu121 transformers=4.30.2 accelerate=0.20.3 torch_geometric=2.4.0 numpy=1.26.2 tqdm=4.66.1 cuda=True` |
| Image | `unicrs:baseline`, id `efdc04315ef5`, 13.3 GB unpacked |
| Fallback | `docker save \| gzip` → `~/NTU_Projects/unicrs-image-backup/unicrs-baseline-2026-10-26.tar.gz` (+ `.sha256`); laptop copy pending the dogbench→laptop backup key (Day-0 step 6) |
| wandb note | wandb 0.31.0: GitPython, sentry-sdk, prometheus-client, setproctitle are optional extras and not installed; `WANDB_MODE=disabled` needs none of them |
| Host note | `make smoke` / builds run with `--gpus all`; the NVIDIA MOTD banner in container stdout is harmless and must not be parsed by scripts |

## Wall-Clock Actuals

| Stage | Start | End | Duration | Notes |
|-------|-------|-----|----------|-------|
| Stage 2: Download | | | | |
| Stage 3: KG extract | | | | |
| Stage 4: Data prep | 2026-10-26 | 2026-10-26 | ~2 min | Story 1.4: pre variant from `process.py`, scratch run on dogbench-gpu |
| Stage 5: Prompt pre | | | | |
| Stage 6: Conv train | | | | |
| Stage 6: Inference | | | | |
| Stage 7: Rec train | | | | |
| **Total** | | | | |

## Data Artifacts

| Artifact | Path | Size | Checksum |
|----------|------|------|----------|
| DBpedia ttl | data/dbpedia/mappingbased-objects_lang=en.ttl.bz2 | ~400 MB | (sha256) |
| ReDial sub-KG | data/redial/subkg.json | | |
| Tokenized data | src/data/redial/*.pkl | | |
| ReDial mask variant | data/redial/*_data_processed.jsonl | 70,586 lines (56,355/6,253/7,978) | data/redial/MANIFEST.sha256 |
| ReDial pre variant | data/redial_pre/*_data_processed.jsonl | 139,557 lines (111,458/12,395/15,704) | data/redial_pre/MANIFEST.sha256 |

### Story 1.4 run record (2026-10-26, dogbench-gpu)

| Item | Value |
|------|-------|
| Command | `uv run --isolated --no-project --python 3.10 --with tqdm==4.66.1 python process.py` |
| Python | 3.10.22 (uv-managed) |
| tqdm | 4.66.1 |
| Input | scratch copy of `data/redial/` (`*_data_dbpedia.jsonl`, `entity2id.json`) |
| Output lines | train 111,458 / valid 12,395 / test 15,704 (total 139,557) |
| `item_ids.json` | byte-identical to `data/redial/item_ids.json` (`cmp` clean); 6,281 items |
| Manifests | 17 tracked files in `data/redial/`, 3 files in `data/redial_pre/`, both `LC_ALL=C` sorted, `sha256sum -c` clean |
| Post-merge check | Fresh clone `3a4c8b7` on dogbench-gpu: `sha256sum -c` clean in both folders (2026-10-26) |

## Checkpoints

| Checkpoint | Path | Size |
|------------|------|------|
| Prompt pre | runs/unicrs-prompt-pre/ | |
| Conv | runs/unicrs-conv/ | |
| Rec | runs/unicrs-rec/ | |

## Generated Files

| File | Path |
|------|------|
| Train jsonl | save/redial/dialogpt_train.jsonl |
| Valid jsonl | save/redial/dialogpt_valid.jsonl |
| Test jsonl | save/redial/dialogpt_test.jsonl |