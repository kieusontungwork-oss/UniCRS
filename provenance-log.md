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
| (fill as patches are applied) | | | | |

## Wall-Clock Actuals

| Stage | Start | End | Duration | Notes |
|-------|-------|-----|----------|-------|
| Stage 2: Download | | | | |
| Stage 3: KG extract | | | | |
| Stage 4: Data prep | | | | |
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