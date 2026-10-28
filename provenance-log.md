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
| PyTorch | 1.8.1 + cu111 | 2.1.2 + cu121 | Driver 595.99.02 / CUDA 13.2 (dogbench-gpu, updated 2026-10-27 from 595.91.07) |
| transformers | 4.15.0 | 4.30.2 | API compatibility preserved |
| accelerate | 0.8.0 | 0.20.3 | Accelerate API updated |
| torch-geometric | 2.0.1 | 2.4.0 | PyG wheel index |
| torch-scatter | cu111 build | 2.1.2+pt21cu121 | PyG wheel index |
| torch-sparse | cu111 build | 0.6.18+pt21cu121 | PyG wheel index |

## Hardware

| Spec | Official | Self-run |
|------|----------|----------|
| GPU | 24 GB VRAM | dogbench-gpu: RTX 4060 8 GB VRAM (stages 5–8; laptop RTX 3050 4 GB ran stages 0–4) |
| Host kernel | - | Ubuntu 26.04, kernel 7.0.0-38-generic (story 1.6 reboot, 2026-10-27) |
| RAM | - | 30 GiB |
| Disk | - | 293 GiB free at story 1.3 |

## Batch Size Deviations (8 GB VRAM constraint; chosen by story 1.6 calibration, 2026-10-27)

| Stage | README batch | Self-run batch | grad_accum | Reason |
|-------|--------------|----------------|------------|--------|
| train_pre | 64 train, 128 eval | 16 train, 64 eval | 4 | 8 GB VRAM: 64×1 and 32×2 OOM on the first step; 16×4 real peak 7,377 MiB, absolute worst-case (full-length padding) 7,757 MiB fits |
| train_conv | 8 train, 16 eval | 8 train, 16 eval (= README) | 1 | 8×1 fits with margin (peak 6,449 MiB) and is 21% faster than 4×2; eval 32 OOMs (single 1.47 GiB block) |
| train_rec | 64 train, 64 eval | 16 train, 64 eval | 4 | same memory profile as train_pre (identical 200+200 padding caps) |
| infer_conv | 64 eval | 16 eval | - | 32 OOMs; 16 gives ~0.85 s/batch with the KV patch ≈ 1.1 h for all 70,586 samples |

Effective batches stay at README values (pre 64 / conv 8 / rec 64) via gradient accumulation; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Dockerfile, patch 15) is required for these numbers. Prior 4 GB plan values (pre 16×4 / conv 2×4 / rec 16×4, eval 32) were never executed — the laptop ran stages 0–4 only.

## Code Patches

| File | Line | Old | New | Reason |
|------|------|-----|-----|--------|
| Dockerfile | 2 | `FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base-cuda` | `FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30 AS base-cuda` | Story 1.3: 12.1.0 base is EOL; 12.1.1 pinned by digest for reproducible builds (torch cu121 wheels bundle their own CUDA/cuDNN) |
| Dockerfile | 12 | (absent) | `ENV WANDB_MODE=disabled` | Story 1.3: training scripts import wandb; runs must stay offline, never hit wandb servers |
| .dockerignore | new | (absent) | `data/`, `runs/`, `save/`, `hf_cache/`, `.git` | Story 1.3: build context must not bake datasets, outputs or git history into the image (AD-1: data is mounted, not copied) |
| Makefile | 13 | smoke imports torch, transformers, accelerate, torch_scatter | smoke imports torch, transformers, accelerate, torch_geometric, torch_scatter, torch_sparse, loguru, nltk, wandb, numpy, tqdm and prints versions + `cuda=True` | Story 1.3: smoke must cover every third-party module the training scripts use |
| requirements.lock | (whole file) | 23 top-level pins | full transitive freeze from the working container (+ loguru, nltk, wandb), `--extra-index-url` (cu121) and `--find-links` (pyg) kept, `+cu121` / `+pt21cu121` local versions kept | Story 1.3: locked, rebuildable environment |
| requirements-cpu.lock | (unchanged) | 23 top-level pins | kept minimal (not frozen) | Story 1.3 decision: CPU variant is unused for baseline stages; frozen only if a CPU run becomes load-bearing |
| Makefile | 36–52 (stage 5–7 targets) | commands run from `/app`; scripts live in `/app/src` | `cd src &&` before every `accelerate launch` | Story 1.5 (stack.md patch 1) |
| Makefile | 36–52 (stage 5–7 targets) | `-v $(DATA):/app/data` read-write, nothing staged — the code reads `src/data/<dataset>` relative to cwd | `-v $(DATA):/app/data:ro` plus the target copies its variant into the container's `src/data/redial/`; stage 5 then overwrites the three jsonl with `data/redial_pre/`'s | Story 1.5 (stack.md patch 2; variant directories created by story 1.4) |
| Makefile | 45, 51 (stage6-infer, stage7) | `-v $(SAVE):/app/save` | `-v $(SAVE):/app/src/save` — `infer_conv.py:163` writes `save/redial` relative to `/app/src` | Story 1.5 (stack.md patch 3) |
| Makefile | 51–52 (stage7) | `python data/redial_gen/merge.py --gen_file_prefix dialogpt` run from `/app` | `(cd data/redial_gen && python merge.py --gen_file_prefix unicrs-conv)`, then `mkdir -p save/redial_gen && cp data/redial_gen/*_data_processed.jsonl save/redial_gen/` | Story 1.5 (stack.md patch 4) + prefix change: `infer_conv.py:165` names outputs after `--prompt_encoder`'s parent folder, so stage 6 now writes `save/redial/unicrs-conv_{train,valid,test}.jsonl` and merge must consume that prefix |
| `src/data/redial_gen/merge.py` | 8 | `dataset = 'inspired'` | `dataset = 'redial'` | Story 1.5 (stack.md patch 5): upstream hard-codes the wrong dataset; merge would read `data/inspired/` and `save/inspired/` |
| `src/model_gpt2.py` | 585–588 | `def prepare_inputs_for_generation(self, input_ids, past=None, prompt_embeds=None, **kwargs)` — transformers 4.30.2 passes the cache as `past_key_values`, so `past` stayed empty, the KV cache was unused and generation ran ~10× slower | first line of the body: `past = kwargs.pop("past_key_values", past)` (the return dict already forwards `past_key_values`) | Story 1.5 (stack.md patch 6); **CONFIRMED by story 1.6 (2026-10-27)**: greedy outputs on the first 64 test dialogues byte-identical with and without the patch (both sha256 `3b53499e811e257508aacec2092018ded3b75f6d1910e80548031c748a0bc64b`, `runs/calib/kv-{on,off}/save/redial/`); generation ~8× faster with it (28 s → 3.3 s per 4 batches). Kept |
| Makefile | 36–52 (stage 5–7 targets) | `--prompt_encoder ../runs/unicrs-prompt-pre` (stage 6 infer: `../runs/unicrs-conv`) — `KGPrompt.load` (model_prompt.py:140) reads `<dir>/model.pt`, which exists only under `best/` and `final/` | `--prompt_encoder ../runs/unicrs-prompt-pre/best` (stage 6 infer: `../runs/unicrs-conv/best`, no trailing slash so outputs keep the `unicrs-conv_` prefix) | Story 1.5 (stack.md patch 10) |
| Makefile | 36–52 (stage 5–7 targets) | targets overwrite existing outputs; log starts with docker output | host-side guard aborts a target whose output already exists (`runs/unicrs-prompt-pre`, `runs/unicrs-conv`, `save/redial/unicrs-conv_*.jsonl`, `runs/unicrs-rec`, `save/redial_gen`); stage 7 also requires the stage 6 outputs to exist; every target echoes `git rev-parse HEAD` as the first log line | Story 1.5 (stack.md patch 12, Makefile part; output isolation AD-4, clean-checkout rule) |
| Makefile | 58–60 (`clean`) | `docker rmi` **plus** `rm -rf $(RUNS)/* $(SAVE)/*` | `docker rmi` only | Story 1.5 (stack.md patch 12): `make clean` must never delete outputs |
| Dockerfile | after `COPY . .` | (absent) | `RUN chmod -R a+rwX /app/src` | Story 1.6 (stack.md patch 13, decision D-1, owner approved 2026-10-27): stage targets run `--user $(id -u):$(id -g)`, so the container user needs write access to the baked tree for data staging (`src/data/redial`, `src/data/redial_gen`) and loguru sinks (`src/log`) |
| Makefile | all `docker run` lines (smoke, stage2–stage7) | containers run as root; every file they write into mounted host dirs is root-owned (the laptop's root-owned `hf_cache/` is prior evidence) | `UID_ARGS = --user $(id -u):$(id -g) -e HOME=/tmp` added to every `docker run` | Story 1.6 (stack.md patch 13, decision D-1): outputs under `runs/` and `save/` are owned by the invoking user on the shared server; verified by a uid-1000 dry-run (staging `cp`, `log/` creation, `data/redial_gen` mkdir) |
| `src/train_pre.py`, `src/train_conv.py`, `src/train_rec.py`, `src/infer_conv.py` | 76 / 80 / 80 / 79 | `accelerator = Accelerator(device_placement=False, fp16=args.fp16)` | `accelerator = Accelerator(device_placement=False, mixed_precision=('fp16' if args.fp16 else 'no'))` | Story 1.6 (stack.md patch 14): accelerate 0.20.3 removed the `fp16` kwarg — every training/inference launch died at startup (caught by the first calibration run, 2026-10-27); `mixed_precision='no'` keeps the fp32 recipe identical and `--fp16` is never passed; the `accelerator.use_fp16` reads elsewhere are kept because 0.20.3 still ships the attribute |
| Dockerfile | ENV block | (absent) | `ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Story 1.6 (stack.md patch 15): expandable-segments allocator; calibration showed OOMs driven by reserved-but-unallocated fragmentation (1.40 GiB in the pre-mb16 eval-128 OOM); with it, 16×4 fits with the worst-case `--debug` full-padding probe at 7,757 MiB; allocation-layout change only, math identical |

### Story 1.5 record (2026-10-27, dogbench-gpu)

| Item | Value |
|------|-------|
| Scope | stack.md patches 1, 2, 3, 4, 5, 6, 10 + Makefile parts of 11/12 (staging, abort guards, sha echo, image-only `clean`); patches 7–9 were story 1.3, variant directories were story 1.4 |
| Verification | `make -n` inspection of all four targets; per-target container wiring dry-runs (staging + path assertions, no training); `py_compile` on both patched files; `merge.py` fixture run against a synthetic `redial/` + `save/redial/` tree; guard tests (existing output → abort before docker starts; `make clean` leaves `runs/` and `save/` untouched) |
| Image rebuilt 2026-10-27 | `make build` from this branch (pip layers cached): `unicrs:baseline` `efdc04315ef5` → `bfa428184794` so the baked `/app/src` carries patches 5 and 6; smoke re-passed (`cuda=True`, same versions). **Stage runs before story 1.6 must use the rebuilt image** — the stage targets execute the image's code copy, not the host working tree |
| Effective batches | unchanged by this story: pre 16×4 / conv 2×4 / rec 16×4 = README effective 64 / 8 / 64; story 1.6 recalibrates micro-batches for the 8 GB card and commits the result |
| Expected stage outputs | stage 5 → `runs/unicrs-prompt-pre/best/model.pt`; stage 6 → `runs/unicrs-conv/best/model.pt` + `save/redial/unicrs-conv_{train,valid,test}.jsonl` (70,586 lines total); stage 7 → `runs/unicrs-rec/best/model.pt` + merged jsonl under `save/redial_gen/` |
| KV patch status | applied provisionally; story 1.6 must confirm identical greedy outputs on 64 test dialogues with and without it, else revert |

### Story 1.6 record (2026-10-27, dogbench-gpu)

| Item | Value |
|------|-------|
| Machine | dogbench-gpu: RTX 4060 8 GB (8,188 MiB total), driver 595.99.02 / CUDA 13.2, kernel 7.0.0-38-generic, image `138b2f8823ff` (rebuilt twice during this story: patch 13, patch 14) |
| Reboot check (SPEC Open Question) | unattended-upgrades enabled but `Automatic-Reboot` commented out (default **false**) → no unattended reboots. `/var/run/reboot-required` was present 2026-10-27 (kernel `linux-image-7.0.0-38-generic`, `linux-base`); owner rebooted. First boot on the new kernel lost the GPU: this host uses per-kernel prebuilt modules and `linux-modules-nvidia-595-open-7.0.0-38-generic` had not been pulled in; fixed by installing it (+ second reboot). The accompanying `apt full-upgrade` moved the driver 595.91.07 → **595.99.02**. Post-fix verify: `nvidia-smi` OK; container `cuda=True`. Lesson for future kernel updates: check the matching `linux-modules-nvidia-*` package before rebooting |
| HF cache | warmed before calibration: 816 MB (`microsoft/DialoGPT-small`, `roberta-base`; tokenizer + weights), `ryan`-owned via the patch-13 `--user` pattern |
| Data manifests | re-verified 2026-10-27 before any calibration run: `data/redial` 17/17 OK, `data/redial_pre` 3/3 OK |
| Calibration runs | `--max_train_steps 50`, identical recipe flags to the stage targets, data staged the same way, VRAM sampled at 500 ms (`vram.csv` per label), logs under `runs/logs/calib-<label>.log` (see table below). **train_rec substitution:** calibrated with `--dataset redial` (mask variant) instead of `redial_gen` — merged data needs stage-6 generated files that don't exist yet; without `--use_resp` the rec batches are identical, so timing/memory transfer |

#### Calibration runs (CAP-6)

| Label | Script | micro×accum (eff.) | eval bs | Result | s/step | Peak VRAM | Notes |
|-------|--------|--------------------|---------|--------|--------|-----------|-------|
| pre-mb64 | train_pre | 64×1 (64) | 128 | **OOM** in train forward | — | 7,789 MiB | attempt 1; OOM on the first training step; log `calib-pre-mb64.log` |
| pre-mb32 | train_pre | 32×2 (64) | 128 | **OOM** in train forward | — | 7,805 MiB | OOM again on the first training step — ~7.3 GiB allocated by PyTorch at the first forward (fixed cost: models + full-graph R-GCN + prompt-extended attention); log `calib-pre-mb32.log` |
| pre-mb16 | train_pre | 16×4 (64) | 128 | **train OK, eval 128 OOM** | 2.77 s/step | 7,683 MiB | all 50 train steps completed (2:16, loss 14.57); OOM in the full valid eval forward at eval bs 128 (5.88 GiB allocated + 1.40 GiB reserved/fragmented); log `calib-pre-mb16.log` |
| pre-mb16-e64 | train_pre | 16×4 (64) | 64 | **complete** | 2.78 s/step | 7,621 MiB (train) | full run OK (exit 0). Phase peaks (vram.csv, timestamped): train 7,621 / valid eval @64 5,529 / test eval @64 5,831 MiB. Timings: 50 steps 2:16; valid 12,395 @64 = 2:41; test 15,704 @64 = 3:23. Train peak = 97.7% of usable → margin too thin for the full 5-epoch run (per-epoch shuffle) → step down to micro 8 |
| pre-mb8 | train_pre | 8×8 (64) | 64 | **complete** | 3.99 s/step | 7,083 MiB | safe but 44% slower per step than 16×4 (6,933 train / 7,083 eval peak); fixed floor ≈ 6.5 GiB (params+grads+Adam) dominates → halving micro-batch buys little memory for a big throughput loss |
| pre-mb16-x | train_pre | 16×4 (64) | 64 | **complete** | 2.75 s/step | 7,377 MiB | 16×4 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; phase peaks train 7,375 / evals 7,377; best throughput, headroom 5.6% |
| pre-mb16-x-wc | train_pre | 16×4 (64) | 64 | **complete (worst case)** | — | 7,757 MiB | `--debug` pads every batch to `max_length` → absolute memory ceiling of this config; it **fits** (99.4% of usable) ⇒ no real batch can OOM 16×4+X; real-run peak remains 7,377 (94.5%) |

**Stage 5 chosen config (story 1.6):** micro 16 × accum 4, eval 64, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; measured 2.75 s/step → projection 8,710 steps ≈ 6.7 h + 5×valid (2:41) + 1×test (3:23) ≈ **7.0 h**. Fallback (owner may prefer at go/no-go): 8×8 = 3.99 s/step, peak 7,083 MiB (90.8%), ≈ 10 h.

<!-- story-1.6-calibration-results: appended as runs complete -->

| conv-mb4 | train_conv | 4×2 (8) | 16 | **complete** | 0.58 s/step | 6,353 MiB | 50 steps ≈ 29 s (conv effective batch 8 is small); phase peaks: train 4,691 / valid eval 5,253 / test eval 6,353; evals ≈ 1.16 s/batch @16 (valid 6,253 = 7:37, test 7,978 = 9:37); projection ≈ 11.3 h train + ~1.4 h evals ≈ 12.8 h |
| conv-mb8-e32 | train_conv | 8×1 (8) | 32 | **train OK, eval 32 OOM** | 0.46 s/step | 7,667 MiB | all 50 train steps at micro 8 completed (23 s — 21% faster than 4×2); eval forward at bs 32 requested a single 1.47 GiB block → OOM; eval 16 is the ceiling |
| conv-mb8 | train_conv | 8×1 (8) | 16 | **complete (chosen)** | 0.48 s/step | 6,449 MiB | clean end-to-end run; phase peaks: train 6,447 / valid 6,449 / test 6,449; durations: 50 steps 24 s, valid 6,253@16 = 7:36, test 7,978@16 = 9:46. **Stage 6 chosen config:** 8×1, eval 16 → 70,445 steps × 0.48 s ≈ 9.4 h + 10×valid + 1×test ≈ 1.4 h ≈ **10.8 h** (vs 4×2 ≈ 12.8 h; eval 32 impossible — 1.47 GiB single block OOM) |
| rec-mb16 | train_rec | 16×4 (64) | 64 | **complete (chosen)** | 2.70 s/step | 7,437 MiB | clean end-to-end; phase peaks train 7,435 / valid 7,437 / test 7,437; evals fast (valid 6,253 = 62 s, test 7,978 = 94 s). **Stage 7 chosen config:** 16×4, eval 64 → 4,402 steps × 2.70 s ≈ 3.3 h + 5×(valid+test) ≈ 13 min ≈ **3.5 h**; same 200+200 padding caps as pre ⇒ worst-case bound proven by the pre-mb16-x-wc probe (7,757 MiB fits) |
| kv-on | infer_conv (64 test dialogues) | — (eval 16) | 16 | **complete** | 0.82 s/batch | 3,127 MiB | image `unicrs:baseline` (patch 6 in); 4 batches in 3.3 s; 56 samples logged; output sha256 `3b53499e…` |
| kv-off | infer_conv (64 test dialogues) | — (eval 16) | 16 | **complete** | 7.0 s/batch | 5,087 MiB | image `unicrs:kvoff` = same tree, patch 6 reverted (only difference, verified by diff); 28 s for the same 4 batches; output **byte-identical** → patch 6 CONFIRMED, generation ~8× faster |

### Projections (CAP-6, from the calibration measurements above)

| Stage | Chosen config | Work | Projected wall-clock |
|-------|---------------|------|----------------------|
| 5 — train_pre | 16×4, eval 64, alloc-conf | 8,710 steps × 2.75 s + 5×valid (2:41) + 1×test (3:23) | **≈ 6.9 h** |
| 6 — train_conv | 8×1, eval 16, alloc-conf | 70,445 steps × 0.48 s + 10×valid (7:36) + 1×test (9:46) | **≈ 10.8 h** |
| 6 — infer_conv ×3 | eval 16, KV patch on | 70,586 samples ≈ 4,412 batches × 0.85 s + 3× model load | **≈ 1.1 h** |
| 7 — merge + train_rec | 16×4, eval 64, alloc-conf | merge (minutes) + 4,402 steps × 2.70 s + 5×(valid 62 s + test 94 s) | **≈ 3.5 h** |
| **Total (stages 5–7)** | | | **≈ 22.3 GPU-hours** (vs the pre-calibration estimate 12–21 h; conv training is the long pole as predicted; without the KV patch, infer alone would add ~8 h) |

### Owner go/no-go (CAP-6 gate)

**GO** — owner verdict given 2026-10-28 in the working session, immediately after reviewing the calibration record and PR #8 ("go + merge. Do not implement the story 1.7 yet"):
- approved the full stages 5–7 run at the chosen configs (≈ 22.3 GPU-hours projected);
- merged fork PR #8 himself via the agent (merge commit `026bc11`);
- explicitly instructed that story 1.7 must **not** be started yet — launch waits for his separate go.

Recorded on ConvRecX issue #12 (comment + close): https://github.com/kieusontungwork-oss/ConvRecX/issues/12



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
| Train jsonl | save/redial/unicrs-conv_train.jsonl |
| Valid jsonl | save/redial/unicrs-conv_valid.jsonl |
| Test jsonl | save/redial/unicrs-conv_test.jsonl |
| Merged rec files (stage 7) | save/redial_gen/{train,valid,test}_data_processed.jsonl |

> Names changed by story 1.5: `infer_conv.py:165` derives the output prefix from
> `--prompt_encoder`'s parent folder (`unicrs-conv`), replacing the earlier
> `dialogpt_*` expectation; merge consumes exactly that prefix.