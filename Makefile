.PHONY: build smoke stage0 stage1 stage2 stage3 stage4 stage5 stage6 stage7 stage8 clean

IMAGE ?= unicrs:baseline
VARIANT ?= cuda

# stack.md patch 13 (story 1.6, decision D-1): run as the invoking host user
# (-e HOME=/tmp: no writable home needed) so outputs under runs/ and save/
# are never root-owned on the shared server.
UID_ARGS = --user $$(id -u):$$(id -g) -e HOME=/tmp
DATA ?= $(PWD)/data
RUNS ?= $(PWD)/runs
SAVE ?= $(PWD)/save
HF_CACHE ?= $(PWD)/hf_cache

build:
	docker build --build-arg VARIANT=$(VARIANT) -t $(IMAGE) .

smoke:
	docker run --rm --gpus all $(UID_ARGS) $(IMAGE) python -c "import torch, transformers, accelerate, torch_geometric, torch_scatter, torch_sparse, loguru, nltk, wandb, numpy, tqdm; print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} torch_geometric={torch_geometric.__version__} numpy={numpy.__version__} tqdm={tqdm.__version__} cuda={torch.cuda.is_available()}')"

stage0:
	@echo "Stage 0: Prereqs - run on host, not in container"
	@echo "  1. Install nvidia-container-toolkit (if CUDA)"
	@echo "  2. Fork and clone RUCAIBox/UniCRS"
	@echo "  3. Warm HF cache: microsoft/DialoGPT-small, roberta-base"

stage1: build smoke

stage2:
	docker run --rm --gpus all $(UID_ARGS) -v $(DATA):/app/data -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cd data && wget -nc https://downloads.dbpedia.org/repo/dbpedia/mappings/mappingbased-objects/2021.09.01/mappingbased-objects_lang=en.ttl.bz2 -P dbpedia/"

stage3:
	docker run --rm --gpus all $(UID_ARGS) -v $(DATA):/app/data -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cd data && python dbpedia/extract_kg.py && python redial/extract_subkg.py && python redial/remove_entity.py"

stage4:
	docker run --rm --gpus all $(UID_ARGS) -v $(DATA):/app/data -v $(PWD)/src:/app/src -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cp -r data/redial src/data/ && cd src && python data/redial/process.py && python data/redial/process_mask.py"

stage5:
	@test ! -e $(RUNS)/unicrs-prompt-pre || { echo "ABORT: runs/unicrs-prompt-pre already exists (full-run targets never overwrite)"; exit 1; }
	@git rev-parse HEAD
	@docker run --rm --gpus all $(UID_ARGS) -v $(DATA):/app/data:ro -v $(RUNS):/app/runs -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cd src && cp -r /app/data/redial/. data/redial/ && cp /app/data/redial_pre/*_data_processed.jsonl data/redial/ && accelerate launch train_pre.py --dataset redial --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --num_train_epochs 5 --gradient_accumulation_steps 4 --per_device_train_batch_size 16 --per_device_eval_batch_size 64 --num_warmup_steps 1389 --max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 5e-4 --output_dir ../runs/unicrs-prompt-pre"

stage6-train:
	@test ! -e $(RUNS)/unicrs-conv || { echo "ABORT: runs/unicrs-conv already exists (full-run targets never overwrite)"; exit 1; }
	@git rev-parse HEAD
	@docker run --rm --gpus all $(UID_ARGS) -v $(DATA):/app/data:ro -v $(RUNS):/app/runs -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cd src && cp -r /app/data/redial/. data/redial/ && accelerate launch train_conv.py --dataset redial --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder ../runs/unicrs-prompt-pre/best --num_train_epochs 10 --gradient_accumulation_steps 1 --ignore_pad_token_for_loss --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --num_warmup_steps 6345 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir ../runs/unicrs-conv"

stage6-infer:
	@test ! -e $(SAVE)/redial/unicrs-conv_train.jsonl -a ! -e $(SAVE)/redial/unicrs-conv_valid.jsonl -a ! -e $(SAVE)/redial/unicrs-conv_test.jsonl || { echo "ABORT: save/redial/unicrs-conv_*.jsonl already exist (full-run targets never overwrite)"; exit 1; }
	@git rev-parse HEAD
	@docker run --rm --gpus all $(UID_ARGS) -v $(DATA):/app/data:ro -v $(RUNS):/app/runs -v $(SAVE):/app/src/save -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c 'cd src && cp -r /app/data/redial/. data/redial/ && for split in train valid test; do accelerate launch infer_conv.py --dataset redial --split $$split --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder ../runs/unicrs-conv/best --per_device_eval_batch_size 16 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32; done'

stage6: stage6-train stage6-infer

stage7:
	@test -f $(SAVE)/redial/unicrs-conv_train.jsonl -a -f $(SAVE)/redial/unicrs-conv_valid.jsonl -a -f $(SAVE)/redial/unicrs-conv_test.jsonl || { echo "ABORT: stage6-infer outputs missing under save/redial/ - run stage6-infer first"; exit 1; }
	@test ! -e $(RUNS)/unicrs-rec -a ! -e $(SAVE)/redial_gen || { echo "ABORT: runs/unicrs-rec or save/redial_gen already exists (full-run targets never overwrite)"; exit 1; }
	@git rev-parse HEAD
	@docker run --rm --gpus all $(UID_ARGS) -v $(DATA):/app/data:ro -v $(RUNS):/app/runs -v $(SAVE):/app/src/save -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cd src && cp -r /app/data/redial/. data/redial/ && mkdir -p data/redial_gen && cp -r data/redial/. data/redial_gen/ && (cd data/redial_gen && python merge.py --gen_file_prefix unicrs-conv) && mkdir -p save/redial_gen && cp data/redial_gen/*_data_processed.jsonl save/redial_gen/ && accelerate launch train_rec.py --dataset redial_gen --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_rec 10 --prompt_encoder ../runs/unicrs-prompt-pre/best --num_train_epochs 5 --gradient_accumulation_steps 4 --per_device_train_batch_size 16 --per_device_eval_batch_size 64 --num_warmup_steps 530 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir ../runs/unicrs-rec"

stage8:
	@echo "Stage 8: Record Recall@1/10/50 from train_rec eval logs"
	@echo "Write baseline report to _bmad-output/implementation-artifacts/baselines/"

clean:
	docker rmi $(IMAGE) 2>/dev/null || true