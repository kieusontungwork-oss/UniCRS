.PHONY: build smoke stage0 stage1 stage2 stage3 stage4 stage5 stage6 stage7 stage8 clean

IMAGE ?= unicrs:baseline
VARIANT ?= cuda
DATA ?= $(PWD)/data
RUNS ?= $(PWD)/runs
SAVE ?= $(PWD)/save
HF_CACHE ?= $(PWD)/hf_cache

build:
	docker build --build-arg VARIANT=$(VARIANT) -t $(IMAGE) .

smoke:
	docker run --rm --gpus all $(IMAGE) python -c "import torch, transformers, accelerate, torch_scatter; print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} cuda={torch.cuda.is_available()}')"

stage0:
	@echo "Stage 0: Prereqs - run on host, not in container"
	@echo "  1. Install nvidia-container-toolkit (if CUDA)"
	@echo "  2. Fork and clone RUCAIBox/UniCRS"
	@echo "  3. Warm HF cache: microsoft/DialoGPT-small, roberta-base"

stage1: build smoke

stage2:
	docker run --rm --gpus all -v $(DATA):/app/data -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cd data && wget -nc https://downloads.dbpedia.org/repo/dbpedia/mappings/mappingbased-objects/2021.09.01/mappingbased-objects_lang=en.ttl.bz2 -P dbpedia/"

stage3:
	docker run --rm --gpus all -v $(DATA):/app/data -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cd data && python dbpedia/extract_kg.py && python redial/extract_subkg.py && python redial/remove_entity.py"

stage4:
	docker run --rm --gpus all -v $(DATA):/app/data -v $(PWD)/src:/app/src -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cp -r data/redial src/data/ && cd src && python data/redial/process.py && python data/redial/process_mask.py"

stage5:
	docker run --rm --gpus all -v $(DATA):/app/data -v $(RUNS):/app/runs -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "accelerate launch train_pre.py --dataset redial --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --num_train_epochs 5 --gradient_accumulation_steps 4 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --num_warmup_steps 1389 --max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 5e-4 --output_dir ../runs/unicrs-prompt-pre"

stage6-train:
	docker run --rm --gpus all -v $(DATA):/app/data -v $(RUNS):/app/runs -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "accelerate launch train_conv.py --dataset redial --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder ../runs/unicrs-prompt-pre --num_train_epochs 10 --gradient_accumulation_steps 4 --ignore_pad_token_for_loss --per_device_train_batch_size 2 --per_device_eval_batch_size 32 --num_warmup_steps 6345 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir ../runs/unicrs-conv"

stage6-infer:
	docker run --rm --gpus all -v $(DATA):/app/data -v $(RUNS):/app/runs -v $(SAVE):/app/save -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c 'for split in train valid test; do accelerate launch infer_conv.py --dataset redial --split $$split --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder ../runs/unicrs-conv --per_device_eval_batch_size 32 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32; done'

stage6: stage6-train stage6-infer

stage7:
	docker run --rm --gpus all -v $(DATA):/app/data -v $(RUNS):/app/runs -v $(SAVE):/app/save -v $(HF_CACHE):/hf_cache $(IMAGE) \
		bash -c "cp -r data/redial/. data/redial_gen/ && python data/redial_gen/merge.py --gen_file_prefix dialogpt && accelerate launch train_rec.py --dataset redial_gen --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_rec 10 --prompt_encoder ../runs/unicrs-prompt-pre --num_train_epochs 5 --gradient_accumulation_steps 4 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --num_warmup_steps 530 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir ../runs/unicrs-rec"

stage8:
	@echo "Stage 8: Record Recall@1/10/50 from train_rec eval logs"
	@echo "Write baseline report to _bmad-output/implementation-artifacts/baselines/"

clean:
	docker rmi $(IMAGE) 2>/dev/null || true
	rm -rf $(RUNS)/* $(SAVE)/*