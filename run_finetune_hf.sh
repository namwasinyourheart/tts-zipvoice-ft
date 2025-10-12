#!/bin/bash

# This script is an example of fine-tuning ZipVoice on your custom HF dataset.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

stage=1
stop_stage=7

nj=20
is_zh_en=0 # Set to 0 for other languages like Vietnamese
lang=vi # Set to your language code for espeak

# Your HuggingFace dataset name or local path
hf_dataset_name="your-hf-dataset-name"

# You can set `max_len` according to statistics from the command 
# `lhotse cut describe data/manifests/custom-finetune_cuts_train.jsonl.gz`.
max_len=20

# Download directory for pre-trained models
download_dir=download/

if [ $is_zh_en -eq 1 ]; then
      tokenizer=emilia
else
      tokenizer=espeak
      [ "$lang" = "default" ] && { echo "Error: lang is not set!" >&2; exit 1; }
fi

### Prepare the training data (1 - 4)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests directly from HuggingFace dataset"
      
      python3 prepare_hf_dataset_direct.py \
            --dataset-name "${hf_dataset_name}" \
            --output-dir data/manifests \
            --prefix custom-finetune

      # The output manifest files are "data/manifests/custom-finetune_cuts_train.jsonl.gz"
      # and "data/manifests/custom-finetune_cuts_dev.jsonl.gz".
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests"
      for subset in train dev;do
            python3 -m zipvoice.bin.prepare_tokens \
                  --input-file data/manifests/custom-finetune_cuts_${subset}.jsonl.gz \
                  --output-file data/manifests/custom-finetune_cuts_${subset}_tok.jsonl.gz \
                  --tokenizer ${tokenizer} \
                  --lang ${lang}
      done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
      echo "Stage 3: Compute Fbank for custom dataset"
      for subset in train dev; do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset custom-finetune \
                  --subset ${subset}_tok \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Download pre-trained model"
      hf_repo=k2-fsa/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt tokens.txt model.json; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi

### Training ZipVoice (5 - 6)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Fine-tune the ZipVoice model"

      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 1 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters 10000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --model-config ${download_dir}/zipvoice/model.json \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${download_dir}/zipvoice/tokens.txt \
            --dataset custom \
            --train-manifest data/fbank/custom-finetune_cuts_train_tok.jsonl.gz \
            --dev-manifest data/fbank/custom-finetune_cuts_dev_tok.jsonl.gz \
            --exp-dir exp/zipvoice_finetune_hf
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Average the checkpoints"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 10000 \
            --avg 2 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_finetune_hf
fi

### Inference with PyTorch models (7)

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Inference of the fine-tuned model"

      # Create a dummy test.tsv for inference example
      echo -e "test_001\tXin chào thế giới.\t/path/to/your/prompt.wav" > test.tsv

      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_finetune_hf/ \
            --checkpoint-name iter-10000-avg-2.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list test.tsv \
            --res-dir results/test_finetune_hf
fi
