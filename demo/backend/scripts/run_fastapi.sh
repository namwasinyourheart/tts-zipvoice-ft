#!/bin/bash

export MODEL_NAME=zipvoice
export MODEL_DIR=/media/nampv1/hdd/models/tts/zipvoice_ft/
export CHECKPOINT_NAME=exp/zipvoice_finetune/checkpoint-4000.pt
export VOCODER_PATH=None
export TOKENIZER=espeak
export LANG=vi
export TOKENS_FILE=tokens.txt
export DEVICE=cuda

uvicorn app.main:app --host 0.0.0.0 --port 13082 --reload