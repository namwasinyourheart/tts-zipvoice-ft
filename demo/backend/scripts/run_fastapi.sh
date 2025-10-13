#!/bin/bash

export MODEL_NAME=zipvoice
export MODEL_DIR=/media/nampv1/hdd/models/tts/zipvoice_ft/
export CHECKPOINT_NAME=exp/zipvoice_finetune/checkpoint-4000.pt
export VOCODER_DIRNAME=vocoder/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21
export VOCODER_PATH=None
export TOKENIZER=espeak
export LANG=vi
export TOKENS_FILE=tokens.txt
export DEVICE=cpu

uvicorn app.main:app --host 0.0.0.0 --port 13082 --reload