#!/bin/bash

# scripts_in_memory_minimal.sh
# Prepare dataset in-memory, add tokens, compute Fbank, ready for training

set -euo pipefail

# Add project root to PYTHONPATH
set +u
export PYTHONPATH=/home/nampv1/projects/tts/tts-ft/ZipVoice/zipvoice/bin:$PYTHONPATH
set -u

# ---------------------------
# Config
# ---------------------------
root_data_dir="/media/nampv1/hdd/data/TTS-viVoice-1017h"
hf_raw_data_dir="${root_data_dir}/raw/hf/subset_887"
fbank_in_memory_dir="${root_data_dir}/fbank_in_memory"
nj=1
tokenizer="espeak"
lang="vi"
sampling_rate=24000
max_len=20   # max duration of utterances in seconds

exp_dir="exp/zipvoice_finetune"

# ---------------------------
# Stage 1: Prepare dataset in-memory, add tokens, compute Fbank
# ---------------------------
echo "Stage 1: Prepare dataset in-memory"

python3 - <<EOF
import logging
from prepare_pipeline_in_memory import prepare_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s")

cut_sets = prepare_pipeline(
    hf_raw_data_dir="${hf_raw_data_dir}",
    subsets=("train","dev"),
    tokenizer="${tokenizer}",
    lang="${lang}",
    sampling_rate=${sampling_rate},
    num_jobs=${nj},
    storage_base="${fbank_in_memory_dir}",
)

train_cut_set = cut_sets.get("train", None)
if train_cut_set is None:
    logging.error("Train split not found in dataset. Exiting.")
    exit(1)

dev_cut_set = cut_sets.get("dev", None)
if dev_cut_set is None:
    logging.warning("Dev split not found in dataset. Continuing with only train.")

logging.info("Stage 1 complete: CutSets in-memory, Fbank ready")
EOF

# ---------------------------
# Stage 2: Ready for training
# ---------------------------
echo "Stage 2: Training ZipVoice can start using train_cut_set and dev_cut_set in-memory"
echo "Use these CutSets directly in zipvoice.bin.train_zipvoice"
