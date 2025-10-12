#!/usr/bin/env bash
# run_prepare_dataset.sh
# Example script to generate Lhotse manifests from a Hugging Face dataset directory.

set -euo pipefail

# ----------- Configuration -----------
HF_RAW_DATA_DIR="/media/nampv1/hdd/data/TTS-viVoice-1017h/raw/hf/subset_887"        # path to the Hugging Face dataset (load_from_disk)
OUTPUT_DIR="/media/nampv1/hdd/data/TTS-viVoice-1017h/manifests"          # where to save Lhotse manifests
PREFIX="custom"                      # dataset name prefix
SUBSET="train"                       # or "dev"
NUM_JOBS=4                    # number of parallel threads
SAMPLING_RATE=24000                  # target sampling rate

# ----------- Execution -----------
echo "Preparing dataset..."
python3 -m zipvoice.bin.prepare_dataset1 \
  --hf-raw-data-dir "${HF_RAW_DATA_DIR}" \
  --prefix "${PREFIX}" \
  --subset "${SUBSET}" \
  --num-jobs "${NUM_JOBS}" \
  --sampling-rate "${SAMPLING_RATE}" \
  --output-dir "${OUTPUT_DIR}"

echo "✅ Done. Output saved to ${OUTPUT_DIR}/${PREFIX}_cuts_${SUBSET}.jsonl.gz"
