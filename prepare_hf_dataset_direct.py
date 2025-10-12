#!/usr/bin/env python3
"""
Script để chuyển đổi HuggingFace dataset trực tiếp sang Lhotse manifest
cho ZipVoice fine-tuning mà không cần lưu file audio trung gian.

Usage:
python prepare_hf_dataset_direct.py \
    --dataset-name your_hf_dataset \
    --output-dir data/manifests \
    --prefix custom-finetune
"""

import argparse
import logging
import os
import random
from pathlib import Path

from datasets import load_dataset
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached


def prepare_hf_dataset_direct(
    dataset_name: str,
    output_dir: str,
    prefix: str,
    split_ratio: float = 0.9,
):
    """
    Chuyển đổi HF dataset trực tiếp sang Lhotse manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Kiểm tra xem manifest đã tồn tại chưa
    train_manifest = output_dir / f"{prefix}_cuts_train.jsonl.gz"
    dev_manifest = output_dir / f"{prefix}_cuts_dev.jsonl.gz"
    if train_manifest.is_file() and dev_manifest.is_file():
        logging.info("Manifests already exist, skipping preparation.")
        return

    logging.info(f"Loading dataset: {dataset_name}")
    try:
        if os.path.isdir(dataset_name):
            dataset = load_dataset(dataset_name)
        else:
            dataset = load_dataset(dataset_name)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # Giả sử dataset chỉ có split 'train'
    if 'train' not in dataset:
        logging.error(f"'train' split not found in dataset: {list(dataset.keys())}")
        return

    split_data = dataset['train']
    logging.info(f"Processing 'train' split with {len(split_data)} samples.")

    # Trộn và chia train/dev
    indices = list(range(len(split_data)))
    random.shuffle(indices)
    train_size = int(len(indices) * split_ratio)
    train_indices = indices[:train_size]
    dev_indices = indices[train_size:]

    logging.info(f"Splitting into: {len(train_indices)} train, {len(dev_indices)} dev samples")

    # Tạo CutSet cho train và dev
    for subset_name, subset_indices in [("train", train_indices), ("dev", dev_indices)]:
        logging.info(f"Creating manifest for {subset_name} subset...")
        cut_set = CutSet.from_cuts(
            (
                # Tạo một 'cut' cho mỗi sample
                SupervisionSegment(
                    id=f"{prefix}_{subset_name}_{idx:06d}",
                    recording_id=f"{prefix}_{subset_name}_{idx:06d}",
                    start=0.0,
                    duration=len(split_data[idx]['audio']['array']) / split_data[idx]['audio']['sampling_rate'],
                    text=split_data[idx]['text'].strip(),
                    language="vietnamese", # Có thể thay đổi
                    speaker=split_data[idx].get('channel', 'unknown'),
                ).to_cut(
                    # Tạo Recording object từ audio array trong bộ nhớ
                    Recording.from_file(
                        path=f"memory://{prefix}_{subset_name}_{idx:06d}.wav",
                        audio_bytes=sf.write(
                            io.BytesIO(),
                            split_data[idx]['audio']['array'],
                            split_data[idx]['audio']['sampling_rate'],
                            format='WAV'
                        ).getvalue(),
                    )
                )
                for idx in subset_indices
            )
        )

        # Lưu manifest
        manifest_path = output_dir / f"{prefix}_cuts_{subset_name}.jsonl.gz"
        cut_set.to_file(manifest_path)
        logging.info(f"Saved manifest to {manifest_path}")

    logging.info("\n✅ Direct manifest creation completed!")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Convert HF dataset to Lhotse manifest directly."
    )
    parser.add_argument("--dataset-name", required=True, help="HF dataset name or local path")
    parser.add_argument("--output-dir", default="data/manifests", help="Output directory for manifests")
    parser.add_argument("--prefix", default="custom-finetune", help="Prefix for manifest files")
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Train/dev split ratio")

    # Thêm thư viện soundfile và io để xử lý audio trong bộ nhớ
    global sf, io
    try:
        import soundfile as sf
        import io
    except ImportError:
        logging.error("Please install soundfile: pip install soundfile")
        exit(1)

    args = parser.parse_args()
    prepare_hf_dataset_direct(args.dataset_name, args.output_dir, args.prefix, args.split_ratio)


if __name__ == "__main__":
    main()
