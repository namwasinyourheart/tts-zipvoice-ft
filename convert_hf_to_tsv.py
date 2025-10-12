#!/usr/bin/env python3
"""
Script để chuyển đổi HuggingFace dataset sang định dạng TSV cho ZipVoice fine-tuning.

Dataset structure expected:
DatasetDict({
    train: Dataset({
        features: ['channel', 'text', 'audio'],
        num_rows: 2508
    })
})

Usage:
python convert_hf_to_tsv.py --dataset-name your_hf_dataset --output-dir data/raw
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import soundfile as sf
from datasets import load_dataset
import hashlib


def create_unique_id(text: str, index: int) -> str:
    """Tạo unique ID từ text và index."""
    # Tạo hash từ text để đảm bảo unique
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"hf_{index"06d"}_{text_hash}"


def save_audio_to_wav(audio_array: np.ndarray, sampling_rate: int, output_path: str) -> str:
    """Lưu audio array thành file WAV và trả về đường dẫn đầy đủ."""
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize audio array nếu cần
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Lưu file WAV
    sf.write(output_path, audio_array, sampling_rate)
    return output_path


def convert_hf_to_tsv(dataset_name: str, output_dir: str = "data/raw", split_ratio: float = 0.9):
    """
    Chuyển đổi HF dataset sang định dạng TSV cho ZipVoice.

    Args:
        dataset_name: Tên hoặc path của HF dataset
        output_dir: Thư mục đầu ra cho TSV và audio files
        split_ratio: Tỷ lệ chia train/dev (default 0.9 = 90% train, 10% dev)
    """
    print(f"Loading dataset: {dataset_name}")
    try:
        # Load dataset từ HF Hub hoặc local path
        if os.path.exists(dataset_name):
            dataset = load_dataset('json', data_files=dataset_name)
        else:
            dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded with splits: {list(dataset.keys())}")

    # Tạo output directories
    raw_dir = Path(output_dir)
    audio_dir = raw_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Xử lý từng split (thường chỉ có 'train')
    for split_name, split_data in dataset.items():
        print(f"Processing split: {split_name} ({len(split_data)} samples)")

        # Trộn dữ liệu để chia train/dev ngẫu nhiên
        indices = list(range(len(split_data)))
        random.shuffle(indices)

        train_size = int(len(indices) * split_ratio)
        train_indices = indices[:train_size]
        dev_indices = indices[train_size:]

        print(f"Split: {len(train_indices)} train, {len(dev_indices)} dev samples")

        # Tạo TSV files cho train và dev
        for subset_name, subset_indices in [("train", train_indices), ("dev", dev_indices)]:
            tsv_path = raw_dir / f"custom_{subset_name}.tsv"
            print(f"Creating {tsv_path}...")

            with open(tsv_path, 'w', encoding='utf-8') as tsv_file:
                for idx in subset_indices:
                    sample = split_data[idx]

                    # Lấy thông tin từ sample
                    text = sample['text'].strip()
                    speaker_id = sample.get('channel', 'unknown')
                    audio_info = sample['audio']

                    # Tạo unique ID
                    uniq_id = create_unique_id(text, idx)

                    # Xử lý audio
                    audio_array = audio_info['array']
                    sampling_rate = audio_info['sampling_rate']
                    original_path = audio_info.get('path', f'unknown_{idx}.wav')

                    # Tạo tên file WAV mới
                    wav_filename = f"{uniq_id}_{speaker_id}.wav"
                    wav_path = audio_dir / wav_filename

                    # Lưu audio thành file WAV
                    full_wav_path = save_audio_to_wav(audio_array, sampling_rate, str(wav_path))

                    # Ghi vào TSV file
                    # Format: {uniq_id}\t{text}\t{wav_path}
                    tsv_line = f"{uniq_id}\t{text}\t{full_wav_path}\n"
                    tsv_file.write(tsv_line)

            print(f"Created {tsv_path} with {len(subset_indices)} entries")

    print("
✅ Conversion completed!"    print(f"📁 TSV files created in: {raw_dir}")
    print(f"🎵 Audio files saved in: {audio_dir}")
    print("
📋 Next steps:"    print("1. Run: bash run_finetune.sh")
    print("2. Or manually run the stages in run_finetune.sh")


def main():
    parser = argparse.ArgumentParser(description="Convert HF dataset to TSV format for ZipVoice fine-tuning")
    parser.add_argument("--dataset-name", required=True,
                        help="HF dataset name or local path")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Output directory for TSV and audio files")
    parser.add_argument("--split-ratio", type=float, default=0.9,
                        help="Train/dev split ratio (default: 0.9)")

    args = parser.parse_args()

    convert_hf_to_tsv(args.dataset_name, args.output_dir, args.split_ratio)


if __name__ == "__main__":
    main()
