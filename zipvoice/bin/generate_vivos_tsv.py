import os
from pathlib import Path
from tqdm.auto import tqdm

def generate_vivos_tsv(vivos_root="vivos", output_dir="data/raw", prefix="vivos"):
    """
    Sinh file TSV (train/dev/test) từ dataset VIVOS gốc.
    """
    subsets = ["train", "test"]
    os.makedirs(output_dir, exist_ok=True)

    for subset in subsets:
        subset_dir = Path(vivos_root) / subset
        prompts_path = subset_dir / "prompts.txt"
        waves_dir = subset_dir / "waves"

        # 1. Đọc transcript
        transcripts = {}
        with open(prompts_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wav_id, text = parts
                    transcripts[wav_id] = text.strip()
                else:
                    print(f"⚠️  Bỏ qua dòng lỗi: {line}")

        # 2. Duyệt WAVs
        tsv_lines = []
        wav_files = list(waves_dir.rglob("*.wav"))

        for wav_path in tqdm(wav_files, desc=f"{subset}"):
            wav_id = wav_path.stem
            if wav_id not in transcripts:
                print(f"⚠️  Không tìm thấy transcript cho {wav_id}")
                continue
            text = transcripts[wav_id].replace("\t", " ").strip()
            tsv_lines.append(f"{wav_id}\t{text}\t{wav_path.resolve()}")

        # 3. Ghi TSV
        tsv_out = Path(output_dir) / f"{prefix}_{subset}.tsv"
        with open(tsv_out, "w", encoding="utf-8") as f:
            for line in tsv_lines:
                f.write(line + "\n")

        print(f"✅ TSV saved: {tsv_out} ({len(tsv_lines)} samples)")

if __name__ == "__main__":
    generate_vivos_tsv(vivos_root="/media/nampv1/hdd/data/VIVOS/raw/vivos", output_dir="media/nampv1/hdd/data/TTS-VIVOS", prefix="vivos")