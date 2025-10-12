#!/usr/bin/env python3
# prepare_dataset_hf_array_writewav.py
# Converts a Hugging Face dataset (with audio arrays only) into Lhotse CutSet manifests.
# Writes WAV files to disk and creates Recordings from those files so manifests are serializable.

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from datasets import load_from_disk
from lhotse import CutSet, RecordingSet, validate_recordings_and_supervisions
from lhotse.audio import Recording
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-raw-data-dir",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset directory (load_from_disk).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="custom",
        help="Prefix of the output manifest file.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        help="Subset name, typically 'train' or 'dev'.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Number of threads (not used in array-based processing).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="Target sampling rate (will resample cuts to this rate).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="Directory to save manifest files and WAVs.",
    )
    return parser.parse_args()


def _ensure_arr_shape_for_soundfile(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array has shape (n_samples, n_channels) for soundfile.write.
    Accepts 1-D (samples,), 2-D (samples, channels) or (channels, samples).
    Returns array with shape (n_samples, n_channels) or (n_samples,) for mono.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # If first dim looks like channels (small) and second is samples, transpose.
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            # assume (channels, samples) -> transpose
            return arr.T
        # else assume already (samples, channels)
        return arr
    # higher dims unexpected -> flatten to 1D
    return arr.reshape(-1)


def _write_wav_and_get_recording(recording_id: str, audio_array: np.ndarray, sampling_rate: int, audio_dir: Path) -> Recording:
    """
    Write the numpy audio array to a WAV file under audio_dir and return a Recording.from_file.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{recording_id}.wav"

    # If file already exists, skip writing
    if not wav_path.is_file():
        arr = _ensure_arr_shape_for_soundfile(audio_array)
        # ensure float32 for soundfile; soundfile will handle int dtypes too but convert to float32 to be safe
        if np.issubdtype(arr.dtype, np.floating):
            arr_to_write = arr.astype(np.float32)
        else:
            arr_to_write = arr
        # Write WAV PCM16 (soundfile will convert floats to PCM)
        sf.write(str(wav_path), arr_to_write, samplerate=sampling_rate, format="WAV", subtype="PCM_16")

    # Create Recording from file path
    recording = Recording.from_file(path=str(wav_path), recording_id=recording_id)
    return recording


def _parse_supervision(supervision: list, recording_dict: dict) -> Optional[SupervisionSegment]:
    uniq_id, text, start, end = supervision
    try:
        recording = recording_dict[uniq_id]
        duration = end - start if end is not None else recording.duration
        # allow tiny numerical slack
        if duration > recording.duration + 1e-6:
            logging.warning(f"Supervision {uniq_id} duration {duration} > recording.duration {recording.duration}; trimming later.")
        text = re.sub("_", " ", text)
        text = re.sub(r"\s+", " ", text)
        return SupervisionSegment(
            id=uniq_id,
            recording_id=recording.id,
            start=start,
            duration=duration,
            channel=recording.channel_ids,
            text=text.strip(),
        )
    except Exception as e:
        logging.warning(f"Error processing supervision {uniq_id}: {e}")
        return None


def prepare_dataset(
    hf_raw_data_dir: Pathlike,
    prefix: str,
    subset: str,
    sampling_rate: int,
    output_dir: Pathlike,
):
    logging.info(f"Preparing {prefix} dataset subset={subset} from {hf_raw_data_dir}.")
    output_dir = Path(output_dir)
    audio_out_dir = output_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{prefix}_cuts_{subset}.jsonl.gz"
    out_path = output_dir / file_name
    if out_path.is_file():
        logging.info(f"{file_name} exists, skipping.")
        return

    logging.info("Loading Hugging Face dataset...")
    dataset_dict = load_from_disk(hf_raw_data_dir)
    if subset not in dataset_dict:
        raise ValueError(f"Subset '{subset}' not found in dataset dict: {list(dataset_dict.keys())}")
    dataset = dataset_dict[subset]

    recording_dict = {}
    supervision_list = []

    logging.info("Building recordings (writing WAVs) and supervisions...")
    for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        uniq_id = f"{subset}_{i}"
        # Expect sample["audio"] to be a dict with 'array' and 'sampling_rate'
        if "audio" not in sample:
            logging.warning(f"Sample {i} missing 'audio' field; skipping.")
            continue
        audio_field = sample["audio"]
        if isinstance(audio_field, dict):
            array = audio_field.get("array")
            sr = int(audio_field.get("sampling_rate", sampling_rate))
        else:
            array = audio_field
            sr = sampling_rate

        if array is None:
            logging.warning(f"Sample {i} has no audio array; skipping.")
            continue

        try:
            recording = _write_wav_and_get_recording(uniq_id, array, sr, audio_out_dir)
        except Exception as e:
            logging.warning(f"Failed to create recording for sample {i}: {e}")
            continue

        recording_dict[uniq_id] = recording

        text = sample.get("text", "").strip()
        start, end = 0.0, None
        supervision_list.append((uniq_id, text, start, end))

    if not recording_dict:
        raise RuntimeError("No recordings were created. Check your Hugging Face dataset 'audio' field.")

    logging.info("Building supervisions...")
    supervisions = []
    for supervision in tqdm(supervision_list, desc="Processing supervisions"):
        seg = _parse_supervision(supervision, recording_dict)
        if seg is not None:
            supervisions.append(seg)

    logging.info("Creating manifests and validating...")
    recording_set = RecordingSet.from_recordings(recording_dict.values())
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    logging.info("Creating CutSet...")
    cut_set = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    cut_set = cut_set.sort_by_recording_id()
    cut_set = cut_set.resample(sampling_rate)
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)

    logging.info(f"Saving to {out_path}")
    cut_set.to_file(out_path)
    logging.info("Done!")
    logging.info(f"WAVs saved under {audio_out_dir}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_args()
    prepare_dataset(
        hf_raw_data_dir=args.hf_raw_data_dir,
        prefix=args.prefix,
        subset=args.subset,
        sampling_rate=args.sampling_rate,
        output_dir=args.output_dir,
    )
