#!/usr/bin/env python3
import io
import logging
import re
from pathlib import Path
from functools import partial

import numpy as np
import soundfile as sf
import torch
from datasets import load_from_disk
from lhotse import (
    Recording,
    SupervisionSegment,
    RecordingSet,
    SupervisionSet,
    CutSet,
    split_parallelize_combine,
    LilcomChunkyWriter,
)
from lhotse.audio import AudioSource
from tqdm.auto import tqdm

from zipvoice.utils.feature import VocosFbank
from zipvoice.tokenizer.tokenizer import add_tokens

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)


def _array_to_wav_bytes(array: np.ndarray, sampling_rate: int) -> bytes:
    """
    Encode a numpy audio array into WAV bytes (PCM16) in-memory.
    Accepts 1-D (samples,), 2-D (samples, channels) or (channels, samples).
    """
    arr = np.asarray(array)

    # Convert shape to (samples, channels) if needed
    if arr.ndim == 1:
        arr_out = arr
    elif arr.ndim == 2:
        # Heuristic: if first dim looks like channels (small) and second >> first, treat as (channels, samples)
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr_out = arr.T
        else:
            arr_out = arr
    else:
        # flatten unexpected dims
        arr_out = arr.reshape(-1)

    # If float, ensure float32 in -1..1 range. soundfile will convert to PCM16 on write.
    if np.issubdtype(arr_out.dtype, np.floating):
        arr_out = arr_out.astype(np.float32)
    else:
        # if integer types, cast to int16 (soundfile will accept ints too)
        arr_out = arr_out.astype(np.int16)

    buf = io.BytesIO()
    # write WAV as PCM_16 for broad compatibility
    sf.write(buf, arr_out, samplerate=sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------
# 1️⃣ Prepare dataset in-memory
# ---------------------------
def prepare_dataset_from_hf_arrays(dataset, subset="train", sampling_rate=24000):
    recordings = []
    supervisions = []

    for i, sample in enumerate(tqdm(dataset[subset], desc=f"Processing {subset}")):
        uniq_id = f"{subset}_{i}"

        # support HF audio field being dict or ArrowAudio
        audio_field = sample["audio"]
        if isinstance(audio_field, dict):
            audio_array = audio_field.get("array")
            sr = int(audio_field.get("sampling_rate", sampling_rate))
        else:
            # audio_field might be an array-like directly
            audio_array = audio_field
            sr = sampling_rate

        if audio_array is None:
            logging.warning(f"Sample {i} has no audio array: skipping")
            continue

        # Encode to WAV bytes (Lhotse expects bytes for AudioSource type='memory')
        wav_bytes = _array_to_wav_bytes(audio_array, sr)

        # Create Recording from bytes (in-memory WAV)
        # Recording.from_bytes will set AudioSource(type='memory', source=bytes)
        recording = Recording.from_bytes(wav_bytes, recording_id=uniq_id)

        recordings.append(recording)

        # Prepare supervision
        text = sample.get("text", "")
        text = re.sub("_", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        num_channels = recording.num_channels if hasattr(recording, "num_channels") else 1
        seg = SupervisionSegment(
            id=uniq_id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            channel=list(range(num_channels)),
            text=text,
        )
        supervisions.append(seg)

    if not recordings:
        raise RuntimeError("No recordings were created. Check your Hugging Face dataset 'audio' field.")

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    cut_set = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    cut_set = cut_set.resample(sampling_rate)
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
    return cut_set


# ---------------------------
# 2️⃣ Add tokens in-memory
# ---------------------------
def prepare_tokens_in_memory(cut_set, tokenizer="emilia", lang="en-us", num_jobs=20):
    logging.info("Adding tokens in-memory")
    _add_tokens = partial(add_tokens, tokenizer=tokenizer, lang=lang)
    cut_set = split_parallelize_combine(num_jobs=num_jobs, manifest=cut_set, fn=_add_tokens)
    return cut_set


# ---------------------------
# 3️⃣ Compute Fbank in-memory
# ---------------------------
def compute_fbank_in_memory(
    cut_set, sampling_rate=24000, num_jobs=1, storage_path="data/fbank/tmp_feats"
):
    """
    Compute and store features from an in-memory CutSet.

    Note:
    - `num_jobs` defaults to 1 to avoid multiprocessing spawn issues in some environments.
      You can increase it if running as a standalone Python process (not via here-doc/stdin).
    - Features (Lilcom) will be stored under `storage_path`.
    """
    logging.info(f"Computing Fbank features in-memory; storing to {storage_path}")
    cut_set = cut_set.resample(sampling_rate)
    extractor = VocosFbank()
    # ensure storage path exists
    storage_dir = Path(storage_path)
    storage_dir.parent.mkdir(parents=True, exist_ok=True)
    cut_set = cut_set.compute_and_store_features(
        extractor=extractor,
        storage_path=str(storage_path),
        num_jobs=num_jobs,
        storage_type=LilcomChunkyWriter,
    )
    return cut_set


# ---------------------------
# 4️⃣ Full pipeline
# ---------------------------
def prepare_pipeline(
    hf_raw_data_dir,
    subsets=("train"),
    tokenizer="emilia",
    lang="default",
    sampling_rate=24000,
    num_jobs=1,
    storage_base="/tmp/fbank_in_memory",
):
    logging.info(f"Loading Hugging Face dataset from {hf_raw_data_dir}")
    dataset = load_from_disk(hf_raw_data_dir)

    cut_sets = {}
    lang = "vi"
    for subset in subsets:
        logging.info(f"Preparing subset: {subset}")
        cut_set = prepare_dataset_from_hf_arrays(dataset, subset=subset, sampling_rate=sampling_rate)
        cut_set = prepare_tokens_in_memory(cut_set, tokenizer=tokenizer, lang=lang, num_jobs=num_jobs)
        storage_path = str(Path(storage_base) / f"{Path(hf_raw_data_dir).stem}_{subset}")
        cut_set = compute_fbank_in_memory(
            cut_set, sampling_rate=sampling_rate, num_jobs=num_jobs, storage_path=storage_path
        )
        cut_sets[subset] = cut_set
        logging.info(f"Subset {subset} is ready in-memory.")

        logging.info("\n")
        logging.info("cut_set:", cut_set)
        logging.info("\n")

    return cut_sets


def prepare_pipeline(hf_raw_data_dir, 
subsets=("train", "dev"), 
tokenizer="emilia", 
lang="default", 
sampling_rate=24000, 
num_jobs=20,
storage_base="/tmp/fbank_in_memory"):
    logging.info(f"Loading Hugging Face dataset from {hf_raw_data_dir}")
    from datasets import load_from_disk
    dataset = load_from_disk(hf_raw_data_dir)

    # dataset = dataset.select(range(10))

    cut_sets = {}
    for subset in subsets:
        if subset not in dataset:
            logging.warning(f"Subset '{subset}' not found in dataset, skipping.")
            continue

        logging.info(f"Preparing subset: {subset}")
        cut_set = prepare_dataset_from_hf_arrays(dataset, subset=subset, sampling_rate=sampling_rate)
        cut_set = prepare_tokens_in_memory(cut_set, tokenizer=tokenizer, lang=lang, num_jobs=num_jobs)
        # cut_set = compute_fbank_in_memory(cut_set, sampling_rate=sampling_rate, num_jobs=num_jobs,
        #                                   storage_path=f"/tmp/fbank_in_memory/subset_887_{subset}")
        
        storage_path = str(Path(storage_base) / f"{Path(hf_raw_data_dir).stem}_{subset}")
        cut_set = compute_fbank_in_memory(
            cut_set, sampling_rate=sampling_rate, num_jobs=num_jobs, storage_path=storage_path
        )
        cut_sets[subset] = cut_set
        logging.info(f"Subset {subset} is ready in-memory.")

        for cut in cut_set:
            print(f"ID: {cut.id}, Duration: {cut.duration}, Supervisions: {cut.supervisions}")

            # print(cut)

            print(cut.features.storage_path)

            # for feature in cut.features:
            #     print(feature)

            
            features = cut.load_features()  # numpy array, shape: (num_frames, feature_dim)
            
            print(features)
            print(features.shape)

            break

        # print("cut_set:", cut_set)

    return cut_sets



# ---------------------------
# 5️⃣ Example usage
# ---------------------------
if __name__ == "__main__":
    hf_raw_data_dir = "/media/nampv1/hdd/data/TTS-viVoice-1017h/raw/hf/subset_887"
    cut_sets = prepare_pipeline(
        hf_raw_data_dir=hf_raw_data_dir,
        subsets=("train",),
        tokenizer="emilia",
        lang="default",
        sampling_rate=24000,
        num_jobs=1,  # use 1 to be safe; increase when running as a standalone script file
        storage_base="/media/nampv1/hdd/data/TTS-viVoice-1017h/fbank",
    )

    logging.info("All subsets prepared in-memory. Ready for training.")
    # cut_sets contains the in-memory CutSet(s)
