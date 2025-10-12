#!/usr/bin/env python3
"""
A standalone script to demonstrate and play with Lhotse concepts.

This script will:
1. Create dummy audio files and a corresponding metadata TSV file.
2. Load them into Lhotse's RecordingSet and SupervisionSet.
3. Create a CutSet.
4. Demonstrate manipulations like appending, mixing, and padding cuts.

To run:
1. pip install lhotse numpy soundfile
2. python lhotse_demo.py
"""

import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from lhotse import (CutSet, Recording, RecordingSet, SupervisionSegment,
                    SupervisionSet, fix_manifests,
                    validate_recordings_and_supervisions)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_dummy_data(data_dir: Path = Path("lhotse_demo_data")):
    """Creates dummy audio files and a metadata.tsv for the demo."""
    logging.info(f"Creating dummy data in '{data_dir}'...")
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sampling_rate = 16000  # 16kHz
    
    # Audio 1: 5 seconds of a 440Hz sine wave (note A4)
    t = np.linspace(0., 5., int(5 * sampling_rate), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data1 = amplitude * np.sin(2. * np.pi * 440. * t)
    sf.write(audio_dir / "audio1.wav", data1.astype(np.int16), sampling_rate)

    # Audio 2: 6 seconds of a 880Hz sine wave (note A5)
    t = np.linspace(0., 6., int(6 * sampling_rate), endpoint=False)
    data2 = amplitude * np.sin(2. * np.pi * 880. * t)
    sf.write(audio_dir / "audio2.wav", data2.astype(np.int16), sampling_rate)

    # Metadata TSV file
    with open(data_dir / "metadata.tsv", "w") as f:
        # Format: uniq_id\ttext\twav_path\tstart_time\tend_time
        f.write(
            f"full_sentence\tThis is a full sentence.\t{audio_dir / 'audio1.wav'}\n"
        )
        f.write(
            f"first_part\tThis is the first part.\t{audio_dir / 'audio2.wav'}\t0.5\t3.0\n"
        )
        f.write(
            f"second_part\tAnd this is the second part.\t{audio_dir / 'audio2.wav'}\t3.5\t5.5\n"
        )
    logging.info("Dummy data created successfully.")
    return data_dir


def main():
    # --- Step 1: Create dummy data ---
    data_dir = create_dummy_data()
    tsv_path = data_dir / "metadata.tsv"

    # --- Step 2: Load Recordings and Supervisions (similar to prepare_dataset.py) ---
    logging.info("\n--- Loading manifests from TSV ---")
    recordings = {}
    supervisions = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            wav_path = Path(parts[2])
            if wav_path.name not in recordings:
                recordings[wav_path.name] = Recording.from_file(wav_path)
            
            if len(parts) == 3:
                # Full utterance
                rec = recordings[wav_path.name]
                supervisions.append(SupervisionSegment(
                    id=parts[0], recording_id=rec.id, start=0.0, duration=rec.duration, text=parts[1]
                ))
            else:
                # Partial utterance
                start, end = float(parts[3]), float(parts[4])
                supervisions.append(SupervisionSegment(
                    id=parts[0], recording_id=recordings[wav_path.name].id, start=start, duration=round(end-start, 2), text=parts[1]
                ))

    recording_set = RecordingSet.from_recordings(recordings.values())
    supervision_set = SupervisionSet.from_segments(supervisions)

    logging.info(f"Loaded {len(recording_set)} recordings.")
    logging.info(f"Loaded {len(supervision_set)} supervisions.")

    # --- Step 3: Create a CutSet ---
    logging.info("\n--- Creating CutSet ---")
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    cut_set = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    logging.info("CutSet created:")
    print(cut_set)

    # --- Step 4: Play with Cuts ---
    logging.info("\n--- Manipulating Cuts ---")
    results_dir = data_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Get individual cuts by their ID
    cut_full = cut_set['full_sentence']
    cut_part1 = cut_set['first_part']
    cut_part2 = cut_set['second_part']

    # 1. Load audio from a cut into a numpy array
    logging.info(f"Loading audio for cut '{cut_part1.id}'...")
    audio_array = cut_part1.load_audio()
    logging.info(f"Audio loaded. Shape: {audio_array.shape}, Dtype: {audio_array.dtype}")

    # 2. Append two cuts
    logging.info("Appending cut_part1 and cut_part2...")
    appended_cut = cut_part1.append(cut_part2)
    appended_cut.to_file(results_dir / "appended.wav")
    logging.info(f"Saved appended audio to '{results_dir / 'appended.wav'}' (Duration: {appended_cut.duration}s)")

    # 3. Mix two cuts (superimposing them)
    logging.info("Mixing cut_part1 and cut_part2...")
    # We need to pad the shorter cut to match the longer one for mixing
    mixed_cut = cut_part1.pad(duration=cut_part2.duration).mix(cut_part2)
    mixed_cut.to_file(results_dir / "mixed.wav")
    logging.info(f"Saved mixed audio to '{results_dir / 'mixed.wav'}'")

    # 4. Amplify a cut (make it louder)
    logging.info("Amplifying cut_full...")
    louder_cut = cut_full.amplify(factor=1.8)
    louder_cut.to_file(results_dir / "louder.wav")
    logging.info(f"Saved louder audio to '{results_dir / 'louder.wav'}'")
    
    logging.info("\nDemo finished. Check the 'lhotse_demo_data/results' directory!")

if __name__ == "__main__":
    main()
