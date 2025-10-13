
from pydub import AudioSegment, silence
import tempfile
import hashlib
import matplotlib.pylab as plt
import librosa
from transformers import pipeline
import re
import torch
import numpy as np
import os
# from scipy.io import wavfile
# from scipy.signal import resample_poly

_ref_audio_cache = {}
asr_pipe = None

# def resample_to_24khz(input_path: str, output_path: str):
#     """
#     Resample WAV audio file to 24,000 Hz using scipy.
#     Parameters:
#     - input_path (str): Path to the input WAV file.
#     - output_path (str): Path to save the output WAV file.
#     """
#     # Load WAV file
#     orig_sr, audio = wavfile.read(input_path)

#     # Convert to mono if stereo
#     if len(audio.shape) == 2:
#         audio = audio.mean(axis=1)

#     # Convert to float32 for processing
#     if audio.dtype != np.float32:
#         audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

#     # Resample
#     target_sr = 24000
#     resampled = resample_poly(audio, target_sr, orig_sr)

#     # Convert back to int16 for saving
#     resampled_int16 = (resampled * 32767).astype(np.int16)

    # Save output
    # wavfile.write(output_path, target_sr, resampled_int16)

import librosa
import soundfile as sf

def resample_to_24khz(input_path: str, output_path: str):
    # Load file (librosa tự đọc được hầu hết định dạng)
    audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
    
    # Resample
    target_sr = 24000
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    # Ghi lại file WAV
    sf.write(output_path, resampled, target_sr)


def chunk_text(text, max_chars=135):

    # print(text)

    # Bước 1: Tách câu theo dấu ". "
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    
    # Ghép câu ngắn hơn 4 từ với câu liền kề
    i = 0
    while i < len(sentences):
        if len(sentences[i].split()) < 4:
            if i == 0 and i + 1 < len(sentences):
                # Ghép với câu sau
                sentences[i + 1] = sentences[i] + ', ' + sentences[i + 1]
                del sentences[i]
            else:
                if i - 1 >= 0:
                    # Ghép với câu trước
                    sentences[i - 1] = sentences[i - 1] + ', ' + sentences[i]
                    del sentences[i]
                    i -= 1
        else:
            i += 1

    # print(sentences)

    # Bước 2: Tách phần quá dài trong câu theo dấu ", "
    final_sentences = []
    for sentence in sentences:
        parts = [p.strip() for p in sentence.split(', ')]
        buffer = []
        for part in parts:
            buffer.append(part)
            total_words = sum(len(p.split()) for p in buffer)
            if total_words > 20:
                # Tách câu ra
                long_part = ', '.join(buffer)
                final_sentences.append(long_part)
                buffer = []
        if buffer:
            final_sentences.append(', '.join(buffer))

    # print(final_sentences)

    if len(final_sentences[-1].split()) < 4 and len(final_sentences) >= 2:
        final_sentences[-2] = final_sentences[-2] + ", " + final_sentences[-1]
        final_sentences = final_sentences[0:-1]
    
    # print(final_sentences)

    return final_sentences

def initialize_asr_pipeline(device="cuda", dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="vinai/PhoWhisper-medium",
        torch_dtype=dtype,
        device=device,
    )

# transcribe
def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device="cuda")
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()

def caculate_spec(audio):
    # Compute spectrogram (Short-Time Fourier Transform)
    stft = librosa.stft(audio, n_fft=512, hop_length=256, win_length=512)
    spectrogram = np.abs(stft)
    # Convert to dB
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram_db

def save_spectrogram(audio, path):
    spectrogram = caculate_spec(audio)
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()

def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio

def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device="cuda"):

    show_info("Converting audio...")

    # ref_audio_orig_converted = ref_audio_orig.replace(".wav", "_24k.wav").replace(".mp3", "_24k.mp3").replace(".m4a", "_24k.m4a").replace(".flac", "_24k.flac")

    # resample_to_24khz(ref_audio_orig, ref_audio_orig_converted)

    # ref_audio_orig = ref_audio_orig_converted

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:

        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text
