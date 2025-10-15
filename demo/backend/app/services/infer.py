import os
import json
import tempfile
import subprocess
from typing import Optional
from app.core.config import settings
from .log_utils import setup_logger
from .utils import transcribe

logger = setup_logger(__name__)

VOICE_INFO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "voices", "voice_info.json")

DEFAULT_VOICE = "lan_chi-female-north"

_voice_map = None

def load_voice_map(voice_info_path: str = VOICE_INFO_PATH):
    with open(voice_info_path, "r", encoding="utf-8") as f:
        voice_info = json.load(f)
    return {v["id"]: v for v in voice_info}

def _ensure_voice_map():
    global _voice_map
    if _voice_map is None:
        _voice_map = load_voice_map()

_ensure_voice_map()

print(_voice_map)

def _get_voice_by_id(voice_id: str):
    return _voice_map.get(voice_id)


def tts_infer_voice(
    text: str,
    voice: str = DEFAULT_VOICE,
    device=settings.DEVICE
):
    voice_data = _get_voice_by_id(voice)
    print("voice_data", voice_data)
    if not voice_data:
        logger.warning(f"Voice '{voice}' not found, fallback to default '{DEFAULT_VOICE}'")
        voice_data = _get_voice_by_id(DEFAULT_VOICE)

    voice_path = voice_data['path']

    logger.info(f"Using voice preset: {voice_data['name']} ({voice_path})")

    with open(voice_path, "rb") as f:
        ref_audio = f.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio.write(ref_audio)
        tmp_audio_path = tmp_audio.name

    try:
        ref_text = transcribe(tmp_audio_path)["text"]
        logger.info(f"Ref text (auto): {ref_text}")
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        ref_text = ""

    model_dir_old = settings.MODEL_DIR.rstrip("/")
    checkpoint_path_old = settings.CHECKPOINT_NAME
    vocoder_dirname_old = settings.VOCODER_DIRNAME

    checkpoint_name = os.path.basename(checkpoint_path_old)
    model_subdir = os.path.dirname(checkpoint_path_old)
    model_dir_new = os.path.join(model_dir_old, model_subdir)
    vocoder_path = os.path.join(model_dir_old, vocoder_dirname_old)
    output_wav = os.path.join(tempfile.gettempdir(), "result.wav")

    cmd = [
        "python3", "-m", "zipvoice.bin.infer_zipvoice",
        "--model-name", "zipvoice",
        "--model-dir", model_dir_new,
        "--checkpoint-name", checkpoint_name,
        "--vocoder-path", vocoder_path,
        "--tokenizer", "espeak",
        "--lang", settings.LANG,
        "--prompt-wav", tmp_audio_path,
        "--prompt-text", ref_text,
        "--text", text,
        "--res-wav-path", output_wav
    ]

    logger.info(f"Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info("==== STDOUT ====\n" + result.stdout)
    logger.info("==== STDERR ====\n" + result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Inference failed (code {result.returncode})")

    with open(output_wav, "rb") as f:
        audio_bytes = f.read()

    return {"audio_bytes": audio_bytes, "sampling_rate": 24000}


def tts_infer_clone(
    text: str,
    ref_audio: bytes,
    device=settings.DEVICE
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio.write(ref_audio)
        tmp_audio_path = tmp_audio.name

    try:
        ref_text = transcribe(tmp_audio_path)["text"]
        logger.info(f"Ref text (auto): {ref_text}")
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        ref_text = ""

    model_dir_old = settings.MODEL_DIR.rstrip("/")
    checkpoint_path_old = settings.CHECKPOINT_NAME
    vocoder_dirname_old = settings.VOCODER_DIRNAME

    checkpoint_name = os.path.basename(checkpoint_path_old)
    model_subdir = os.path.dirname(checkpoint_path_old)
    model_dir_new = os.path.join(model_dir_old, model_subdir)
    vocoder_path = os.path.join(model_dir_old, vocoder_dirname_old)
    output_wav = os.path.join(tempfile.gettempdir(), "result.wav")

    cmd = [
        "python3", "-m", "zipvoice.bin.infer_zipvoice",
        "--model-name", "zipvoice",
        "--model-dir", model_dir_new,
        "--checkpoint-name", checkpoint_name,
        "--vocoder-path", vocoder_path,
        "--tokenizer", "espeak",
        "--lang", settings.LANG,
        "--prompt-wav", tmp_audio_path,
        "--prompt-text", ref_text,
        "--text", text,
        "--res-wav-path", output_wav
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Voice cloning failed (code {result.returncode})")

    with open(output_wav, "rb") as f:
        audio_bytes = f.read()

    return {"audio_bytes": audio_bytes, "sampling_rate": 24000}
