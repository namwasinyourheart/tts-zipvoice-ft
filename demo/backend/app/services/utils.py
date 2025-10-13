import requests

def transcribe(audio_path: str) -> dict:
    ASR_API_URL = "https://ai.vnpost.vn/voiceai/asr/asr/v1/file"

    with open(audio_path, "rb") as f:
        files = {
            "audio_file": (audio_path, f, "audio/wav"),
        }
        data = {
            "enhance_speech": "true",
            "postprocess_text": "true",
        }

        response = requests.post(ASR_API_URL, files=files, data=data)
        response.raise_for_status()

        result = response.json()
        text = result.get("text", "")

        return {"text": text}
