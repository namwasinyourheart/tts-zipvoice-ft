import tempfile
import shutil
import requests
from .config import BACKEND_URL

# --- Interface chung ---
class TTSProvider:
    def generate(self, text, model=None, ref_audio=None, ref_text=None):
        raise NotImplementedError


# --- Backend Provider ---
class BackendTTS(TTSProvider):
    def __init__(self, api_url=None):
        self.api_url = api_url or BACKEND_URL

#     def generate(self, text, model="vnpost-tts-1.0", ref_audio=None, ref_text=None):
#         if not text:
#             return None, "Please enter text."
#         payload = {"text": text, "model": model}
#         if ref_audio:
#             payload["ref_audio"] = ref_audio
#         if ref_text:
#             payload["ref_text"] = ref_text
#         try:
#             resp = requests.post(self.api_url, json=payload, timeout=30)
#             if resp.status_code == 200:
#                 data = resp.json()
#                 audio_url = data.get("audio_url")
#                 if audio_url:
#                     audio_resp = requests.get(f"{BACKEND_URL}{audio_url}", stream=True, timeout=30)
#                     if audio_resp.status_code == 200:
#                         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#                             shutil.copyfileobj(audio_resp.raw, tmp)
#                             return tmp.name, "Success"
#                 return None, "Audio file not found."
#             return None, f"Backend error: {resp.text}"
#         except Exception as e:
#             return None, str(e)


    def generate(self, text, ref_audio=None, ref_text=None, model=None):
        """
        Gửi request đến API FastAPI /tts:
        - text: nội dung văn bản
        - ref_text: văn bản tham chiếu (tùy chọn)
        - ref_audio: đường dẫn file âm thanh tham chiếu (bắt buộc)
        """
        if not text:
            return None, "Please enter text."
        if not ref_audio:
            return None, "Please provide reference audio."

        try:
            # Chuẩn bị dữ liệu form
            data = {"text": text}
            if ref_text:
                data["ref_text"] = ref_text

            # Mở file ref_audio dưới dạng binary
            with open(ref_audio, "rb") as f:
                files = {"ref_audio": (ref_audio, f, "audio/wav")}
                resp = requests.post(self.api_url, data=data, files=files, timeout=60)

            # Kiểm tra phản hồi
            if resp.status_code == 200:
                # Lưu file âm thanh tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(resp.content)
                    return tmp.name, "Success"

            return None, f"Backend error: {resp.text}"

        except Exception as e:
            return None, str(e)


# --- gTTS Provider ---
from gtts import gTTS
class GTTSProvider(TTSProvider):
    def generate(self, text, model=None, ref_audio=None, ref_text=None):
        if not text:
            return None, "Please enter text."
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts = gTTS(text, lang="vi")
            tts.save(tmp.name)
            return tmp.name, "Success"


# --- pyttsx3 Provider ---
import pyttsx3
class Pyttsx3Provider(TTSProvider):
    def generate(self, text, model=None, ref_audio=None, ref_text=None):
        if not text:
            return None, "Please enter text."
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            engine = pyttsx3.init()
            engine.save_to_file(text, tmp.name)
            engine.runAndWait()
            return tmp.name, "Success"


# --- TTS Service ---
class TTSService:
    def __init__(self, model_name="vnpost/vnpost-tts-1.0"):
        # parse provider + model
        if "/" in model_name:
            provider, model = model_name.split("/", 1)
        else:
            provider, model = model_name, None

        providers = {
            "vnpost": BackendTTS(),
            "gtts": GTTSProvider(),
            "pyttsx3": Pyttsx3Provider()
        }
        self.provider = providers.get(provider, BackendTTS())
        self.model = model

    def generate_speech(self, text, ref_audio=None, ref_text=None):
        return self.provider.generate(text, model=self.model, ref_audio=ref_audio, ref_text=ref_text)
