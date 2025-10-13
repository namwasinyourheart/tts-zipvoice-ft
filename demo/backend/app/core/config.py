import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = os.getenv("MODEL_NAME", "zipvoice")   # zipvoice | zipvoice_distill
    MODEL_DIR: str = os.getenv("MODEL_DIR", "zipvoice")
    CHECKPOINT_NAME: str = os.getenv("CHECKPOINT_NAME", "model.pt")
    VOCODER_DIRNAME: str = os.getenv("VOCODER_DIRNAME", "vocoder")
    
    TOKENIZER: str = os.getenv("TOKENIZER", "emilia")
    LANG: str = os.getenv("LANG", "en-us")
    TOKENS_FILE: str = os.getenv("TOKENS_FILE", "tokens.txt")
    DEVICE: str = os.getenv("DEVICE", "cuda")  # cpu | cuda
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/asr")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ASR_API_ENDPOINT: str = os.getenv("ASR_API_ENDPOINT", "https://ai.vnpost.vn/voiceai/asr/asr/v1/file")


settings = Settings()