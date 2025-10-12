from pydantic import BaseModel
from typing import List, Optional, Any

class TTSResponse(BaseModel):
    audio: bytes
    

class TTSRequest(BaseModel):
    text: str
    ref_audio: bytes
    ref_text: str