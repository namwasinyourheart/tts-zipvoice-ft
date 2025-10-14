from pydantic import BaseModel
from typing import List, Optional, Any

class TTSRequest(BaseModel):
    text: str
    ref_audio: bytes
    ref_text: Optional[str] = None