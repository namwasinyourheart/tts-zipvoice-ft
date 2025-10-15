from pydantic import BaseModel
from typing import Optional

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    ref_text: Optional[str] = None
