from fastapi import APIRouter, File, Query, UploadFile
from app.services.infer import tts_infer
from app.schemas.tts import TTSResponse

router = APIRouter()

@router.post("/tts", response_model=TTSResponse)
async def tts(
    text: str,
    ref_audio: UploadFile = File(...),
    ref_text: str = Query(None)
):
    ref_audio_bytes = await ref_audio.read()
    return tts_infer(text, ref_audio_bytes, ref_text)