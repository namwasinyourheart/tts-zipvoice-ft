from fastapi import APIRouter, File, Form, UploadFile
from app.services.infer import tts_infer
from app.schemas.tts import TTSRequest
from fastapi.responses import StreamingResponse
import io

router = APIRouter()



@router.post("/tts")
async def tts(
    text: str = Form(...),
    ref_text: str = Form(None),
    ref_audio: UploadFile = File(...)
):
    # Read binary data from uploaded file
    ref_audio_bytes = await ref_audio.read()

    # Run inference
    result = tts_infer(text, ref_audio_bytes, ref_text)
    audio_bytes = result["audio_bytes"]

    # Return audio stream
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
