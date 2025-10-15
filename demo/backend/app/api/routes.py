
# curl -X POST "https://ai.vnpost.vn/voiceai/tts/v1/synthesize" \
#   -F "text=Xin chào, đây là ví dụ TTS." \
#   -F "voice=Lại Văn Sâm" \
#   --output output_synthesize.wav



# curl -X POST "https://ai.vnpost.vn/voiceai/tts/v1/clone" \
#   -F "text=Chào bạn, đây là giọng clone từ audio của bạn." \
#   -F "reference_audio=@/path/to/your_reference_audio.wav" \
#   --output output_clone.wav



from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
from app.services.infer import tts_infer_voice, tts_infer_clone, DEFAULT_VOICE, _voice_map

router = APIRouter(tags=["Text-to-Speech"])


@router.get("/voices")
async def get_available_voices():
    """
    Get list of available voices with metadata and file path.
    """
    
    for voice in _voice_map.values():
        del voice["path"]

    return JSONResponse(content=list(_voice_map.values()))

@router.post("/synthesize")
async def tts_synthesize(
    text: str = Form(..., description="Text to synthesize"),
    voice: str = Form(DEFAULT_VOICE, description="Predefined voice key")
):
    """
    Synthesize speech using a Vietnamese voice.
    """
    voice_names = [v["name"] for v in _voice_map.values()]
    if voice not in voice_names:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice '{voice}'. Supported voices: {', '.join(voice_names)}"
        )

    voice_id = next((v["id"] for v in _voice_map.values() if v["name"] == voice), None)
    

    result = tts_infer_voice(text=text, voice=voice_id)
    return StreamingResponse(io.BytesIO(result["audio_bytes"]), media_type="audio/wav")

@router.post("/clone")
async def tts_clone(
    text: str = Form(..., description="Text to synthesize with cloned voice"),
    reference_audio: UploadFile = File(..., description="Reference voice sample (.wav or .mp3)")
):
    """
    Clone a custom voice from uploaded reference audio and generate new speech.
    """
    filename = reference_audio.filename.lower()
    if not (filename.endswith(".wav") or filename.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")

    ref_audio_bytes = await reference_audio.read()
    result = tts_infer_clone(text=text, ref_audio=ref_audio_bytes)
    return StreamingResponse(io.BytesIO(result["audio_bytes"]), media_type="audio/wav")
