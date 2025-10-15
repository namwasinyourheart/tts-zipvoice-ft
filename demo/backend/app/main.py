
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as tts_router

app = FastAPI(title="VnPost TTS API")


# CORS (tùy chỉnh theo môi trường của bạn)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tts_router, prefix="/v1")

@app.get("/")
async def hello():
    return {"message": "Welcome to VnPost TTS API!"}
