import sys, os
sys.path.append("../../../../..")
sys.path.append("/home/nampv1/projects/tts/tts-ft/ZipVoice")

from .utils import preprocess_ref_audio_text
from zipvoice.bin.infer_zipvoice import generate_sentence

from app.core.config import settings
import json
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from vocos import Vocos
from typing import Optional

_model = None 
_vocoder = None 
_tokenizer = None 
_feature_extractor = None 
_sampling_rate = None 


def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


def _load_tts_model():
    
    
    
    model_dir = settings.MODEL_DIR
    tokens_file = os.path.join(model_dir, "base/tokens.txt")
    config_file = os.path.join(model_dir, "base/model.json")

    checkpoint_name = settings.CHECKPOINT_NAME
    
    tokenizer = EspeakTokenizer(token_file=tokens_file, lang=settings.LANG)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(config_file, "r") as f:
        model_config = json.load(f)


    model = ZipVoice(
        **model_config["model"],
        **tokenizer_config,
    )

    
    load_checkpoint(filename=os.path.join(model_dir, checkpoint_name), model=model, strict=True)

    model = model.to(settings.DEVICE)
    model.eval()

    vocoder = get_vocoder(None)
    vocoder = vocoder.to(settings.DEVICE)
    vocoder.eval()


    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    sampling_rate = model_config["feature"]["sampling_rate"]



    return model, tokenizer, vocoder, feature_extractor, sampling_rate
    
def _ensure_model():
    global _model, _vocoder, _tokenizer, _feature_extractor, _sampling_rate

    if _model is None or _vocoder is None or _tokenizer is None or _feature_extractor is None or _sampling_rate is None:
        _model, _tokenizer, _vocoder, _feature_extractor, _sampling_rate = _load_tts_model()
    
    

def tts_infer(text, ref_audio, ref_text, clip_short=True, show_info=print, device="cuda"):

    _ensure_model()

    # ref_audio = preprocess_ref_audio_text(ref_audio, ref_text, clip_short, show_info, device)
    res_dir = "results"
    os.makedirs(res_dir, exist_ok=True)

    final_wav = generate_sentence(
        "/home/nampv1/projects/tts/tts-ft/ZipVoice/demo/generated.wav",
        text,
        ref_audio,
        ref_text,
        model = _model,
        vocoder=_vocoder,
        tokenizer=_tokenizer,
        feature_extractor=_feature_extractor,
        device=device,
        speed=1.0
    )
    
    return {
        "audio": final_wav,
        "sampling_rate": _sampling_rate,
    }
    
    
