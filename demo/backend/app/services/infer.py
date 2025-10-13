import sys, os
# sys.path.append("../../../../..")
# sys.path.append("/home/nampv1/projects/tts/tts-ft/ZipVoice")
# sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../..")))


from zipvoice.bin.infer_zipvoice import generate_sentence

from app.core.config import settings
import json
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from vocos import Vocos
from typing import Optional

from .utils import transcribe
from .log_utils import setup_logger

import torch

logger = setup_logger(__name__)

_model = None 
_vocoder = None 
_tokenizer = None 
_feature_extractor = None 
_sampling_rate = None 


print(settings.dict())



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
    
    logger.info(f"Loading tokenizer...")
    tokenizer = EspeakTokenizer(token_file=tokens_file, lang=settings.LANG)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    
    with open(config_file, "r") as f:
        model_config = json.load(f)


    model = ZipVoice(
        **model_config["model"],
        **tokenizer_config,
    )

    logger.info(f"Loading model...")
    load_checkpoint(filename=os.path.join(model_dir, checkpoint_name), model=model, strict=True)

    model = model.to(settings.DEVICE)
    model.eval()

    logger.info(f"Loading vocoder...")
    vocoder_local_path = os.path.join(model_dir, settings.VOCODER_DIRNAME)

    vocoder = get_vocoder(vocoder_local_path)
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
    
    

def tts_infer(text, ref_audio, ref_text, clip_short=True, show_info=print, device=settings.DEVICE):

    _ensure_model()

    # ref_audio = preprocess_ref_audio_text(ref_audio, ref_text, clip_short, show_info, device)

    # if no ref_text provided, transcribe it from ref_audio
    logger.info("Getting ref text from ref audio...")
    if not ref_text:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_audio.write(ref_audio)
            tmp_audio_path = tmp_audio.name
        ref_text = transcribe(tmp_audio_path)["text"]
        # logger.info(f"Ref text: {ref_text}")

    
    logger.info(f"Text: {text}")
    # logger.info(f"Ref audio: {ref_audio}")
    logger.info(f"Ref text: {ref_text}")

    logger.info("Generating audio...")
    metrics, final_wav, sampling_rate = generate_sentence(
        "../../responses/generated.wav",
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
    logger.info(f"Generated audio: {final_wav}")
    
    return {
        "audio": final_wav,
        "sampling_rate": sampling_rate,
    }
    
    
