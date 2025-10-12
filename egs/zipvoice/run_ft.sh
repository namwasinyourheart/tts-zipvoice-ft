# #!/bin/bash
export PYTHONPATH=/home/nampv1/projects/tts/tts-ft/ZipVoice/zipvoice/bin:$PYTHONPATH

hf_raw_data_dir="/media/nampv1/hdd/data/TTS-viVoice-1017h/raw/hf/subset_887"
tokenizer="espeak"
lang="vi"
sampling_rate=24000
exp_dir="/media/nampv1/hdd/data/TTS-viVoice-1017h/exp/zipvoice_finetune"
model_dir="/media/nampv1/hdd/models/zipvoice/zipvoice"
nj=1

python3 - <<EOF
from prepare_pipeline_in_memory import prepare_pipeline
from zipvoice.bin import train_zipvoice

# Load in-memory CutSets
cut_sets = prepare_pipeline(
    hf_raw_data_dir="${hf_raw_data_dir}",
    subsets=("train","dev"),
    tokenizer="${tokenizer}",
    lang="${lang}",
    sampling_rate=${sampling_rate}, 
    num_jobs=${nj}
)

train_cut_set = cut_sets["train"]
dev_cut_set = cut_sets.get("dev", None)


# Replace the token-conversion loop in your heredoc with this (use .map to produce new CutSets)
from zipvoice.tokenizer.tokenizer import EspeakTokenizer

tokenizer_obj = EspeakTokenizer(token_file=f"${model_dir}/tokens.txt", lang="vi")

def convert_cut_tokens(cut):
    # assume single supervision per cut
    sup = cut.supervisions[0]
    # 1) if sup.tokens already ints -> keep
    if hasattr(sup, "tokens") and isinstance(sup.tokens, list) and len(sup.tokens) > 0 and isinstance(sup.tokens[0], int):
        return cut
    # 2) if sup.tokens exists and are str (phonemes) -> convert
    if hasattr(sup, "tokens") and isinstance(sup.tokens, list) and len(sup.tokens) > 0 and isinstance(sup.tokens[0], str):
        ids = tokenizer_obj.tokens_to_token_ids([sup.tokens])[0]
        sup.tokens = ids
        return cut
    # 3) if tokens stored in sup.custom['tokens'] as str list -> convert and move to sup.tokens
    if getattr(sup, "custom", None) and "tokens" in sup.custom:
        t = sup.custom.pop("tokens")
        if isinstance(t, list) and len(t) > 0 and isinstance(t[0], str):
            ids = tokenizer_obj.tokens_to_token_ids([t])[0]
            sup.tokens = ids
            return cut
        # if custom['tokens'] already ids
        if isinstance(t, list) and len(t) > 0 and isinstance(t[0], int):
            sup.tokens = t
            return cut
    # 4) fallback: compute ids from text
    ids = tokenizer_obj.texts_to_token_ids([sup.text])[0]
    sup.tokens = ids
    return cut

train_cut_set = train_cut_set.map(convert_cut_tokens)
if dev_cut_set is not None:
    dev_cut_set = dev_cut_set.map(convert_cut_tokens)



# Fine-tune in-memory
import argparse
parser = train_zipvoice.get_parser()
args = parser.parse_args([
    "--world-size", "1",
    "--use-fp16", "1",
    "--finetune", "1",
    "--base-lr", "0.0001",
    "--num-iters", "10000",
    "--save-every-n", "1000",
    # "--max-duration", "500",
    "--max-len", "20",
    # "--model-config", "conf/zipvoice_base.json",
    "--model-config", "${model_dir}/model.json",
    "--checkpoint", "${model_dir}/model.pt",
    "--tokenizer", "${tokenizer}",
    "--lang", "${lang}",
    "--token-file", "${model_dir}/tokens.txt",
    "--dataset", "custom",
    "--exp-dir", "${exp_dir}"
])

from pathlib import Path
args.exp_dir = Path(args.exp_dir) 

args.on_the_fly_feats = False
args.return_cuts = False
args.bucketing_sampler = False
args.max_duration = 10
args.shuffle = False
args.num_workers = 1


train_zipvoice.run_in_memory(rank=0, world_size=1, args=args,
                             train_cut_set=train_cut_set,
                             dev_cut_set=dev_cut_set)
EOF
