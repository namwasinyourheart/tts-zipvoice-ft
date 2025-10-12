#!/usr/bin/env python3
import torch
from zipvoice.bin.train_zipvoice import run, get_parser
from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.utils.common import AttributeDict
from lhotse import CutSet

# ---------------------------
# 1️⃣ Prepare in-memory CutSets
# ---------------------------
# Assuming you already have these:
# train_cut_set, dev_cut_set = prepare_pipeline(...)

# Example: inspect one cut
for cut in train_cut_set[:5]:
    print(f"ID: {cut.id}, Duration: {cut.duration}")
    print("Features path:", cut.features.storage_path)
    features = cut.load_features()
    print("Features shape:", features.shape)

# ---------------------------
# 2️⃣ Setup args
# ---------------------------
parser = get_parser()
args = parser.parse_args(args=[])  # or fill with your default args
args.exp_dir = "exp/zipvoice_inmemory"
args.dataset = "custom"
args.train_manifest = None  # not used
args.dev_manifest = None    # not used
args.tokenizer = ""
args.token_file = "data/tokens_emilia.txt"
args.use_fp16 = False
args.num_epochs = 5
args.world_size = 1

# ---------------------------
# 3️⃣ Setup datamodule
# ---------------------------
datamodule = TtsDataModule(args)

train_dl = datamodule.train_dataloaders(train_cut_set)
valid_dl = datamodule.dev_dataloaders(dev_cut_set)

# ---------------------------
# 4️⃣ Setup params & run training
# ---------------------------
params = AttributeDict()
params.update(vars(args))
params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run single-GPU training
run(rank=0, world_size=1, args=args)
