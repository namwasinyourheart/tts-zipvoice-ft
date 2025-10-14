FROM nvidia/cuda:12.9.0-devel-ubuntu22.04


RUN apt-get update && apt-get install -y --no-install-recommends \
python3.11 python3-pip ffmpeg libsndfile1 git \
&& rm -rf /var/lib/apt/lists/*

# FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04


# RUN apt-get update && apt-get install -y --no-install-recommends \
# python3.11 python3-pip ffmpeg libsndfile1 git \
# && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN pip install torch==2.8.0+cu129 -f https://download.pytorch.org/whl/torch

RUN pip install torchaudio==2.8.0+cu129 -f https://download.pytorch.org/whl/torchaudio

RUN pip install k2==1.24.4.dev20250807+cuda12.9.torch2.8.0 -f https://k2-fsa.github.io/k2/cuda.html

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

ENV MODEL_NAME=zipvoice
ENV MODEL_DIR="/app/models/"
# ENV CHECKPOINT_NAME="exp/zipvoice_finetune/checkpoint-4000.pt"
ENV CHECKPOINT_NAME="base/iter-96000-avg-2.pt"
ENV VOCODER_DIRNAME="vocoder/charactr--vocos-mel-24khz"
ENV TOKENIZER=espeak
ENV LANG=vi
ENV TOKENS_FILE="base/tokens.txt"
ENV MODEL_CONFIG_FILE="base/model.json"
ENV DEVICE=cuda

EXPOSE 13082

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "13082"]
