FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# cuBLAS for CTranslate2 (faster-whisper). This is the only CUDA pip lib we need:
# - Transcription uses cuBLAS (not cuDNN)
# - Diarization runs on CPU (pyannote RNNs need cuDNN which conflicts with base image)
# - Synthesis uses transformers (no cuDNN)
RUN pip install --no-cache-dir nvidia-cublas-cu12
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install all Python packages
RUN pip install --no-cache-dir runpod pydub soundfile scipy
RUN pip install --no-cache-dir faster-whisper
# Pin pyannote <3.4 (avoids torchcodec dep) + huggingface_hub <0.24
# (pyannote 3.x uses use_auth_token internally, removed in hf_hub 0.24+)
RUN pip install --no-cache-dir "pyannote.audio>=3.1,<3.4" "huggingface_hub>=0.20,<0.24"
RUN pip install --no-cache-dir f5-tts

# Re-pin torch ecosystem to match base image CUDA.
# pyannote/f5-tts pull in incompatible torchvision versions.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    torchaudio==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Pre-download Whisper large-v3 weights (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
