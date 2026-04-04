FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA pip libs:
# - cublas: CTranslate2 (faster-whisper) loads it via ctypes
# - nvrtc: torchcodec (used by pyannote 4.x for audio I/O) links against libnvrtc
RUN pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install all Python packages (pyannote 4.x, latest huggingface_hub — no version pins)
RUN pip install --no-cache-dir runpod pydub soundfile scipy
RUN pip install --no-cache-dir faster-whisper
RUN pip install --no-cache-dir pyannote.audio
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
