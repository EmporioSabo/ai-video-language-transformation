FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Reinstall torchvision to match the base image's PyTorch version
# The base image has PyTorch 2.4.0+cu124, so we need a matching torchvision
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Install packages in stages to isolate conflicts
RUN pip install --no-cache-dir runpod pydub soundfile scipy
RUN pip install --no-cache-dir faster-whisper
RUN pip install --no-cache-dir pyannote.audio
RUN pip install --no-cache-dir f5-tts

# Install NVIDIA CUDA pip packages so CTranslate2 can find cuBLAS and cuDNN at runtime.
# CTranslate2 (used by faster-whisper) loads these via ctypes and looks for pip-installed
# nvidia packages or LD_LIBRARY_PATH — it does NOT search /usr/local/cuda/lib64 by default.
RUN pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12

# Set LD_LIBRARY_PATH so CTranslate2 can find libcublas.so.12 and libcudnn.so at runtime.
# This resolves: "RuntimeError: Library libcublas.so.12 is not found or cannot be loaded"
# See: https://github.com/SYSTRAN/faster-whisper/issues/717
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Pre-download Whisper large-v3 weights (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# F5-TTS and pyannote models will be downloaded on first use at runtime.
# F5-TTS has import-time dependency issues during Docker build (no GPU context),
# but works fine at runtime. First job will be slower (~30s extra download).

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
