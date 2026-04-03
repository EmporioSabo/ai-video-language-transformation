FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CTranslate2 (used by faster-whisper) loads CUDA libs via ctypes and needs
# pip-installed nvidia packages + LD_LIBRARY_PATH to find them.
# See: https://github.com/SYSTRAN/faster-whisper/issues/717
RUN pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Install all Python packages. These may pull in conflicting torch/torchvision
# versions — we fix that in the next step.
RUN pip install --no-cache-dir runpod pydub soundfile scipy
RUN pip install --no-cache-dir faster-whisper
RUN pip install --no-cache-dir pyannote.audio
RUN pip install --no-cache-dir f5-tts

# CRITICAL: Re-pin torch ecosystem AFTER all other installs.
# pyannote.audio and f5-tts drag in incompatible torchvision versions that
# break C++ extensions ("partially initialized module 'torchvision'").
# --force-reinstall overwrites whatever they pulled in.
# --no-deps prevents pip from re-resolving and breaking things again.
# +cu124 suffix ensures we get CUDA 12.4 wheels specifically.
# Versions from: https://pytorch.org/get-started/previous-versions/
# Issue context: https://github.com/pytorch/pytorch/issues/77441
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    torchaudio==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Pre-download Whisper large-v3 weights (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# F5-TTS and pyannote models download on first use at runtime.
# First job will be ~30s slower.

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
