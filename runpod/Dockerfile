FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    faster-whisper>=1.0.0 \
    pyannote.audio \
    f5-tts>=0.3.0 \
    soundfile>=0.12.0 \
    pydub>=0.25.1 \
    scipy \
    numpy

# Pre-download model weights to avoid cold-start delays
# Whisper large-v3 (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# F5-TTS model weights
RUN python -c "from f5_tts.api import F5TTS; F5TTS()"

# Note: pyannote models require HF_TOKEN at runtime and cannot be pre-downloaded
# without accepting the model license. They'll be cached after first use.

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
