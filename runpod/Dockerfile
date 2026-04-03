FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Fix torchvision compatibility with base image PyTorch version
RUN pip install --no-cache-dir --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu124

# Install packages in stages to isolate conflicts
RUN pip install --no-cache-dir runpod pydub soundfile scipy
RUN pip install --no-cache-dir faster-whisper
RUN pip install --no-cache-dir pyannote.audio
RUN pip install --no-cache-dir f5-tts

# Pre-download Whisper large-v3 weights (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# F5-TTS and pyannote models will be downloaded on first use at runtime.
# F5-TTS has import-time dependency issues during Docker build (no GPU context),
# but works fine at runtime. First job will be slower (~30s extra download).

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
