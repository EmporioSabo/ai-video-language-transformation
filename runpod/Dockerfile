FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# cuBLAS for CTranslate2 (faster-whisper)
RUN pip install --no-cache-dir nvidia-cublas-cu12
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

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

# Remove torchcodec — it needs libnvrtc.so.13 which isn't in this image.
# F5-TTS and other packages fall back to torchaudio/soundfile for audio I/O.
RUN pip uninstall -y torchcodec || true

# Debug: print CUDA/cuDNN versions so we can diagnose version mismatches
RUN python -c "\
import torch; \
print(f'torch={torch.__version__}'); \
print(f'CUDA available={torch.cuda.is_available()}'); \
print(f'cuDNN version={torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else \"NOT AVAILABLE\"}'); \
print(f'cuDNN enabled={torch.backends.cudnn.enabled}'); \
import subprocess; subprocess.run(['find', '/usr', '-name', 'libcudnn*', '-not', '-path', '*/dist-packages/*'], check=False); \
subprocess.run(['find', '/usr/local/lib/python3.11/dist-packages/nvidia', '-name', 'libcudnn*'], check=False); \
"

# Pre-download Whisper large-v3 weights (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

ADD https://raw.githubusercontent.com/emporiosabo/AIVideoLanguageTransformation/main/runpod/handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
