FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA pip packages for CTranslate2 (faster-whisper) and PyTorch:
# - nvidia-cublas-cu12: CTranslate2 loads cuBLAS via ctypes
# - nvidia-cudnn-cu12==9.1.0.70: exact version required by torch 2.4.0+cu124
RUN pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12==9.1.0.70
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install all Python packages. These may pull in conflicting torch/torchvision
# versions — we fix that in the next step.
RUN pip install --no-cache-dir runpod pydub soundfile scipy
RUN pip install --no-cache-dir faster-whisper
RUN pip install --no-cache-dir pyannote.audio
RUN pip install --no-cache-dir f5-tts

# Re-pin torch ecosystem AFTER all other installs.
# pyannote.audio and f5-tts drag in incompatible torchvision versions that
# break C++ extensions ("partially initialized module 'torchvision'").
# --force-reinstall overwrites whatever they pulled in.
# --no-deps prevents pip from re-resolving and breaking things again.
# Versions from: https://pytorch.org/get-started/previous-versions/
# Issue context: https://github.com/pytorch/pytorch/issues/77441
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    torchaudio==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Re-pin cudnn to the exact version torch expects, in case anything above changed it
RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70

# CRITICAL: Fix cuDNN conflict between system cuDNN 8.x (from base image)
# and pip cuDNN 9.x (needed by torch 2.4.0+cu124).
# - Remove system cuDNN 8 to prevent CUDNN_STATUS_NOT_INITIALIZED
# - Register pip-installed cuDNN 9 + cuBLAS with the system linker (ldconfig)
#   so torch and CTranslate2 can find them via dlopen at runtime
RUN find /usr -name "libcudnn*" -not -path "*/dist-packages/*" -delete 2>/dev/null || true && \
    echo "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib" > /etc/ld.so.conf.d/nvidia-pip.conf && \
    echo "/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib" >> /etc/ld.so.conf.d/nvidia-pip.conf && \
    ldconfig

# Verify torch + CUDA actually loads (catches cuDNN issues at build time, not runtime)
RUN python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Pre-download Whisper large-v3 weights (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# F5-TTS and pyannote models download on first use at runtime.

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
