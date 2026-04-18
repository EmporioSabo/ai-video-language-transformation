FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU before other deps (prevents pip pulling CUDA variant)
RUN pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Python dependencies (webapp-only, no GPU packages)
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "webapp/streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.maxUploadSize=2048"]
