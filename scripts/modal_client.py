"""Modal GPU client — calls Modal web endpoints via plain HTTP (no modal package needed)."""

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests

# Set these as HF Spaces secrets: MODAL_TRANSCRIBE_URL, MODAL_SYNTHESIZE_URL
# Obtain URLs by running: modal deploy modal_app.py  (printed at the end)
TRANSCRIBE_URL = os.getenv("MODAL_TRANSCRIBE_URL", "")
SYNTHESIZE_URL = os.getenv("MODAL_SYNTHESIZE_URL", "")

TIMEOUT = 1800  # 30 min max per call


def transcribe(audio_path: Path, language: str = "zh") -> dict:
    """Compress audio and call Modal transcribe web endpoint."""
    if not TRANSCRIBE_URL:
        raise RuntimeError("MODAL_TRANSCRIBE_URL secret not set on HF Spaces.")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-b:a", "32k", tmp_path],
        check=True, capture_output=True,
    )
    audio_b64 = base64.b64encode(Path(tmp_path).read_bytes()).decode()
    Path(tmp_path).unlink(missing_ok=True)

    resp = requests.post(
        TRANSCRIBE_URL,
        json={"audio_b64": audio_b64, "language": language},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def synthesize(segments: list, voice_refs_b64: dict) -> dict:
    """Call Modal synthesize web endpoint."""
    if not SYNTHESIZE_URL:
        raise RuntimeError("MODAL_SYNTHESIZE_URL secret not set on HF Spaces.")

    resp = requests.post(
        SYNTHESIZE_URL,
        json={"segments": segments, "voice_refs": voice_refs_b64},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_daily_usage() -> dict:
    """Stub — Modal bills per second; no daily job cap needed."""
    today = time.strftime("%Y-%m-%d")
    return {
        "date": today,
        "job_count": 0,
        "total_gpu_seconds": 0.0,
        "limit": 9999,
    }
