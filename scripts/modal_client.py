"""Modal GPU client — replaces runpod_client.py for transcribe + synthesize."""

import base64
import subprocess
import tempfile
import time
from pathlib import Path


def _get_app():
    import modal
    return modal.Function.lookup("ai-video-language-transformation", "transcribe"), \
           modal.Function.lookup("ai-video-language-transformation", "synthesize")


def transcribe(audio_path: Path, language: str = "zh") -> dict:
    """Submit transcription to Modal and return result."""
    import modal

    # Compress to MP3 32kbps mono — keeps payload small for any video length
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-b:a", "32k", tmp_path],
        check=True, capture_output=True,
    )
    audio_b64 = base64.b64encode(Path(tmp_path).read_bytes()).decode()
    Path(tmp_path).unlink(missing_ok=True)

    fn = modal.Function.lookup("ai-video-language-transformation", "transcribe")
    return fn.remote(audio_b64, language)


def synthesize(segments: list, voice_refs_b64: dict) -> dict:
    """Submit TTS synthesis to Modal and return result with tts_zip_b64."""
    import modal

    fn = modal.Function.lookup("ai-video-language-transformation", "synthesize")
    return fn.remote(segments, voice_refs_b64)


def get_daily_usage() -> dict:
    """Stub — Modal bills per second; no daily job cap needed."""
    today = time.strftime("%Y-%m-%d")
    return {
        "date": today,
        "job_count": 0,
        "total_gpu_seconds": 0.0,
        "limit": 9999,
    }
