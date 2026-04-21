"""RunPod serverless API client with timeout safety and spending controls."""

import base64
import json
import time
from pathlib import Path

import requests

from config import ROOT

# ── Configuration ─────────────────────────────────────────────────────────────

RUNPOD_API_KEY = ""
RUNPOD_ENDPOINT_ID = ""

# Safety limits
MAX_JOB_SECONDS = 1800  # 30 minutes max per job
POLL_INTERVAL = 10  # seconds between status checks
MAX_DAILY_JOBS = 40  # max jobs per day

# Usage tracking file
USAGE_FILE = ROOT / "data" / "runpod_usage.json"


def _load_config():
    """Load RunPod config from environment."""
    import os
    global RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
    RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")


def _api_url(path=""):
    return f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}{path}"


def _headers():
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


# ── Spending controls ─────────────────────────────────────────────────────────

def _load_usage() -> dict:
    if USAGE_FILE.exists():
        return json.loads(USAGE_FILE.read_text())
    return {"jobs": []}


def _save_usage(usage: dict):
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(usage, indent=2))


def _check_daily_limit():
    """Raise if daily job limit exceeded."""
    usage = _load_usage()
    today = time.strftime("%Y-%m-%d")
    today_jobs = [j for j in usage["jobs"] if j.get("date") == today]
    if len(today_jobs) >= MAX_DAILY_JOBS:
        raise RuntimeError(
            f"Daily job limit reached ({MAX_DAILY_JOBS} jobs). "
            f"Try again tomorrow or increase MAX_DAILY_JOBS."
        )


def _record_job(job_id: str, stage: str, duration_sec: float):
    """Record a completed job for usage tracking."""
    usage = _load_usage()
    usage["jobs"].append({
        "job_id": job_id,
        "stage": stage,
        "date": time.strftime("%Y-%m-%d"),
        "duration_sec": round(duration_sec, 1),
    })
    _save_usage(usage)


# ── Core API ──────────────────────────────────────────────────────────────────

def submit_job(payload: dict) -> str:
    """Submit a job to RunPod serverless endpoint. Returns job ID."""
    _load_config()
    _check_daily_limit()

    resp = requests.post(
        _api_url("/run"),
        headers=_headers(),
        json={"input": payload},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


def poll_job(job_id: str) -> dict:
    """Poll until job completes or times out. Returns result dict."""
    _load_config()
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > MAX_JOB_SECONDS:
            # Cancel the job
            cancel_job(job_id)
            raise TimeoutError(
                f"Job {job_id} exceeded {MAX_JOB_SECONDS}s timeout. Cancelled."
            )

        resp = requests.get(_api_url(f"/status/{job_id}"), headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")

        if status == "COMPLETED":
            _record_job(job_id, data.get("input", {}).get("stage", "unknown"), elapsed)
            return data["output"]
        elif status == "FAILED":
            raise RuntimeError(f"RunPod job {job_id} failed: {data.get('error', 'unknown')}")
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            time.sleep(POLL_INTERVAL)
        else:
            raise RuntimeError(f"Unexpected job status: {status}")


def cancel_job(job_id: str):
    """Cancel a running job."""
    _load_config()
    try:
        requests.post(_api_url(f"/cancel/{job_id}"), headers=_headers())
    except Exception:
        pass


# ── High-level stage functions ────────────────────────────────────────────────

def transcribe(audio_path: Path, language: str = "zh") -> dict:
    """Submit transcription job and return result."""
    import subprocess, tempfile
    # Encode to MP3 32kbps mono — keeps payload ~1-2MB regardless of video length
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-b:a", "32k", tmp_path],
        check=True, capture_output=True,
    )
    audio_b64 = base64.b64encode(Path(tmp_path).read_bytes()).decode()
    Path(tmp_path).unlink(missing_ok=True)
    job_id = submit_job({
        "stage": "transcribe",
        "audio_b64": audio_b64,
        "language": language,
    })
    return poll_job(job_id)


def diarize(audio_path: Path, segments: list, num_speakers: int = 2, hf_token: str = "") -> dict:
    """Submit diarization job and return result with speaker labels + voice refs."""
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
    job_id = submit_job({
        "stage": "diarize",
        "audio_b64": audio_b64,
        "segments": segments,
        "num_speakers": num_speakers,
        "hf_token": hf_token,
    })
    return poll_job(job_id)


def synthesize(segments: list, voice_refs_b64: dict) -> dict:
    """Submit TTS synthesis job and return result with TTS ZIP."""
    job_id = submit_job({
        "stage": "synthesize",
        "segments": segments,
        "voice_refs": voice_refs_b64,
    })
    return poll_job(job_id)


def get_daily_usage() -> dict:
    """Return today's usage stats."""
    usage = _load_usage()
    today = time.strftime("%Y-%m-%d")
    today_jobs = [j for j in usage["jobs"] if j.get("date") == today]
    total_seconds = sum(j.get("duration_sec", 0) for j in today_jobs)
    return {
        "date": today,
        "job_count": len(today_jobs),
        "total_gpu_seconds": round(total_seconds, 1),
        "limit": MAX_DAILY_JOBS,
    }
