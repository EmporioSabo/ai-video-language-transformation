"""File-based job manager for tracking video processing pipelines."""

import json
import shutil
import time
import uuid
from pathlib import Path

from config import ROOT

JOBS_DIR = ROOT / "data" / "jobs"


def create_job(video_bytes: bytes, filename: str, language: str = "zh",
               tts_model: str = "voxtral") -> str:
    """Create a new job directory and save the uploaded video. Returns job_id.

    Args:
        video_bytes: Raw video file bytes.
        filename: Original filename (used for output naming).
        language: BCP-47 source language code.
        tts_model: TTS engine — "voxtral" (API) or "f5tts" (RunPod GPU).
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save video
    video_path = job_dir / filename
    video_path.write_bytes(video_bytes)

    # Initialize status
    status = {
        "job_id": job_id,
        "filename": filename,
        "language": language,
        "tts_model": tts_model,
        "stage": "created",
        "progress": 0,
        "error": None,
        "result_path": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _save_status(job_id, status)
    return job_id


def get_job_status(job_id: str) -> dict:
    """Read current job status."""
    status_file = JOBS_DIR / job_id / "status.json"
    if not status_file.exists():
        return {"error": f"Job {job_id} not found"}
    return json.loads(status_file.read_text())


def update_job_status(job_id: str, stage: str = None, progress: int = None,
                      error: str = None, result_path: str = None):
    """Update job status fields."""
    status = get_job_status(job_id)
    if "error" in status and status.get("job_id") is None:
        return

    if stage is not None:
        status["stage"] = stage
    if progress is not None:
        status["progress"] = progress
    if error is not None:
        status["error"] = error
        status["stage"] = "failed"
    if result_path is not None:
        status["result_path"] = result_path
    status["updated_at"] = time.time()
    _save_status(job_id, status)


def get_job_dir(job_id: str) -> Path:
    """Return the job's working directory."""
    return JOBS_DIR / job_id


def list_jobs() -> list:
    """List all jobs sorted by creation time (newest first)."""
    if not JOBS_DIR.exists():
        return []
    jobs = []
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and (d / "status.json").exists():
            jobs.append(get_job_status(d.name))
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return jobs


def cleanup_old_jobs(max_age_days: int = 7):
    """Delete jobs older than max_age_days."""
    if not JOBS_DIR.exists():
        return 0
    cutoff = time.time() - (max_age_days * 86400)
    removed = 0
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        status = get_job_status(d.name)
        if status.get("created_at", 0) < cutoff:
            shutil.rmtree(d)
            removed += 1
    return removed


def _save_status(job_id: str, status: dict):
    status_file = JOBS_DIR / job_id / "status.json"
    status_file.write_text(json.dumps(status, indent=2))
