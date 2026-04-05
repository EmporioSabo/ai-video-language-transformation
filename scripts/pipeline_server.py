"""
Server-side pipeline orchestrator.

Runs the full Chinese → English video transformation pipeline using:
- Local CPU for: audio extraction, translation, alignment, merging, subtitles
- RunPod serverless GPU for: transcription, TTS synthesis
- Local CPU for: diarization (pyannote — avoids Docker/CUDA conflicts on RunPod)

Designed to run in a background thread, updating job status as it progresses.
"""

import base64
import json
import os
import threading
import zipfile
from pathlib import Path

import job_manager
import runpod_client
from config import (
    DEEPL_API_KEY, GEMINI_API_KEY, WHISPER_LANGUAGE,
    TARGET_LANGUAGE, OUTPUT_VIDEO_SUFFIX,
)
from extract_audio import extract_audio, extract_voice_reference
from translate import translate_with_deepl, review_with_gemini
from align_audio import align_segments
from merge_video import merge_audio_video
from generate_subtitles import generate_srt
from diarize_local import diarize as diarize_local


STAGES = [
    ("extract", "Extracting audio", 10),
    ("transcribe", "Transcribing Chinese audio (GPU)", 25),
    ("diarize", "Identifying speakers (GPU)", 40),
    ("translate", "Translating to English", 55),
    ("synthesize", "Synthesizing English speech (GPU)", 75),
    ("align", "Aligning audio segments", 85),
    ("merge", "Merging audio into video", 92),
    ("subtitles", "Generating subtitles", 97),
    ("complete", "Done", 100),
]


def run_pipeline(job_id: str):
    """Run the full pipeline for a job. Call from a background thread."""
    job_dir = job_manager.get_job_dir(job_id)
    status = job_manager.get_job_status(job_id)
    video_path = job_dir / status["filename"]

    try:
        # ── 1. Extract audio ──────────────────────────────────────────────
        job_manager.update_job_status(job_id, stage="extract", progress=5)
        audio_dir = job_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        whisper_audio = extract_audio(video_path, audio_dir)
        job_manager.update_job_status(job_id, stage="extract", progress=10)

        # ── 2. Transcribe (RunPod GPU) ────────────────────────────────────
        job_manager.update_job_status(job_id, stage="transcribe", progress=15)
        result = runpod_client.transcribe(whisper_audio, language=WHISPER_LANGUAGE)
        segments = result["segments"]

        # Save transcript
        transcript_path = job_dir / "transcript_zh.json"
        transcript_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        job_manager.update_job_status(job_id, stage="transcribe", progress=25)

        # ── 3. Diarize (local CPU — avoids RunPod Docker/CUDA issues) ────
        job_manager.update_job_status(job_id, stage="diarize", progress=30)
        hf_token = os.getenv("HF_TOKEN", "")
        result = diarize_local(str(whisper_audio), segments, hf_token=hf_token)
        segments = result["segments"]
        voice_refs_b64 = result["voice_refs"]

        # Save voice references to disk
        voice_ref_dir = job_dir / "voice_refs"
        voice_ref_dir.mkdir(exist_ok=True)
        for filename, b64_data in voice_refs_b64.items():
            (voice_ref_dir / filename).write_bytes(base64.b64decode(b64_data))

        job_manager.update_job_status(job_id, stage="diarize", progress=40)

        # ── 4. Translate (CPU, DeepL + Gemini) ────────────────────────────
        job_manager.update_job_status(job_id, stage="translate", progress=45)
        segments = translate_with_deepl(segments)

        if GEMINI_API_KEY:
            segments = review_with_gemini(segments)
        else:
            for seg in segments:
                seg["text_en"] = seg["text_en_deepl"]

        # Save translation
        translation_path = job_dir / "translation_en.json"
        translation_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        job_manager.update_job_status(job_id, stage="translate", progress=55)

        # ── 5. Synthesize TTS (RunPod GPU) ────────────────────────────────
        job_manager.update_job_status(job_id, stage="synthesize", progress=60)
        result = runpod_client.synthesize(segments, voice_refs_b64)
        segments = result["segments"]

        # Extract TTS ZIP to disk
        tts_dir = job_dir / "tts_segments"
        tts_dir.mkdir(exist_ok=True)
        import io
        tts_zip_bytes = base64.b64decode(result["tts_zip_b64"])
        with zipfile.ZipFile(io.BytesIO(tts_zip_bytes)) as zf:
            zf.extractall(tts_dir)

        # Update translation with TTS metadata
        translation_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        job_manager.update_job_status(job_id, stage="synthesize", progress=75)

        # ── 6. Align (CPU) ────────────────────────────────────────────────
        job_manager.update_job_status(job_id, stage="align", progress=80)
        aligned_path = job_dir / "aligned_en.wav"
        align_segments(translation_path, tts_dir, aligned_path)
        job_manager.update_job_status(job_id, stage="align", progress=85)

        # ── 7. Merge (CPU) ────────────────────────────────────────────────
        job_manager.update_job_status(job_id, stage="merge", progress=90)
        output_path = job_dir / f"{video_path.stem}{OUTPUT_VIDEO_SUFFIX}.mp4"
        merge_audio_video(video_path, aligned_path, output_path)
        job_manager.update_job_status(job_id, stage="merge", progress=92)

        # ── 8. Subtitles (CPU) ────────────────────────────────────────────
        job_manager.update_job_status(job_id, stage="subtitles", progress=95)
        srt_path = job_dir / f"{video_path.stem}_en.srt"
        generate_srt(translation_path, srt_path, bilingual=True)
        job_manager.update_job_status(job_id, stage="subtitles", progress=97)

        # ── Done ──────────────────────────────────────────────────────────
        job_manager.update_job_status(
            job_id, stage="complete", progress=100,
            result_path=str(output_path),
        )

    except Exception as e:
        job_manager.update_job_status(job_id, error=str(e))


def start_pipeline_async(job_id: str) -> threading.Thread:
    """Launch the pipeline in a background thread."""
    thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()
    return thread
