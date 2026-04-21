"""
Server-side pipeline orchestrator.

Runs the full video → English dubbing pipeline using:
- Local CPU for: audio extraction, translation, alignment, merging, subtitles
- Modal serverless GPU for: transcription (Whisper, all languages)
- Local CPU for: diarization (pyannote)
- Voxtral API for: TTS synthesis (voice cloning, multilingual, no GPU needed)

Designed to run in a background thread, updating job status as it progresses.
"""

import base64
import json
import os
import threading
from pathlib import Path

import job_manager
import modal_client as gpu_client
from config import (
    DEEPL_API_KEY, GEMINI_API_KEY,
    TARGET_LANGUAGE, OUTPUT_VIDEO_SUFFIX,
)
from extract_audio import extract_audio, extract_voice_reference
from translate import translate_with_deepl as translate_with_deepl_zh, review_with_gemini as review_with_gemini_zh
from translate_multilang import (
    translate_with_deepl as translate_with_deepl_ml,
    review_with_gemini as review_with_gemini_ml,
)
from synthesize_voxtral import synthesize_segments as synthesize_voxtral
from align_audio import align_segments
from merge_video import merge_audio_video
from generate_subtitles import generate_srt
from diarize_local import diarize as diarize_local
from transcribe_voxtral import VOXTRAL_LANGUAGES


STAGES = [
    ("extract", "Extracting audio", 10),
    ("transcribe", "Transcribing audio (GPU)", 25),
    ("diarize", "Identifying speakers", 40),
    ("translate", "Translating to English", 55),
    ("synthesize", "Synthesizing English speech", 75),
    ("align", "Aligning audio segments", 85),
    ("merge", "Merging audio into video", 92),
    ("subtitles", "Generating subtitles", 97),
    ("complete", "Done", 100),
]

# Reverse lookup: BCP-47 → display name (for Gemini prompts)
_LANG_NAMES = {v: k for k, v in VOXTRAL_LANGUAGES.items()}
_LANG_NAMES["zh"] = "Chinese"


def run_pipeline(job_id: str):
    """Run the full pipeline for a job. Call from a background thread."""
    job_dir = job_manager.get_job_dir(job_id)
    status = job_manager.get_job_status(job_id)
    video_path = job_dir / status["filename"]
    language = status.get("language", "zh")
    tts_model = status.get("tts_model", "voxtral")
    lang_name = _LANG_NAMES.get(language, language)

    try:
        # ── 1. Extract audio ──────────────────────────────────────────────
        job_manager.update_job_status(job_id, stage="extract", progress=5)
        audio_dir = job_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        whisper_audio = extract_audio(video_path, audio_dir)
        job_manager.update_job_status(job_id, stage="extract", progress=10)

        # ── 2. Transcribe (Modal GPU — Whisper supports all languages) ───
        job_manager.update_job_status(job_id, stage="transcribe", progress=15)
        result = gpu_client.transcribe(whisper_audio, language=language)
        segments = result["segments"]

        # For non-Chinese, rename text_zh → text_src for compatibility
        if language != "zh":
            for seg in segments:
                seg["text_src"] = seg.pop("text_zh", "")

        transcript_path = job_dir / f"transcript_{language}.json"
        transcript_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        job_manager.update_job_status(job_id, stage="transcribe", progress=25)

        # ── 3. Diarize (local CPU) ───────────────────────────────────────
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

        if language == "zh":
            segments = translate_with_deepl_zh(segments)
            if GEMINI_API_KEY:
                segments = review_with_gemini_zh(segments)
            else:
                for seg in segments:
                    seg["text_en"] = seg["text_en_deepl"]
        else:
            segments = translate_with_deepl_ml(segments, language)
            if GEMINI_API_KEY:
                segments = review_with_gemini_ml(segments, lang_name)
            else:
                for seg in segments:
                    seg["text_en"] = seg.get("text_en_deepl", seg.get("text_src", ""))

        translation_path = job_dir / "translation_en.json"
        translation_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        job_manager.update_job_status(job_id, stage="translate", progress=55)

        # ── 5. Synthesize TTS ─────────────────────────────────────────────
        job_manager.update_job_status(job_id, stage="synthesize", progress=60)
        tts_dir = job_dir / "tts_segments"

        if tts_model == "f5tts":
            # F5-TTS via RunPod GPU (voice cloning, works cross-lingually)
            voice_refs_for_runpod = {}
            for fname, b64_data in voice_refs_b64.items():
                voice_refs_for_runpod[fname] = b64_data
            result = gpu_client.synthesize(segments, voice_refs_for_runpod)
            tts_dir.mkdir(parents=True, exist_ok=True)

            import io
            import zipfile
            tts_zip_b64 = result.get("tts_zip_b64", "")
            if tts_zip_b64:
                zip_bytes = base64.b64decode(tts_zip_b64)
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    for name in zf.namelist():
                        (tts_dir / name).write_bytes(zf.read(name))

            segments = result.get("segments", segments)
        else:
            # Voxtral API (no GPU needed, voice cloning via ref_audio)
            segments = synthesize_voxtral(segments, voice_ref_dir, tts_dir)

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
