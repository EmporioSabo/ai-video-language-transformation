"""
Core pipeline for Chinese → English video transformation.
Stages: extract → transcribe → translate → synthesize → merge
"""

import io
import subprocess
import time
from pathlib import Path

import requests
from pydub import AudioSegment

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


# ── 1. Audio extraction ───────────────────────────────────────────────────────

def extract_whisper_audio(video_path: Path, output_dir: str) -> Path:
    """Extract 16 kHz mono WAV from video for Whisper transcription."""
    out = Path(output_dir) / "whisper_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(out)],
        check=True, capture_output=True,
    )
    return out


def extract_voice_sample(video_path: Path, output_dir: str, duration: int = 30) -> Path:
    """Extract first N seconds at 44.1 kHz for ElevenLabs voice cloning."""
    out = Path(output_dir) / "voice_sample.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-t", str(duration), "-ar", "44100", "-ac", "1", str(out)],
        check=True, capture_output=True,
    )
    return out


# ── 2. Transcription ──────────────────────────────────────────────────────────

def transcribe(audio_path: Path) -> list:
    """
    Transcribe Chinese audio with faster-whisper (CPU, small model).
    Returns list of dicts: {start, end, text}
    First call downloads the model (~500 MB).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments_iter, _ = model.transcribe(
        str(audio_path), language="zh", beam_size=5, word_timestamps=False
    )
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments_iter
        if s.text.strip()
    ]


# ── 3. Translation ────────────────────────────────────────────────────────────

def translate(segments: list, deepl_key: str) -> list:
    """
    Translate Chinese segments to English with DeepL.
    Returns segments with added 'text_en' field.
    """
    import deepl

    translator = deepl.Translator(deepl_key)
    result = []
    for seg in segments:
        translated = translator.translate_text(
            seg["text"], source_lang="ZH", target_lang="EN-US"
        )
        result.append({**seg, "text_en": translated.text})
    return result


# ── 4. Voice cloning + synthesis ──────────────────────────────────────────────

def synthesize(voice_sample: Path, segments: list, elevenlabs_key: str, output_dir: str) -> Path:
    """
    Clone speaker voice with ElevenLabs, synthesize English audio for each segment,
    and assemble a full-length audio track aligned to the original timestamps.

    Requires ElevenLabs Starter plan ($5/month) for Instant Voice Cloning.
    Uses the ElevenLabs REST API directly (no SDK dependency).
    """
    headers = {"xi-api-key": elevenlabs_key}

    # Clone voice from sample
    voice_name = f"TempVoice_{int(time.time())}"
    with open(voice_sample, "rb") as f:
        resp = requests.post(
            f"{ELEVENLABS_BASE}/voices/add",
            headers=headers,
            data={"name": voice_name, "description": "Temporary voice — will be deleted"},
            files={"files": (voice_sample.name, f, "audio/wav")},
        )
    resp.raise_for_status()
    voice_id = resp.json()["voice_id"]

    # Silent base track covering the full video duration
    total_ms = int(segments[-1]["end"] * 1000) + 2000  # +2 s padding
    track = AudioSegment.silent(duration=total_ms)

    try:
        for seg in segments:
            text = seg.get("text_en", "").strip()
            if not text:
                continue

            # Generate TTS audio
            tts_resp = requests.post(
                f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "output_format": "mp3_44100_128",
                },
            )
            tts_resp.raise_for_status()
            seg_audio = AudioSegment.from_mp3(io.BytesIO(tts_resp.content))

            # Speed up if synthesized audio overflows the time slot (max 1.2×)
            target_ms = int((seg["end"] - seg["start"]) * 1000)
            if len(seg_audio) > target_ms:
                factor = min(len(seg_audio) / target_ms, 1.20)
                seg_audio = seg_audio.speedup(playback_speed=factor)

            # Overlay segment at its original timestamp
            start_ms = int(seg["start"] * 1000)
            track = track.overlay(seg_audio, position=start_ms)

    finally:
        # Always clean up the cloned voice from ElevenLabs account
        requests.delete(f"{ELEVENLABS_BASE}/voices/{voice_id}", headers=headers)

    out_path = Path(output_dir) / "english_audio.wav"
    track.export(str(out_path), format="wav")
    return out_path


# ── 5. Video merge ────────────────────────────────────────────────────────────

def merge(video_path: Path, audio_path: Path, output_dir: str) -> Path:
    """Replace original audio track with the new English audio using FFmpeg."""
    out_path = Path(output_dir) / f"{video_path.stem}_EN.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",          # copy video stream (no re-encoding)
            "-map", "0:v:0",         # video from first input
            "-map", "1:a:0",         # audio from second input
            "-shortest",
            str(out_path),
        ],
        check=True, capture_output=True,
    )
    return out_path
