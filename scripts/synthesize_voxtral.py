"""Synthesize English speech using Voxtral TTS API (Mistral).

Replaces F5-TTS on RunPod GPU with an API call — no GPU needed for synthesis.
Supports voice cloning with 2-3s of reference audio via ref_audio parameter.

Output format is compatible with align_audio.py (per-segment WAV files).
"""

import base64
import io
import os
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# Voxtral TTS supports: en, fr, de, es, nl, pt, it, hi, ar
VOXTRAL_TTS_LANGUAGES = {
    "en", "fr", "de", "es", "nl", "pt", "it", "hi", "ar",
}


def _get_client():
    from mistralai.client import Mistral

    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY not set. Add it to your .env file.\n"
            "Get a key at: https://console.mistral.ai/"
        )
    return Mistral(api_key=MISTRAL_API_KEY)


def synthesize_segment(
    text: str,
    voice_ref_path: str | Path,
    model: str = "voxtral-mini-tts-latest",
) -> tuple[bytes, int]:
    """Synthesize a single text segment with voice cloning.

    Args:
        text: English text to synthesize.
        voice_ref_path: Path to reference WAV for voice cloning.
        model: Voxtral TTS model name.

    Returns:
        Tuple of (wav_bytes, sample_rate).
    """
    client = _get_client()

    # Read reference audio for voice cloning (base64-encoded)
    ref_bytes = Path(voice_ref_path).read_bytes()
    ref_b64 = base64.b64encode(ref_bytes).decode()

    response = client.audio.speech.complete(
        model=model,
        input=text,
        ref_audio=ref_b64,
        response_format="wav",
    )

    # Response contains audio_data as base64 string
    if hasattr(response, "audio_data") and response.audio_data:
        wav_bytes = base64.b64decode(response.audio_data)
    elif hasattr(response, "data"):
        wav_bytes = response.data if isinstance(response.data, bytes) else base64.b64decode(response.data)
    else:
        raise RuntimeError(f"Unexpected response format: {type(response).__name__}")

    data, sr = sf.read(io.BytesIO(wav_bytes))
    return wav_bytes, sr


def synthesize_segments(
    segments: list[dict],
    voice_ref_dir: Path,
    output_dir: Path,
    model: str = "voxtral-mini-tts-latest",
) -> list[dict]:
    """Synthesize all segments using Voxtral TTS with per-speaker voice cloning.

    Args:
        segments: List of segment dicts with 'text_en' and 'speaker' fields.
        voice_ref_dir: Directory with voice references ({SPEAKER_XX}_ref.wav).
        output_dir: Directory to write per-segment WAV files.
        model: Voxtral TTS model name.

    Returns:
        Updated segments list with tts_file and tts_duration fields.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map speaker labels to reference audio paths
    voice_refs = {}
    for f in sorted(voice_ref_dir.glob("*_ref.wav")):
        speaker = f.stem.replace("_ref", "")
        voice_refs[speaker] = f
    default_ref = next(iter(voice_refs.values()), None)

    if not default_ref:
        raise RuntimeError(f"No voice references found in {voice_ref_dir}")

    print(f"Voice references: {list(voice_refs.keys())}")
    print(f"Synthesizing {len(segments)} segments with Voxtral TTS...")

    failed_count = 0
    total_with_text = 0

    for seg in segments:
        text = seg.get("text_en", seg.get("text_en_deepl", ""))
        if not text.strip():
            continue

        total_with_text += 1
        seg_filename = f"segment_{seg['id']:04d}.wav"
        output_path = output_dir / seg_filename

        # Skip if already exists (resumability)
        if output_path.exists():
            data, sr = sf.read(output_path)
            seg["tts_file"] = seg_filename
            seg["tts_duration"] = round(len(data) / sr, 3)
            continue

        # Pick voice reference for this speaker
        speaker = seg.get("speaker", "")
        ref_path = voice_refs.get(speaker, default_ref)

        try:
            wav_bytes, sr = synthesize_segment(text, ref_path, model=model)
            output_path.write_bytes(wav_bytes)
            data, sr = sf.read(output_path)
            seg["tts_file"] = seg_filename
            seg["tts_duration"] = round(len(data) / sr, 3)
        except Exception as e:
            print(f"[TTS ERROR] Segment {seg.get('id')}: {e}")
            seg["tts_file"] = None
            seg["tts_duration"] = 0
            seg["tts_error"] = str(e)
            failed_count += 1

    # Fail loudly if all segments failed
    if total_with_text > 0 and failed_count == total_with_text:
        raise RuntimeError(
            f"All {total_with_text} TTS segments failed. Check errors above."
        )

    if failed_count > 0:
        print(f"WARNING: {failed_count}/{total_with_text} segments failed")

    print(f"Synthesis complete: {total_with_text - failed_count}/{total_with_text} segments OK")
    return segments
