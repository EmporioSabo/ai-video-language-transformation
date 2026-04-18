"""Transcribe audio files using Mistral Voxtral API.

Supports 30+ languages — use this instead of the Whisper notebook
for non-Chinese videos. No GPU required (API-based).

Output format is compatible with the existing pipeline (align_audio.py, merge_video.py).
The existing Chinese transcription notebooks are untouched.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# Languages supported by Voxtral: display name → BCP-47 code
VOXTRAL_LANGUAGES = {
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    "Czech": "cs",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Greek": "el",
    "Bulgarian": "bg",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Estonian": "et",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Indonesian": "id",
    "Hindi": "hi",
}

VOXTRAL_MODELS = [
    "voxtral-small-2507",  # 24B — best quality
    "voxtral-mini-2507",   # 3B  — faster, cheaper
]


def transcribe_audio_voxtral(
    audio_path: Path,
    language: str = None,
    model: str = "voxtral-small-2507",
) -> list[dict]:
    """Transcribe an audio file using Voxtral and return timestamped segments.

    Args:
        audio_path: Path to the audio file (WAV, MP3, M4A, FLAC).
        language: BCP-47 code (e.g. "fr", "es", "de"). None = auto-detect.
        model: "voxtral-small-2507" (quality) or "voxtral-mini-2507" (speed).

    Returns:
        List of segment dicts with keys: id, text_src, start, end, words.
        Compatible with translate_multilang.py and align_audio.py.
    """
    from mistralai.client import Mistral

    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY not set. Add it to your .env file:\n"
            "  MISTRAL_API_KEY=your_key_here\n"
            "Get a key at: https://console.mistral.ai/"
        )

    client = Mistral(api_key=MISTRAL_API_KEY)

    print(f"Transcribing: {audio_path.name}")
    print(f"  Model: {model}  |  Language: {language or 'auto-detect'}")
    t0 = time.time()

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=model,
            file=(audio_path.name, f),
            response_format="verbose_json",
            language=language,  # omit to let Voxtral detect automatically
            timestamp_granularities=["segment"],
        )

    elapsed = time.time() - t0
    detected = getattr(response, "language", "unknown")
    print(f"  Done in {elapsed:.1f}s — detected language: {detected}")

    # Normalise to the same segment structure used by the existing pipeline.
    # Key 'text_src' holds the source-language text (instead of 'text_zh').
    segments = []
    raw_segments = getattr(response, "segments", None) or []
    for i, seg in enumerate(raw_segments):
        segments.append({
            "id": i,
            "text_src": (getattr(seg, "text", "") or "").strip(),
            "start": round(float(getattr(seg, "start", 0.0)), 3),
            "end": round(float(getattr(seg, "end", 0.0)), 3),
            "words": [],  # word-level not always available in Voxtral verbose_json
        })

    print(f"  → {len(segments)} segments extracted")
    return segments


def transcribe_all_voxtral(
    audio_dir: Path,
    transcripts_dir: Path,
    language: str,
    model: str = "voxtral-small-2507",
) -> None:
    """Transcribe all WAV files in audio_dir and save JSON transcripts.

    Output files are named  <stem>_<lang>.json  (e.g. video-a_fr.json).
    Already-transcribed files are skipped (resumable).
    """
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        return

    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_files:
        output_path = transcripts_dir / f"{audio_path.stem}_{language}.json"
        if output_path.exists():
            print(f"Skipping {audio_path.name} — already transcribed.")
            continue

        segments = transcribe_audio_voxtral(audio_path, language=language, model=model)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path}")

    print(f"\nAll transcripts saved to {transcripts_dir}")


if __name__ == "__main__":
    import argparse
    from config import AUDIO_ORIGINAL_DIR, TRANSCRIPTS_DIR

    parser = argparse.ArgumentParser(description="Transcribe audio with Voxtral")
    parser.add_argument("--lang", required=True, help="Source language BCP-47 code (e.g. fr, es, de)")
    parser.add_argument("--model", default="voxtral-small-2507", choices=VOXTRAL_MODELS)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_ORIGINAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=TRANSCRIPTS_DIR)
    args = parser.parse_args()

    transcribe_all_voxtral(args.audio_dir, args.out_dir, args.lang, args.model)
