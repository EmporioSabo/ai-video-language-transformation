"""Translate transcripts from any source language to English.

Mirrors translate.py but works with any language.
Reads segments produced by transcribe_voxtral.py (key: 'text_src').
Writes 'text_en_deepl' and 'text_en' — fully compatible with align_audio.py.

The original translate.py (Chinese → English) is completely untouched.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TARGET_LANGUAGE = "EN-US"

# BCP-47 code → DeepL source language code
# Languages absent from this map are not supported by DeepL;
# they will fall back to Gemini-only translation.
DEEPL_SOURCE_CODES = {
    "fr": "FR",
    "es": "ES",
    "de": "DE",
    "it": "IT",
    "pt": "PT",
    "ru": "RU",
    "ja": "JA",
    "ko": "KO",
    "nl": "NL",
    "pl": "PL",
    "tr": "TR",
    "uk": "UK",
    "sv": "SV",
    "no": "NB",
    "da": "DA",
    "fi": "FI",
    "cs": "CS",
    "ro": "RO",
    "hu": "HU",
    "el": "EL",
    "bg": "BG",
    "sk": "SK",
    "sl": "SL",
    "et": "ET",
    "lv": "LV",
    "lt": "LT",
    "id": "ID",
    "ar": "AR",
}


def translate_with_deepl(segments: list[dict], source_lang_bcp47: str) -> list[dict]:
    """Translate all segments using DeepL."""
    import deepl

    deepl_code = DEEPL_SOURCE_CODES.get(source_lang_bcp47)
    if not deepl_code:
        print(f"  DeepL does not support '{source_lang_bcp47}' — skipping DeepL step.")
        for seg in segments:
            seg["text_en_deepl"] = seg["text_src"]
        return segments

    translator = deepl.Translator(DEEPL_API_KEY)
    texts = [seg["text_src"] for seg in segments]

    print(f"Translating {len(texts)} segments with DeepL ({deepl_code} → {TARGET_LANGUAGE})...")
    for seg, text in zip(segments, tqdm(texts)):
        result = translator.translate_text(text, source_lang=deepl_code, target_lang=TARGET_LANGUAGE)
        seg["text_en_deepl"] = result.text

    return segments


def review_with_gemini(segments: list[dict], source_lang_name: str) -> list[dict]:
    """Refine translations with Gemini for spoken-English naturalness."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    system_prompt = f"""You are a translation reviewer for a video being dubbed from {source_lang_name} to English.

Your task: review and refine the English translations so they:
1. Sound natural as SPOKEN English (read aloud by TTS — not screen text)
2. Are concise — avoid overly formal or wordy phrasing
3. Preserve the speaker's tone and intent

For each segment output ONLY the refined English translation. No explanations."""

    print(f"Reviewing {len(segments)} translations with Gemini...")
    failed = 0
    for seg in tqdm(segments):
        prompt = (
            f"Original {source_lang_name}: {seg['text_src']}\n"
            f"DeepL translation: {seg.get('text_en_deepl', '')}\n\n"
            "Refined English (spoken, natural):"
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=system_prompt + "\n\n" + prompt,
                config={"temperature": 0.3, "max_output_tokens": 500},
            )
            seg["text_en"] = response.text.strip().strip('"')
        except Exception as e:
            failed += 1
            seg["text_en"] = seg.get("text_en_deepl", seg["text_src"])
            if "429" in str(e) and failed == 1:
                print("\n  Gemini quota exceeded — falling back to DeepL for remaining segments.")
            elif failed == 1:
                print(f"\n  Gemini error: {e} — falling back for this segment.")

    if failed:
        print(f"  {failed} segment(s) used DeepL fallback.")
    return segments


def translate_transcript_multilang(
    transcript_path: Path,
    output_path: Path,
    source_lang_bcp47: str,
    source_lang_name: str,
) -> None:
    """Full translation pipeline for one transcript file."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Step 1: DeepL (if API key present and language supported)
    if DEEPL_API_KEY:
        segments = translate_with_deepl(segments, source_lang_bcp47)
    else:
        print("No DEEPL_API_KEY — skipping DeepL pass.")
        for seg in segments:
            seg["text_en_deepl"] = seg["text_src"]

    # Step 2: Gemini review for spoken-English naturalness
    if GEMINI_API_KEY:
        segments = review_with_gemini(segments, source_lang_name)
    else:
        print("No GEMINI_API_KEY — using DeepL translations directly.")
        for seg in segments:
            seg["text_en"] = seg.get("text_en_deepl", seg["text_src"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def translate_all_multilang(
    transcripts_dir: Path,
    translations_dir: Path,
    source_lang_bcp47: str,
    source_lang_name: str,
) -> None:
    """Translate all *_{lang}.json transcripts in transcripts_dir."""
    transcript_files = sorted(transcripts_dir.glob(f"*_{source_lang_bcp47}.json"))
    if not transcript_files:
        print(f"No transcript files found in {transcripts_dir} for language '{source_lang_bcp47}'")
        return

    translations_dir.mkdir(parents=True, exist_ok=True)

    for transcript_path in transcript_files:
        stem = transcript_path.stem.replace(f"_{source_lang_bcp47}", "")
        output_path = translations_dir / f"{stem}_en.json"
        print(f"\n{'='*60}")
        print(f"Processing: {transcript_path.name}")
        translate_transcript_multilang(transcript_path, output_path, source_lang_bcp47, source_lang_name)

    print(f"\nAll translations saved to {translations_dir}")


if __name__ == "__main__":
    import argparse
    from config import TRANSCRIPTS_DIR, TRANSLATIONS_DIR
    from transcribe_voxtral import VOXTRAL_LANGUAGES

    parser = argparse.ArgumentParser(description="Translate multilingual transcripts to English")
    parser.add_argument("--lang", required=True, help="Source language BCP-47 code (e.g. fr, es, de)")
    parser.add_argument("--transcripts-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument("--translations-dir", type=Path, default=TRANSLATIONS_DIR)
    args = parser.parse_args()

    # Resolve display name from BCP-47 code
    lang_name = next((k for k, v in VOXTRAL_LANGUAGES.items() if v == args.lang), args.lang)

    translate_all_multilang(args.transcripts_dir, args.translations_dir, args.lang, lang_name)
