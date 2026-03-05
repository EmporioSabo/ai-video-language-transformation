"""Translate Chinese transcripts to English using DeepL + Gemini review."""

import json
import time
import deepl
from pathlib import Path
from tqdm import tqdm

from config import (
    DEEPL_API_KEY, GEMINI_API_KEY, TARGET_LANGUAGE,
    TRANSCRIPTS_DIR, TRANSLATIONS_DIR, GITHUB_GLOSSARY,
)


def build_glossary_prompt():
    """Build a glossary string for the Gemini review prompt."""
    lines = [f'  "{zh}" → "{en}"' for zh, en in GITHUB_GLOSSARY.items()]
    return "\n".join(lines)


def translate_with_deepl(segments: list[dict]) -> list[dict]:
    """Translate all segments using DeepL free API."""
    translator = deepl.Translator(DEEPL_API_KEY)

    # Build DeepL glossary if supported
    texts = [seg["text_zh"] for seg in segments]

    print(f"Translating {len(texts)} segments with DeepL...")
    results = []
    for text in tqdm(texts):
        result = translator.translate_text(text, source_lang="ZH", target_lang=TARGET_LANGUAGE)
        results.append(result.text)

    for seg, translation in zip(segments, results):
        seg["text_en_deepl"] = translation

    return segments


def review_with_gemini(segments: list[dict]) -> list[dict]:
    """Review and refine translations using Gemini for naturalness and glossary consistency."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    glossary = build_glossary_prompt()

    system_prompt = f"""You are a translation reviewer for a GitHub tutorial video being dubbed from Chinese to English.

Your task: review and refine the English translations to ensure they:
1. Use correct GitHub terminology consistently (see glossary below)
2. Sound natural as SPOKEN English (this will be read aloud by TTS, not read on screen)
3. Are concise — avoid overly formal or wordy phrasing
4. Preserve the tutorial's instructional tone

GitHub Terminology Glossary:
{glossary}

For each segment, output ONLY the refined English translation. If the translation is already good, return it unchanged.
Do not add explanations or notes."""

    print(f"Reviewing {len(segments)} translations with Gemini...")
    failed = 0
    for seg in tqdm(segments):
        prompt = f"""Original Chinese: {seg['text_zh']}
DeepL translation: {seg['text_en_deepl']}

Refined English (spoken, natural):"""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=system_prompt + "\n\n" + prompt,
                config={"temperature": 0.3, "max_output_tokens": 500},
            )
            seg["text_en"] = response.text.strip().strip('"')
        except Exception as e:
            failed += 1
            seg["text_en"] = seg["text_en_deepl"]
            if "429" in str(e) and failed == 1:
                print(f"\n  Gemini quota exceeded — falling back to DeepL for remaining segments.")
            elif failed == 1:
                print(f"\n  Gemini error: {e} — falling back to DeepL for this segment.")

    if failed:
        print(f"  {failed} segment(s) used DeepL fallback due to Gemini errors.")
    return segments


def translate_transcript(transcript_path: Path, output_path: Path):
    """Full translation pipeline for one transcript file."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Step 1: DeepL translation
    segments = translate_with_deepl(segments)

    # Step 2: Gemini review for naturalness and glossary
    if GEMINI_API_KEY:
        segments = review_with_gemini(segments)
    else:
        print("No Gemini API key — skipping review pass, using DeepL translations directly.")
        for seg in segments:
            seg["text_en"] = seg["text_en_deepl"]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def translate_all():
    """Translate all transcript files."""
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*_zh.json"))
    if not transcript_files:
        print(f"No transcript files found in {TRANSCRIPTS_DIR}")
        return

    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for transcript_path in transcript_files:
        stem = transcript_path.stem.replace("_zh", "")
        output_path = TRANSLATIONS_DIR / f"{stem}_en.json"
        print(f"\n{'='*60}")
        print(f"Processing: {transcript_path.name}")
        translate_transcript(transcript_path, output_path)

    print(f"\nAll translations saved to {TRANSLATIONS_DIR}")


if __name__ == "__main__":
    translate_all()
