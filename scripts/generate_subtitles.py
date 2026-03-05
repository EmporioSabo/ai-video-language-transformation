"""Generate SRT subtitle files from translation data (optional enhancement)."""

import json
from pathlib import Path

from config import TRANSLATIONS_DIR


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(translation_path: Path, output_path: Path, bilingual: bool = False):
    """Generate an SRT subtitle file from a translation JSON."""
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    lines = []
    for i, seg in enumerate(segments, 1):
        start = seconds_to_srt_time(seg["start"])
        end = seconds_to_srt_time(seg["end"])
        text_en = seg.get("text_en", seg.get("text_en_deepl", ""))

        if bilingual:
            text = f"{seg['text_zh']}\n{text_en}"
        else:
            text = text_en

        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated: {output_path} ({len(segments)} subtitles)")


def generate_all(bilingual: bool = False):
    """Generate subtitles for all translated videos."""
    subtitle_dir = TRANSLATIONS_DIR.parent / "subtitles"

    for trans_path in sorted(TRANSLATIONS_DIR.glob("*_en.json")):
        stem = trans_path.stem.replace("_en", "")
        suffix = "_bilingual" if bilingual else "_en"
        output_path = subtitle_dir / f"{stem}{suffix}.srt"
        generate_srt(trans_path, output_path, bilingual=bilingual)


if __name__ == "__main__":
    generate_all(bilingual=True)
