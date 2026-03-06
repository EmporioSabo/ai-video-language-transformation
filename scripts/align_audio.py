"""Align TTS audio segments to match original video timing.

No time-stretching — speed adjustments are handled during synthesis (F5-TTS speed param).
This script only places segments at their correct timestamps and handles overlaps.
"""

import json
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import (
    TRANSLATIONS_DIR, TTS_DIR, ALIGNED_DIR,
    REFERENCE_SAMPLE_RATE,
)


def align_segments(translation_path: Path, tts_segments_dir: Path, output_path: Path):
    """Place TTS segments at their original timestamps to produce a single audio track."""
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Total duration from last segment + 2s buffer
    total_duration_ms = int(segments[-1]["end"] * 1000) + 2000

    # Silent base track
    aligned = AudioSegment.silent(duration=total_duration_ms, frame_rate=REFERENCE_SAMPLE_RATE)

    processed = 0
    skipped = 0
    truncated = 0

    for i, seg in enumerate(segments):
        tts_file = seg.get("tts_file", f"segment_{seg['id']:04d}.wav")
        tts_path = tts_segments_dir / tts_file
        if not tts_path.exists():
            skipped += 1
            continue

        tts_audio = AudioSegment.from_wav(str(tts_path))

        # Trim leading/trailing silence
        leading = detect_leading_silence(tts_audio, silence_threshold=-40)
        trailing = detect_leading_silence(tts_audio.reverse(), silence_threshold=-40)
        if leading + trailing < len(tts_audio):
            tts_audio = tts_audio[leading:len(tts_audio) - trailing]

        start_ms = int(seg["start"] * 1000)

        # Available window until next segment
        if i + 1 < len(segments):
            next_start_ms = int(segments[i + 1]["start"] * 1000)
        else:
            next_start_ms = start_ms + int((seg["end"] - seg["start"]) * 1000) + 500

        available_ms = next_start_ms - start_ms

        # Truncate with fade-out if segment overflows into next
        if len(tts_audio) > available_ms > 0:
            fade_ms = min(50, available_ms)
            tts_audio = tts_audio[:available_ms].fade_out(fade_ms)
            truncated += 1

        aligned = aligned.overlay(tts_audio, position=start_ms)
        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.export(str(output_path), format="wav")
    print(f"  Aligned {processed}/{len(segments)} segments ({skipped} skipped, {truncated} truncated)")
    print(f"  → {output_path.name}")
    return output_path


def align_all():
    """Align audio for all translated videos."""
    translation_files = sorted(TRANSLATIONS_DIR.glob("*_en.json"))
    if not translation_files:
        print(f"No translation files found in {TRANSLATIONS_DIR}")
        return

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    for trans_path in translation_files:
        stem = trans_path.stem.replace("_en", "")
        tts_dir = TTS_DIR / f"{stem}_segments"
        output_path = ALIGNED_DIR / f"{stem}_en.wav"

        if not tts_dir.exists():
            print(f"No TTS segments found for {stem} at {tts_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Aligning: {stem}")
        align_segments(trans_path, tts_dir, output_path)

    print(f"\nAll aligned audio saved to {ALIGNED_DIR}")


if __name__ == "__main__":
    align_all()
