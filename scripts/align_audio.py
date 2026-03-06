"""Align TTS audio segments to match original video timing."""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import (
    TRANSLATIONS_DIR, TTS_DIR, ALIGNED_DIR, AUDIO_ORIGINAL_DIR,
    MIN_SPEED_FACTOR, MAX_SPEED_FACTOR, CROSSFADE_MS,
    REFERENCE_SAMPLE_RATE,
)


def time_stretch(audio: AudioSegment, factor: float) -> AudioSegment:
    """Time-stretch audio by a factor using sample rate manipulation.

    factor > 1.0 = speed up (shorter duration)
    factor < 1.0 = slow down (longer duration)
    """
    new_sample_rate = int(audio.frame_rate * factor)
    stretched = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
    return stretched.set_frame_rate(audio.frame_rate)


def align_segments(translation_path: Path, tts_segments_dir: Path, output_path: Path):
    """Align TTS segments to match original timing and produce a single audio track.

    Strategy: Place each TTS segment at its original start time. Only apply
    time-stretching when the TTS segment would overflow into the next segment.
    Prefer natural speed with silence padding over aggressive stretching.
    """
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Determine total duration from last segment end time
    total_duration_ms = int(segments[-1]["end"] * 1000) + 2000  # 2s buffer

    # Create silent base track
    aligned = AudioSegment.silent(duration=total_duration_ms, frame_rate=REFERENCE_SAMPLE_RATE)

    processed = 0
    skipped = 0
    stretched_count = 0

    for i, seg in enumerate(segments):
        tts_file = seg.get("tts_file", f"segment_{seg['id']:04d}.wav")
        tts_path = tts_segments_dir / tts_file
        if not tts_path.exists():
            skipped += 1
            continue

        tts_audio = AudioSegment.from_wav(str(tts_path))

        # Trim leading/trailing silence from TTS output
        leading = detect_leading_silence(tts_audio, silence_threshold=-40)
        trailing = detect_leading_silence(tts_audio.reverse(), silence_threshold=-40)
        if leading + trailing < len(tts_audio):
            tts_audio = tts_audio[leading:len(tts_audio) - trailing]

        tts_duration_ms = len(tts_audio)
        start_ms = int(seg["start"] * 1000)

        # Calculate available window: from this segment's start to next segment's start
        # (or this segment's end if it's the last one)
        if i + 1 < len(segments):
            next_start_ms = int(segments[i + 1]["start"] * 1000)
        else:
            next_start_ms = start_ms + int((seg["end"] - seg["start"]) * 1000) + 500

        available_ms = next_start_ms - start_ms

        # Only stretch if TTS would overflow into the next segment
        if tts_duration_ms > available_ms and available_ms > 0:
            speed_factor = tts_duration_ms / available_ms
            if speed_factor <= MAX_SPEED_FACTOR:
                tts_audio = time_stretch(tts_audio, speed_factor)
                stretched_count += 1
            else:
                # Too much stretching needed — cap at max speed and truncate
                tts_audio = time_stretch(tts_audio, MAX_SPEED_FACTOR)
                if len(tts_audio) > available_ms:
                    # Fade out at the end to avoid a hard cut
                    tts_audio = tts_audio[:available_ms].fade_out(min(50, available_ms))
                stretched_count += 1

        # Place segment at its start time
        aligned = aligned.overlay(tts_audio, position=start_ms)
        processed += 1

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.export(str(output_path), format="wav")
    print(f"  Aligned {processed}/{len(segments)} segments ({skipped} skipped, {stretched_count} time-adjusted)")
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
