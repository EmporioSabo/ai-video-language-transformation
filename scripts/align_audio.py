"""Align TTS audio segments to match original video timing."""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import (
    TRANSLATIONS_DIR, TTS_DIR, ALIGNED_DIR, AUDIO_ORIGINAL_DIR,
    MIN_SPEED_FACTOR, MAX_SPEED_FACTOR, CROSSFADE_MS, SILENCE_PADDING_MS,
    REFERENCE_SAMPLE_RATE,
)


def time_stretch(audio: AudioSegment, factor: float) -> AudioSegment:
    """Time-stretch audio by a factor using sample rate manipulation.

    factor > 1.0 = speed up (shorter duration)
    factor < 1.0 = slow down (longer duration)

    For higher quality, install rubberband-cli and use pyrubberband instead.
    """
    # Simple approach: change sample rate then convert back
    new_sample_rate = int(audio.frame_rate * factor)
    stretched = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
    return stretched.set_frame_rate(audio.frame_rate)


def try_rubberband_stretch(audio_path: Path, factor: float, output_path: Path) -> bool:
    """Try high-quality time-stretching with rubberband. Returns True if successful."""
    try:
        import pyrubberband as pyrb
        data, sr = sf.read(str(audio_path))
        stretched = pyrb.time_stretch(data, sr, factor)
        sf.write(str(output_path), stretched, sr)
        return True
    except ImportError:
        return False


def align_segments(translation_path: Path, tts_segments_dir: Path, output_path: Path):
    """Align TTS segments to match original timing and produce a single audio track."""
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Determine total duration from last segment end time
    total_duration_ms = int(segments[-1]["end"] * 1000) + 1000  # Add 1s buffer

    # Create silent base track
    aligned = AudioSegment.silent(duration=total_duration_ms, frame_rate=REFERENCE_SAMPLE_RATE)

    processed = 0
    for seg in segments:
        tts_file = seg.get("tts_file")
        if not tts_file:
            continue

        tts_path = tts_segments_dir / tts_file
        if not tts_path.exists():
            print(f"  Missing TTS file: {tts_file}")
            continue

        tts_audio = AudioSegment.from_wav(str(tts_path))

        # Trim leading/trailing silence from TTS
        leading = detect_leading_silence(tts_audio, silence_threshold=-40)
        trailing = detect_leading_silence(tts_audio.reverse(), silence_threshold=-40)
        if leading + trailing < len(tts_audio):
            tts_audio = tts_audio[leading:len(tts_audio) - trailing]

        # Calculate target duration
        original_duration_ms = int((seg["end"] - seg["start"]) * 1000)
        tts_duration_ms = len(tts_audio)

        if tts_duration_ms > 0 and original_duration_ms > 0:
            speed_factor = tts_duration_ms / original_duration_ms

            if speed_factor > MAX_SPEED_FACTOR:
                # TTS is too long — speed it up to max, accept slight overflow
                tts_audio = time_stretch(tts_audio, MAX_SPEED_FACTOR)
            elif speed_factor < MIN_SPEED_FACTOR:
                # TTS is too short — slow it down to min
                tts_audio = time_stretch(tts_audio, MIN_SPEED_FACTOR)
            elif speed_factor > 1.05:
                # TTS is slightly longer — apply gentle speedup
                tts_audio = time_stretch(tts_audio, speed_factor)
            # If TTS is shorter than original, we just place it and let silence fill the gap

        # Place segment at its start time
        start_ms = int(seg["start"] * 1000)
        aligned = aligned.overlay(tts_audio, position=start_ms)
        processed += 1

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.export(str(output_path), format="wav")
    print(f"  Aligned {processed}/{len(segments)} segments → {output_path.name}")
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
