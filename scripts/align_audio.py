"""Align TTS audio segments to match original video timing.

Uses pyrubberband for high-quality time-stretching (up to MAX_SPEED_FACTOR)
when TTS segments overflow their time windows. Only truncates as a last resort.
Applies LUFS loudness normalization to the final aligned track.
"""

import json
import numpy as np
import soundfile as sf_lib
import pyrubberband as pyrb
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import (
    TRANSLATIONS_DIR, TTS_DIR, ALIGNED_DIR,
    REFERENCE_SAMPLE_RATE, MAX_SPEED_FACTOR,
    CROSSFADE_MS, TARGET_LUFS,
)


def time_stretch_audio(audio: AudioSegment, speed_factor: float) -> AudioSegment:
    """Speed up audio using pyrubberband (preserves pitch)."""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate
    channels = audio.channels

    if channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    # Normalize to float range for pyrubberband
    max_val = float(2 ** (audio.sample_width * 8 - 1))
    samples = samples / max_val

    stretched = pyrb.time_stretch(samples, sample_rate, speed_factor)

    # Convert back to int16
    stretched = np.clip(stretched * max_val, -max_val, max_val - 1).astype(np.int16)

    return AudioSegment(
        data=stretched.tobytes(),
        sample_width=audio.sample_width,
        frame_rate=sample_rate,
        channels=channels,
    )


def normalize_lufs(audio_path: Path, target_lufs: float = TARGET_LUFS):
    """Normalize audio file to target LUFS loudness."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        print("  Warning: pyloudnorm not installed, skipping LUFS normalization")
        return

    data, rate = sf_lib.read(str(audio_path))

    meter = pyln.Meter(rate)
    current_lufs = meter.integrated_loudness(data)

    if np.isinf(current_lufs):
        print("  Warning: Could not measure LUFS (silent audio?), skipping normalization")
        return

    normalized = pyln.normalize.loudness(data, current_lufs, target_lufs)
    sf_lib.write(str(audio_path), normalized, rate)
    print(f"  LUFS: {current_lufs:.1f} → {target_lufs:.1f}")


def align_segments(translation_path: Path, tts_segments_dir: Path, output_path: Path):
    """Place TTS segments at their original timestamps to produce a single audio track.

    Applies rubberband time-stretching for segments that overflow, and only
    truncates as a last resort when speedup exceeds MAX_SPEED_FACTOR.
    """
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Total duration from last segment + 2s buffer
    total_duration_ms = int(segments[-1]["end"] * 1000) + 2000

    # Silent base track
    aligned = AudioSegment.silent(duration=total_duration_ms, frame_rate=REFERENCE_SAMPLE_RATE)

    processed = 0
    skipped = 0
    stretched = 0
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

        # Handle overflow: try time-stretching first, then truncate
        if len(tts_audio) > available_ms > 0:
            speed_factor = len(tts_audio) / available_ms

            if speed_factor <= MAX_SPEED_FACTOR:
                # Speed up with rubberband (pitch-preserving)
                tts_audio = time_stretch_audio(tts_audio, speed_factor)
                stretched += 1
            else:
                # Speed up to max, then truncate the remainder
                tts_audio = time_stretch_audio(tts_audio, MAX_SPEED_FACTOR)
                if len(tts_audio) > available_ms:
                    fade_ms = min(CROSSFADE_MS, available_ms)
                    tts_audio = tts_audio[:available_ms].fade_out(fade_ms)
                truncated += 1

        aligned = aligned.overlay(tts_audio, position=start_ms)
        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.export(str(output_path), format="wav")

    # Apply LUFS normalization
    normalize_lufs(output_path)

    print(f"  Aligned {processed}/{len(segments)} segments "
          f"({skipped} skipped, {stretched} time-stretched, {truncated} truncated)")
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
