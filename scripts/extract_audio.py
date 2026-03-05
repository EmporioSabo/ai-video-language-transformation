"""Extract audio tracks from source videos using FFmpeg."""

import subprocess
from pathlib import Path
from config import SOURCE_DIR, AUDIO_ORIGINAL_DIR, VOICE_REF_DIR, WHISPER_SAMPLE_RATE


def extract_audio(video_path: Path, output_dir: Path, sample_rate: int = WHISPER_SAMPLE_RATE):
    """Extract audio from a video file as WAV (mono, specified sample rate)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",               # Mono
        str(output_path),
    ]
    print(f"Extracting audio: {video_path.name} → {output_path.name}")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def extract_voice_reference(audio_path: Path, start_sec: float = 0, duration_sec: float = 30):
    """Extract a clean voice reference segment for TTS voice cloning.

    You should manually identify a clean segment (no background noise/music)
    and pass the start time. Default extracts first 30 seconds.
    """
    VOICE_REF_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VOICE_REF_DIR / f"{audio_path.stem}_ref.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(output_path),
    ]
    print(f"Extracting voice reference: {output_path.name} ({duration_sec}s from {start_sec}s)")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def extract_all():
    """Extract audio from all source videos."""
    video_files = list(SOURCE_DIR.glob("*.mp4"))
    if not video_files:
        print(f"No MP4 files found in {SOURCE_DIR}")
        return []

    audio_paths = []
    for video_path in sorted(video_files):
        audio_path = extract_audio(video_path, AUDIO_ORIGINAL_DIR)
        audio_paths.append(audio_path)
        # Extract voice reference from first 30s (adjust manually for cleaner segments)
        extract_voice_reference(audio_path)

    print(f"\nExtracted {len(audio_paths)} audio files to {AUDIO_ORIGINAL_DIR}")
    return audio_paths


if __name__ == "__main__":
    extract_all()
