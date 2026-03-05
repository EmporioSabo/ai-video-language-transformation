"""Merge aligned English audio into original video files using FFmpeg."""

import subprocess
from pathlib import Path

from config import SOURCE_DIR, ALIGNED_DIR, OUTPUT_DIR, OUTPUT_VIDEO_SUFFIX


def merge_audio_video(video_path: Path, audio_path: Path, output_path: Path):
    """Replace original audio in video with aligned English audio.

    Uses video stream passthrough (no re-encoding) for speed and quality.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),       # Original video
        "-i", str(audio_path),       # New English audio
        "-c:v", "copy",              # Copy video stream (no re-encoding)
        "-c:a", "aac",               # Encode audio as AAC
        "-b:a", "192k",              # Audio bitrate
        "-map", "0:v:0",             # Take video from first input
        "-map", "1:a:0",             # Take audio from second input
        "-shortest",                 # Cut to shortest stream
        str(output_path),
    ]

    print(f"Merging: {video_path.name} + {audio_path.name} → {output_path.name}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Done: {output_path}")
    return output_path


def merge_all():
    """Merge all aligned audio files with their corresponding videos."""
    aligned_files = sorted(ALIGNED_DIR.glob("*_en.wav"))
    if not aligned_files:
        print(f"No aligned audio files found in {ALIGNED_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for audio_path in aligned_files:
        stem = audio_path.stem.replace("_en", "")
        # Find matching source video
        video_candidates = list(SOURCE_DIR.glob(f"{stem}*.*"))
        video_candidates = [v for v in video_candidates if v.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")]

        if not video_candidates:
            # Try broader match
            video_candidates = list(SOURCE_DIR.glob("*.mp4"))

        if not video_candidates:
            print(f"No source video found for {stem}")
            continue

        video_path = video_candidates[0]
        output_path = OUTPUT_DIR / f"{stem}{OUTPUT_VIDEO_SUFFIX}.mp4"
        merge_audio_video(video_path, audio_path, output_path)

    print(f"\nAll output videos saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    merge_all()
