"""Quality metrics computation for the pipeline dashboard."""

import json
import numpy as np
from pathlib import Path


def compute_overflow_stats(translation_path: Path) -> dict:
    """Compute segment overflow statistics from a translation JSON.

    Returns dict with:
        - total: total segment count
        - overflow_count: segments where TTS > available window
        - overflow_pct: percentage of overflows
        - ratios: list of (tts_duration / available_window) per segment
        - avg_ratio: average duration ratio
    """
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    ratios = []
    overflow_count = 0

    for i, seg in enumerate(segments):
        tts_dur = seg.get("tts_duration", 0)
        if tts_dur <= 0:
            continue

        original_dur = seg["end"] - seg["start"]
        if i + 1 < len(segments):
            available = segments[i + 1]["start"] - seg["start"]
        else:
            available = original_dur + 0.5

        if available <= 0:
            continue

        ratio = tts_dur / available
        ratios.append(ratio)
        if ratio > 1.0:
            overflow_count += 1

    total = len(ratios)
    return {
        "total": total,
        "overflow_count": overflow_count,
        "overflow_pct": round(overflow_count / total * 100, 1) if total > 0 else 0,
        "ratios": ratios,
        "avg_ratio": round(float(np.mean(ratios)), 3) if ratios else 0,
    }


def compute_speaker_stats(translation_path: Path) -> dict:
    """Compute speaker distribution from a translation JSON.

    Returns dict with:
        - speakers: {speaker_label: segment_count}
        - total: total segments with speaker labels
    """
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    speakers = {}
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        speakers[speaker] = speakers.get(speaker, 0) + 1

    return {
        "speakers": speakers,
        "total": sum(speakers.values()),
    }


def compute_lufs(audio_path: Path) -> float | None:
    """Measure integrated LUFS of an audio file."""
    try:
        import pyloudnorm as pyln
        import soundfile as sf
    except ImportError:
        return None

    data, rate = sf.read(str(audio_path))
    meter = pyln.Meter(rate)
    lufs = meter.integrated_loudness(data)
    return round(lufs, 1) if not np.isinf(lufs) else None


def compute_timing_stats(translation_path: Path) -> dict:
    """Compute timing-related stats.

    Returns dict with:
        - total_original_duration: sum of original segment durations (seconds)
        - total_tts_duration: sum of TTS durations (seconds)
        - expansion_ratio: TTS total / original total
        - short_segments: count of segments < 0.5s window
    """
    with open(translation_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    total_original = 0
    total_tts = 0
    short_segments = 0

    for i, seg in enumerate(segments):
        original_dur = seg["end"] - seg["start"]
        tts_dur = seg.get("tts_duration", 0)
        total_original += original_dur
        total_tts += tts_dur

        if original_dur < 0.5:
            short_segments += 1

    return {
        "total_original_duration": round(total_original, 1),
        "total_tts_duration": round(total_tts, 1),
        "expansion_ratio": round(total_tts / total_original, 2) if total_original > 0 else 0,
        "short_segments": short_segments,
    }


def compute_all_metrics(translations_dir: Path, aligned_dir: Path = None) -> dict:
    """Compute all metrics for all videos.

    Returns dict keyed by video stem with sub-dicts for each metric type.
    """
    results = {}
    for trans_path in sorted(translations_dir.glob("*_en.json")):
        stem = trans_path.stem.replace("_en", "")
        results[stem] = {
            "overflow": compute_overflow_stats(trans_path),
            "speakers": compute_speaker_stats(trans_path),
            "timing": compute_timing_stats(trans_path),
        }

        if aligned_dir:
            aligned_path = aligned_dir / f"{stem}_en.wav"
            if aligned_path.exists():
                results[stem]["lufs"] = compute_lufs(aligned_path)

    return results


if __name__ == "__main__":
    from config import TRANSLATIONS_DIR, ALIGNED_DIR

    metrics = compute_all_metrics(TRANSLATIONS_DIR, ALIGNED_DIR)
    for stem, m in metrics.items():
        print(f"\n{'='*40}")
        print(f"Video: {stem}")
        print(f"  Overflow: {m['overflow']['overflow_pct']}% ({m['overflow']['overflow_count']}/{m['overflow']['total']})")
        print(f"  Avg TTS/window ratio: {m['overflow']['avg_ratio']}")
        print(f"  Speakers: {m['speakers']['speakers']}")
        print(f"  Timing: {m['timing']['expansion_ratio']}x expansion, {m['timing']['short_segments']} short segs")
        if m.get("lufs") is not None:
            print(f"  LUFS: {m['lufs']}")
