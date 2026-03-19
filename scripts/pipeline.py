"""End-to-end pipeline orchestration for AI Video Language Transformation.

Usage:
    python pipeline.py                    # Run all stages
    python pipeline.py --stage extract    # Run a specific stage
    python pipeline.py --stage translate  # Run from translate onward
"""

import argparse
import sys
import time
from pathlib import Path

from config import SOURCE_DIR, AUDIO_ORIGINAL_DIR, VOICE_REF_DIR, TRANSCRIPTS_DIR, TRANSLATIONS_DIR, TTS_DIR, ALIGNED_DIR, OUTPUT_DIR


STAGES = ["download", "extract", "transcribe", "diarize", "translate", "synthesize", "align", "merge", "subtitles"]


def check_prerequisites(stage: str) -> bool:
    """Check if prerequisites for a stage are met."""
    checks = {
        "extract": lambda: any(SOURCE_DIR.glob("*.mp4")),
        "transcribe": lambda: any(AUDIO_ORIGINAL_DIR.glob("*.wav")),
        "diarize": lambda: any(TRANSCRIPTS_DIR.glob("*_zh.json")),
        "translate": lambda: any(TRANSCRIPTS_DIR.glob("*_zh.json")),
        "synthesize": lambda: any(TRANSLATIONS_DIR.glob("*_en.json")),
        "align": lambda: any(TTS_DIR.iterdir()) if TTS_DIR.exists() else False,
        "merge": lambda: any(ALIGNED_DIR.glob("*_en.wav")),
        "subtitles": lambda: any(TRANSLATIONS_DIR.glob("*_en.json")),
    }
    check = checks.get(stage)
    return check is None or check()


def run_stage(stage: str):
    """Run a single pipeline stage."""
    print(f"\n{'='*60}")
    print(f"STAGE: {stage.upper()}")
    print(f"{'='*60}\n")

    if not check_prerequisites(stage):
        print(f"Prerequisites not met for stage '{stage}'. Run previous stages first.")
        return False

    start = time.time()

    if stage == "download":
        from download_videos import download_all
        download_all()

    elif stage == "extract":
        from extract_audio import extract_all
        extract_all()

    elif stage == "transcribe":
        print("NOTE: Transcription requires GPU. Run notebooks/01_transcribe.ipynb in Google Colab.")
        print(f"Expected input:  {AUDIO_ORIGINAL_DIR}/*.wav")
        print(f"Expected output: {TRANSCRIPTS_DIR}/*_zh.json")
        return True

    elif stage == "diarize":
        print("NOTE: Diarization requires GPU. Run notebooks/00_diarize.ipynb in Google Colab.")
        print(f"Expected input:  {AUDIO_ORIGINAL_DIR}/*.wav + {TRANSCRIPTS_DIR}/*_zh.json")
        print(f"Expected output: Updated transcripts with speaker labels + voice references")
        return True

    elif stage == "translate":
        from translate import translate_all
        translate_all()

    elif stage == "synthesize":
        print("NOTE: TTS synthesis requires GPU. Run notebooks/03_synthesize.ipynb in Google Colab.")
        print(f"Expected input:  {TRANSLATIONS_DIR}/*_en.json")
        print(f"Expected output: {TTS_DIR}/*/segment_*.wav")
        return True

    elif stage == "align":
        from align_audio import align_all
        align_all()

    elif stage == "merge":
        from merge_video import merge_all
        merge_all()

    elif stage == "subtitles":
        from generate_subtitles import generate_all
        generate_all(bilingual=True)

    elapsed = time.time() - start
    print(f"\nStage '{stage}' completed in {elapsed:.1f}s")
    return True


def run_pipeline(start_stage: str = None):
    """Run the full pipeline or from a specific stage."""
    stages = STAGES
    if start_stage:
        if start_stage not in STAGES:
            print(f"Unknown stage: {start_stage}. Available: {STAGES}")
            sys.exit(1)
        idx = STAGES.index(start_stage)
        stages = STAGES[idx:]

    print(f"Running pipeline stages: {' → '.join(stages)}")
    total_start = time.time()

    for stage in stages:
        success = run_stage(stage)
        if not success:
            print(f"\nPipeline stopped at stage '{stage}'.")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE ({total_elapsed:.1f}s)")
    print(f"Output videos: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Video Language Transformation Pipeline")
    parser.add_argument("--stage", type=str, help=f"Start from a specific stage: {STAGES}")
    args = parser.parse_args()
    run_pipeline(args.stage)
