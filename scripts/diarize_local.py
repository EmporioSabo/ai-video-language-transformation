"""
Local CPU-based speaker diarization using pyannote.audio.

Runs on the local machine (no GPU needed) to avoid Docker/CUDA dependency hell
on RunPod. Returns the same format as runpod_client.diarize() for compatibility.
"""

import base64
import io
import os
from pathlib import Path

import torch
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
from scipy.spatial.distance import cosine


def diarize(audio_path: str, segments: list, num_speakers: int = None,
            hf_token: str = "") -> dict:
    """Run speaker diarization and extract voice references.

    Returns dict with 'segments' (updated with speaker labels)
    and 'voice_refs' (base64-encoded WAV clips per speaker).
    """
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    device = torch.device("cpu")

    # Load diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )
    pipeline.to(device)

    # Run diarization
    # If num_speakers is None, pyannote auto-detects the number of speakers
    diarize_kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipeline(audio_path, **diarize_kwargs)
    speaker_segments = []
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start, "end": turn.end, "speaker": speaker
        })

    # Assign speakers to transcript segments by overlap
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        speaker_overlap = {}
        for ds in speaker_segments:
            overlap = max(0, min(seg_end, ds["end"]) - max(seg_start, ds["start"]))
            if overlap > 0:
                speaker_overlap[ds["speaker"]] = speaker_overlap.get(ds["speaker"], 0) + overlap
        if speaker_overlap:
            seg["speaker"] = max(speaker_overlap, key=speaker_overlap.get)
        else:
            seg["speaker"] = "SPEAKER_00"

    # Extract voice references (longest contiguous segments per speaker)
    audio = AudioSegment.from_file(audio_path)
    speakers = set(seg["speaker"] for seg in segments)
    voice_refs_b64 = {}

    for speaker in speakers:
        sp_segs = sorted(
            [ds for ds in speaker_segments if ds["speaker"] == speaker],
            key=lambda x: x["end"] - x["start"],
            reverse=True,
        )
        combined = AudioSegment.empty()
        total = 0
        for ds in sp_segs:
            if total >= 8:
                break
            start_ms = int(ds["start"] * 1000)
            end_ms = int(ds["end"] * 1000)
            dur = (end_ms - start_ms) / 1000
            combined += audio[start_ms:end_ms]
            total += dur

        buf = io.BytesIO()
        combined.export(buf, format="wav")
        voice_refs_b64[f"{speaker}_ref.wav"] = base64.b64encode(buf.getvalue()).decode()

    return {
        "segments": segments,
        "voice_refs": voice_refs_b64,
    }
