"""
RunPod serverless handler for GPU-intensive pipeline stages.

Stages:
  - transcribe: faster-whisper large-v3 Chinese ASR with word timestamps
  - diarize: pyannote speaker diarization + voice reference extraction
  - synthesize: F5-TTS voice cloning + per-segment synthesis

Input/output via RunPod's built-in file transfer (base64 or URLs).
"""

import base64
import io
import json
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import runpod

# ── Transcription ─────────────────────────────────────────────────────────────

def handle_transcribe(job_input):
    """Transcribe Chinese audio with faster-whisper large-v3."""
    from faster_whisper import WhisperModel

    audio_bytes = base64.b64decode(job_input["audio_b64"])
    language = job_input.get("language", "zh")
    beam_size = job_input.get("beam_size", 5)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = []
        for i, seg in enumerate(segments_iter):
            words = []
            if seg.words:
                words = [
                    {"word": w.word.strip(), "start": round(w.start, 3),
                     "end": round(w.end, 3), "probability": round(w.probability, 3)}
                    for w in seg.words
                ]
            segments.append({
                "id": i,
                "text_zh": seg.text.strip(),
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "words": words,
            })

        return {
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 1),
        }
    finally:
        os.unlink(audio_path)


# ── Diarization ───────────────────────────────────────────────────────────────

def handle_diarize(job_input):
    """Run speaker diarization and extract voice references."""
    import torch
    import numpy as np
    from pyannote.audio import Pipeline, Inference
    from pydub import AudioSegment
    from scipy.spatial.distance import cosine

    audio_bytes = base64.b64decode(job_input["audio_b64"])
    transcript_segments = job_input["segments"]
    num_speakers = job_input.get("num_speakers", 2)
    hf_token = job_input.get("hf_token", os.environ.get("HF_TOKEN", ""))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        )
        pipeline.to(device)

        # Run diarization
        output = pipeline(audio_path, num_speakers=num_speakers)
        speaker_segments = []
        for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start, "end": turn.end, "speaker": speaker
            })

        # Assign speakers to transcript segments by max overlap
        for seg in transcript_segments:
            best_speaker = "UNKNOWN"
            best_overlap = 0.0
            for sp in speaker_segments:
                overlap_start = max(seg["start"], sp["start"])
                overlap_end = min(seg["end"], sp["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = sp["speaker"]
            seg["speaker"] = best_speaker

        # Extract voice references per speaker
        audio = AudioSegment.from_wav(audio_path)
        voice_refs_b64 = {}

        speaker_segs = defaultdict(list)
        for seg in transcript_segments:
            sp = seg.get("speaker", "UNKNOWN")
            duration = seg["end"] - seg["start"]
            speaker_segs[sp].append((duration, seg))

        for speaker, segs in speaker_segs.items():
            segs.sort(key=lambda x: -x[0])
            combined = AudioSegment.empty()
            total = 0
            for dur, seg in segs:
                if total >= 30:
                    break
                start_ms = int(seg["start"] * 1000)
                end_ms = int(seg["end"] * 1000)
                combined += audio[start_ms:end_ms]
                total += dur

            buf = io.BytesIO()
            combined.export(buf, format="wav")
            voice_refs_b64[f"{speaker}_ref.wav"] = base64.b64encode(buf.getvalue()).decode()

        # Free GPU memory
        del pipeline
        torch.cuda.empty_cache()

        return {
            "segments": transcript_segments,
            "voice_refs": voice_refs_b64,
        }
    finally:
        os.unlink(audio_path)


# ── Synthesis ─────────────────────────────────────────────────────────────────

def handle_synthesize(job_input):
    """Synthesize English speech with F5-TTS voice cloning."""
    import soundfile as sf
    import numpy as np

    segments = job_input["segments"]
    voice_refs_b64 = job_input["voice_refs"]  # {filename: base64_wav}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Write voice references to disk
        ref_paths = {}
        for filename, b64_data in voice_refs_b64.items():
            ref_path = tmp / filename
            ref_path.write_bytes(base64.b64decode(b64_data))
            speaker = filename.replace("_ref.wav", "")
            ref_paths[speaker] = str(ref_path)

        # Trim voice refs to 8s max (F5-TTS overflow avoidance)
        for speaker, ref_path in ref_paths.items():
            data, sr = sf.read(ref_path)
            if len(data) / sr > 8:
                data = data[:int(8 * sr)]
                sf.write(ref_path, data, sr)

        default_ref = list(ref_paths.values())[0] if ref_paths else None

        # Load F5-TTS
        from f5_tts.api import F5TTS
        tts = F5TTS()

        # Synthesize each segment
        tts_results = {}
        for seg in segments:
            text = seg.get("text_en", seg.get("text_en_deepl", ""))
            if not text.strip():
                continue

            speaker = seg.get("speaker", "")
            ref_path = ref_paths.get(speaker, default_ref)
            if not ref_path:
                continue

            try:
                wav, sr, _ = tts.infer(
                    ref_file=ref_path,
                    ref_text="",
                    gen_text=text,
                )
                buf = io.BytesIO()
                sf.write(buf, wav, sr, format="WAV")
                seg_filename = f"segment_{seg['id']:04d}.wav"
                tts_results[seg_filename] = base64.b64encode(buf.getvalue()).decode()
                seg["tts_file"] = seg_filename
                seg["tts_duration"] = round(len(wav) / sr, 3)
            except Exception as e:
                seg["tts_file"] = None
                seg["tts_duration"] = 0

        # Package TTS segments as a ZIP (base64-encoded)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, b64_data in tts_results.items():
                zf.writestr(filename, base64.b64decode(b64_data))
        zip_b64 = base64.b64encode(zip_buf.getvalue()).decode()

        return {
            "segments": segments,
            "tts_zip_b64": zip_b64,
        }


# ── RunPod entry point ────────────────────────────────────────────────────────

def handler(job):
    """Main RunPod handler — routes to the appropriate stage."""
    job_input = job["input"]
    stage = job_input.get("stage")

    if stage == "transcribe":
        return handle_transcribe(job_input)
    elif stage == "diarize":
        return handle_diarize(job_input)
    elif stage == "synthesize":
        return handle_synthesize(job_input)
    else:
        return {"error": f"Unknown stage: {stage}"}


runpod.serverless.start({"handler": handler})
