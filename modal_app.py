"""
Modal serverless GPU app — web endpoints for transcribe + synthesize.

Deploy:
  modal deploy modal_app.py

After deploy, copy the printed URLs to HF Spaces secrets:
  MODAL_TRANSCRIBE_URL
  MODAL_SYNTHESIZE_URL
"""

import base64
import io
import os
import tempfile
import zipfile
from pathlib import Path

import modal

# ── Images ────────────────────────────────────────────────────────────────────

transcribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "requests")
    .pip_install(
        "faster-whisper==1.1.1",
        "torch==2.3.1",
        "torchaudio==2.3.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

synthesize_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("fastapi[standard]", "requests")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "f5-tts==1.1.4",
        "soundfile",
    )
)

app = modal.App("ai-video-language-transformation")

whisper_cache = modal.Volume.from_name("whisper-cache", create_if_missing=True)
f5tts_cache = modal.Volume.from_name("f5tts-cache", create_if_missing=True)


# ── Transcribe endpoint ───────────────────────────────────────────────────────

@app.function(
    image=transcribe_image,
    gpu="T4",
    timeout=600,
    volumes={"/root/.cache/huggingface": whisper_cache},
    secrets=[modal.Secret.from_name("ai-video")],
)
@modal.fastapi_endpoint(method="POST")
def transcribe_http(item: dict) -> dict:
    """Transcribe audio with faster-whisper large-v3."""
    from faster_whisper import WhisperModel

    audio_b64 = item["audio_b64"]
    language = item.get("language", "zh")
    beam_size = item.get("beam_size", 5)

    audio_bytes = base64.b64decode(audio_b64)
    suffix = ".mp3" if (
        audio_bytes[:3] == b"ID3" or
        audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa")
    ) else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
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


# ── Synthesize endpoint ───────────────────────────────────────────────────────

@app.function(
    image=synthesize_image,
    gpu="T4",
    timeout=1200,
    volumes={"/root/.cache/huggingface": f5tts_cache},
    secrets=[modal.Secret.from_name("ai-video")],
)
@modal.fastapi_endpoint(method="POST")
def synthesize_http(item: dict) -> dict:
    """Synthesize English speech with F5-TTS voice cloning."""
    import soundfile as sf
    import torch

    torch.backends.cudnn.enabled = False

    segments = item["segments"]
    voice_refs_b64 = item["voice_refs"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        ref_paths = {}
        for filename, b64_data in voice_refs_b64.items():
            ref_path = tmp / filename
            ref_path.write_bytes(base64.b64decode(b64_data))
            speaker = filename.replace("_ref.wav", "")
            ref_paths[speaker] = str(ref_path)

        for speaker, ref_path in ref_paths.items():
            data, sr = sf.read(ref_path)
            if len(data) / sr > 6:
                data = data[:int(6 * sr)]
                sf.write(ref_path, data, sr)

        default_ref = list(ref_paths.values())[0] if ref_paths else None

        from f5_tts.api import F5TTS
        tts = F5TTS()

        def _infer_with_retry(ref_path: str, text: str) -> tuple:
            ref_data, ref_sr = sf.read(ref_path)
            for clip_secs in (6, 4, 3, 2):
                clip_samples = int(clip_secs * ref_sr)
                if len(ref_data) > clip_samples:
                    clipped = ref_data[:clip_samples]
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        sf.write(tf.name, clipped, ref_sr)
                        clip_path = tf.name
                else:
                    clip_path = ref_path
                try:
                    return tts.infer(ref_file=clip_path, ref_text="", gen_text=text)
                except RuntimeError as e:
                    if "Sizes of tensors must match" in str(e) and clip_secs > 2:
                        continue
                    raise
            raise RuntimeError("All clip lengths failed with tensor size mismatch")

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
                wav, sr, _ = _infer_with_retry(ref_path, text)
                buf = io.BytesIO()
                sf.write(buf, wav, sr, format="WAV")
                seg_filename = f"segment_{seg['id']:04d}.wav"
                tts_results[seg_filename] = base64.b64encode(buf.getvalue()).decode()
                seg["tts_file"] = seg_filename
                seg["tts_duration"] = round(len(wav) / sr, 3)
            except Exception as e:
                print(f"[TTS ERROR] Segment {seg.get('id')}: {e}", flush=True)
                seg["tts_file"] = None
                seg["tts_duration"] = 0
                seg["tts_error"] = str(e)

        if not tts_results:
            errors = [s.get("tts_error", "unknown") for s in segments if s.get("tts_file") is None]
            raise RuntimeError(
                f"All {len(segments)} TTS segments failed. "
                f"First error: {errors[0] if errors else 'unknown'}"
            )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, b64_data in tts_results.items():
                zf.writestr(filename, base64.b64decode(b64_data))

        return {
            "segments": segments,
            "tts_zip_b64": base64.b64encode(zip_buf.getvalue()).decode(),
        }
