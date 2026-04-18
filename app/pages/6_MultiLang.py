"""Page 6: Multi-Language Pipeline — transcribe + translate any language with Voxtral.

Uses Mistral Voxtral API for transcription (no GPU needed).
The rest of the pipeline (TTS synthesis, align, merge) is identical to the Chinese pipeline.

The existing Chinese pages are untouched.
"""

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from transcribe_voxtral import (
    VOXTRAL_LANGUAGES,
    VOXTRAL_MODELS,
    transcribe_audio_voxtral,
)
from translate_multilang import translate_transcript_multilang
from extract_audio import extract_audio as _extract_audio


def extract_audio(video_path: Path, audio_path: Path):
    """Extract audio from video to a specific output file path."""
    import subprocess
    from config import WHISPER_SAMPLE_RATE
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(WHISPER_SAMPLE_RATE), "-ac", "1",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

st.header("Multi-Language Pipeline")
st.caption(
    "Transcribe a video in any language using **Voxtral** (Mistral API), "
    "then translate to English — no GPU required for these steps."
)

# ── Language selection ────────────────────────────────────────────────────────

st.subheader("1. Select source language")

lang_names = sorted(VOXTRAL_LANGUAGES.keys())
col_lang, col_model = st.columns(2)

with col_lang:
    selected_lang_name = st.selectbox(
        "Source language",
        lang_names,
        index=lang_names.index("French"),
        help="Language spoken in the video. Choose 'Auto-detect' to let Voxtral decide.",
    )
    use_autodetect = st.checkbox("Auto-detect language (let Voxtral decide)", value=False)
    source_lang_bcp47 = None if use_autodetect else VOXTRAL_LANGUAGES[selected_lang_name]
    source_lang_name = "Unknown" if use_autodetect else selected_lang_name

with col_model:
    selected_model = st.selectbox(
        "Voxtral model",
        VOXTRAL_MODELS,
        index=0,
        help="voxtral-small-2507 = best quality (24B). voxtral-mini-2507 = faster & cheaper (3B).",
    )

# ── Video upload ──────────────────────────────────────────────────────────────

st.subheader("2. Upload video")

uploaded_video = st.file_uploader(
    "Video file",
    type=["mp4", "mkv", "avi", "mov"],
    key="multilang_video",
)

if uploaded_video:
    with st.expander("Preview"):
        st.video(uploaded_video)

# ── Run pipeline ──────────────────────────────────────────────────────────────

st.subheader("3. Transcribe & Translate")

if not uploaded_video:
    st.info("Upload a video to continue.")
    st.stop()

# Session state to persist results across reruns
if "ml_transcript" not in st.session_state:
    st.session_state.ml_transcript = None
if "ml_translation" not in st.session_state:
    st.session_state.ml_translation = None
if "ml_lang_name" not in st.session_state:
    st.session_state.ml_lang_name = None

col_run1, col_run2 = st.columns(2)

with col_run1:
    run_transcription = st.button(
        "Transcribe with Voxtral",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.ml_transcript is not None,
    )

with col_run2:
    run_translation = st.button(
        "Translate to English",
        type="secondary",
        use_container_width=True,
        disabled=st.session_state.ml_transcript is None,
    )

reset = st.button("Reset", use_container_width=True)
if reset:
    st.session_state.ml_transcript = None
    st.session_state.ml_translation = None
    st.session_state.ml_lang_name = None
    st.rerun()

# ── Transcription ─────────────────────────────────────────────────────────────

if run_transcription:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        video_path = tmp / uploaded_video.name
        video_path.write_bytes(uploaded_video.getvalue())

        with st.status("Extracting audio...", expanded=True) as status:
            try:
                st.write("Extracting audio track with FFmpeg...")
                audio_path = tmp / f"{video_path.stem}.wav"
                extract_audio(video_path, audio_path)

                st.write(f"Transcribing with {selected_model}...")
                segments = transcribe_audio_voxtral(
                    audio_path,
                    language=source_lang_bcp47,
                    model=selected_model,
                )

                st.session_state.ml_transcript = segments
                st.session_state.ml_lang_name = source_lang_name
                status.update(label=f"Transcription complete — {len(segments)} segments", state="complete")

            except Exception as e:
                status.update(label=f"Error: {e}", state="error")
                st.exception(e)

# ── Show transcript ───────────────────────────────────────────────────────────

if st.session_state.ml_transcript:
    st.subheader("Transcript preview")
    segments = st.session_state.ml_transcript
    lang_label = st.session_state.ml_lang_name or "Source"

    preview_data = [
        {
            "#": seg["id"],
            "Start": f"{seg['start']:.1f}s",
            "End": f"{seg['end']:.1f}s",
            lang_label: seg.get("text_src", ""),
        }
        for seg in segments[:20]
    ]
    st.dataframe(preview_data, use_container_width=True)
    if len(segments) > 20:
        st.caption(f"Showing first 20 of {len(segments)} segments.")

    # Download transcript JSON
    st.download_button(
        label=f"Download transcript JSON ({lang_label})",
        data=json.dumps(segments, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{Path(uploaded_video.name).stem}_{source_lang_bcp47 or 'auto'}.json",
        mime="application/json",
    )

# ── Translation ───────────────────────────────────────────────────────────────

if run_translation and st.session_state.ml_transcript:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        transcript_path = tmp / "transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.ml_transcript, f, ensure_ascii=False, indent=2)

        output_path = tmp / "translation_en.json"

        with st.status("Translating...", expanded=True) as status:
            try:
                lang_name = st.session_state.ml_lang_name or selected_lang_name
                st.write(f"Translating {len(st.session_state.ml_transcript)} segments ({lang_name} → English)...")
                translate_transcript_multilang(
                    transcript_path,
                    output_path,
                    source_lang_bcp47 or "auto",
                    lang_name,
                )
                translation = json.loads(output_path.read_text(encoding="utf-8"))
                st.session_state.ml_translation = translation
                status.update(label="Translation complete", state="complete")

            except Exception as e:
                status.update(label=f"Error: {e}", state="error")
                st.exception(e)

# ── Show translation ──────────────────────────────────────────────────────────

if st.session_state.ml_translation:
    st.subheader("Translation preview")
    segments = st.session_state.ml_translation
    lang_label = st.session_state.ml_lang_name or "Source"

    preview_data = [
        {
            "#": seg["id"],
            "Start": f"{seg['start']:.1f}s",
            lang_label: seg.get("text_src", ""),
            "English": seg.get("text_en", seg.get("text_en_deepl", "")),
        }
        for seg in segments[:20]
    ]
    st.dataframe(preview_data, use_container_width=True)
    if len(segments) > 20:
        st.caption(f"Showing first 20 of {len(segments)} segments.")

    video_stem = Path(uploaded_video.name).stem
    st.download_button(
        label="Download translation JSON (use in Quick Transform / Full Pipeline)",
        data=json.dumps(segments, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{video_stem}_en.json",
        mime="application/json",
        type="primary",
        use_container_width=True,
    )

    st.info(
        "Next steps:\n"
        "1. Run **TTS synthesis** in Colab using `notebooks/03_synthesize.ipynb` "
        "(F5-TTS supports multilingual voice cloning).\n"
        "2. Use **Quick Transform** (page 5) to align + merge the final video."
    )
