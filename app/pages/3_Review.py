"""Page 3: Review — inspect segments, listen to audio, edit translations."""

import json
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import TRANSLATIONS_DIR, TTS_DIR, OUTPUT_DIR, AUDIO_ORIGINAL_DIR

st.header("Review & Edit")

# Select video
translation_files = sorted(TRANSLATIONS_DIR.glob("*_en.json")) if TRANSLATIONS_DIR.exists() else []
if not translation_files:
    st.info("No translation files found. Run the pipeline first.")
    st.stop()

video_names = [f.stem.replace("_en", "") for f in translation_files]
selected = st.selectbox("Select video", video_names)
trans_path = TRANSLATIONS_DIR / f"{selected}_en.json"

with open(trans_path, "r", encoding="utf-8") as f:
    segments = json.load(f)

# Output video preview
output_video = OUTPUT_DIR / f"{selected}_EN.mp4"
if output_video.exists():
    st.subheader("Output Video")
    st.video(str(output_video))

st.markdown("---")

# Segment table
st.subheader(f"Segments ({len(segments)} total)")

# Build dataframe
df_data = []
for seg in segments:
    tts_dur = seg.get("tts_duration", 0)
    original_dur = seg["end"] - seg["start"]
    ratio = tts_dur / original_dur if original_dur > 0 else 0

    df_data.append({
        "ID": seg["id"],
        "Start": f"{seg['start']:.1f}s",
        "End": f"{seg['end']:.1f}s",
        "Speaker": seg.get("speaker", "?"),
        "Chinese": seg.get("text_zh", ""),
        "English": seg.get("text_en", seg.get("text_en_deepl", "")),
        "TTS Duration": f"{tts_dur:.1f}s",
        "Window": f"{original_dur:.1f}s",
        "Ratio": f"{ratio:.1f}x",
        "Overflow": "⚠️" if ratio > 1.0 else "✅",
    })

df = pd.DataFrame(df_data)

# Filters
col1, col2 = st.columns(2)
with col1:
    show_overflow = st.checkbox("Show only overflow segments", value=False)
with col2:
    speaker_filter = st.selectbox(
        "Filter by speaker",
        ["All"] + list(set(seg.get("speaker", "?") for seg in segments)),
    )

if show_overflow:
    df = df[df["Overflow"] == "⚠️"]
if speaker_filter != "All":
    df = df[df["Speaker"] == speaker_filter]

st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# Individual segment inspector
st.subheader("Segment Inspector")

seg_id = st.number_input("Segment ID", min_value=0, max_value=len(segments) - 1, value=0)
seg = segments[seg_id]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Chinese:** {seg.get('text_zh', '')}")
    st.markdown(f"**English:** {seg.get('text_en', seg.get('text_en_deepl', ''))}")
    st.markdown(f"**Speaker:** {seg.get('speaker', '?')}")
    st.markdown(f"**Time:** {seg['start']:.2f}s → {seg['end']:.2f}s")

with col2:
    tts_dir = TTS_DIR / f"{selected}_segments"
    tts_file = seg.get("tts_file", f"segment_{seg['id']:04d}.wav")
    tts_path = tts_dir / tts_file

    if tts_path.exists():
        st.markdown("**TTS Audio:**")
        st.audio(str(tts_path))
        st.caption(f"Duration: {seg.get('tts_duration', 0):.2f}s")
    else:
        st.warning("TTS segment not found")

# Translation editor
st.markdown("---")
st.subheader("Edit Translation")

new_text = st.text_area(
    "Edit English text for this segment",
    value=seg.get("text_en", seg.get("text_en_deepl", "")),
    key=f"edit_{seg_id}",
)

if st.button("Save Edit"):
    segments[seg_id]["text_en"] = new_text
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    st.success(f"Saved! Re-run synthesis for segment {seg_id} to hear the updated audio.")
