"""Page 1: Upload and manage source videos."""

import streamlit as st
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import SOURCE_DIR, AUDIO_ORIGINAL_DIR, VOICE_REF_DIR

st.header("Upload Videos")

SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Upload new videos
uploaded_files = st.file_uploader(
    "Upload Chinese video files (MP4)",
    type=["mp4", "mkv", "avi", "mov"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded in uploaded_files:
        dest = SOURCE_DIR / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Saved: {uploaded.name}")

st.markdown("---")

# Show existing videos
st.subheader("Source Videos")

video_files = sorted(SOURCE_DIR.glob("*.mp4"))
if not video_files:
    st.info("No source videos found. Upload MP4 files above or place them in `data/source/`.")
else:
    for vf in video_files:
        col1, col2, col3 = st.columns([3, 1, 1])
        size_mb = vf.stat().st_size / (1024 * 1024)
        col1.write(f"**{vf.name}** ({size_mb:.1f} MB)")

        # Check pipeline progress for this video
        stem = vf.stem
        audio_exists = (AUDIO_ORIGINAL_DIR / f"{stem}.wav").exists()
        col2.write("Audio extracted" if audio_exists else "Not extracted")

        ref_exists = any(VOICE_REF_DIR.glob("SPEAKER_*_ref.wav"))
        col3.write("Voice refs ready" if ref_exists else "No voice refs")

    st.markdown("---")
    st.subheader("Preview")
    selected = st.selectbox("Select video to preview", [vf.name for vf in video_files])
    if selected:
        st.video(str(SOURCE_DIR / selected))
