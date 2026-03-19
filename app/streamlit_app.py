"""AI Video Language Transformation — Streamlit Interface."""

import streamlit as st
import sys
from pathlib import Path

# Add scripts/ to path so we can import config, metrics, etc.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

st.set_page_config(
    page_title="AI Video Language Transformation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Video Language Transformation")
st.markdown("Transform Chinese-language videos into English with AI-powered dubbing.")

st.markdown("---")

st.markdown("""
### How it works

1. **Upload** — Add Chinese video files to the source directory
2. **Pipeline** — Run each stage (extract, transcribe, translate, synthesize, align, merge)
3. **Review** — Listen to segments, edit translations, check quality
4. **Metrics** — View overflow stats, LUFS levels, speaker distribution

Use the **sidebar** to navigate between pages.
""")

# Show quick status
from config import SOURCE_DIR, TRANSLATIONS_DIR, TTS_DIR, ALIGNED_DIR, OUTPUT_DIR

col1, col2, col3, col4 = st.columns(4)

source_count = len(list(SOURCE_DIR.glob("*.mp4"))) if SOURCE_DIR.exists() else 0
trans_count = len(list(TRANSLATIONS_DIR.glob("*_en.json"))) if TRANSLATIONS_DIR.exists() else 0
aligned_count = len(list(ALIGNED_DIR.glob("*.wav"))) if ALIGNED_DIR.exists() else 0
output_count = len(list(OUTPUT_DIR.glob("*_EN.mp4"))) if OUTPUT_DIR.exists() else 0

col1.metric("Source Videos", source_count)
col2.metric("Translations", trans_count)
col3.metric("Aligned Audio", aligned_count)
col4.metric("Output Videos", output_count)
