"""Page 2: Pipeline Dashboard — run and monitor stages."""

import streamlit as st
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    SOURCE_DIR, AUDIO_ORIGINAL_DIR, VOICE_REF_DIR,
    TRANSCRIPTS_DIR, TRANSLATIONS_DIR, TTS_DIR,
    ALIGNED_DIR, OUTPUT_DIR,
)

st.header("Pipeline Dashboard")

# Define pipeline stages
STAGES = [
    {
        "name": "Extract Audio",
        "icon": "1",
        "script": "extract_audio.py",
        "local": True,
        "check": lambda: len(list(AUDIO_ORIGINAL_DIR.glob("*.wav"))) > 0,
        "description": "Extract audio tracks from source videos using FFmpeg.",
    },
    {
        "name": "Transcribe",
        "icon": "2",
        "script": None,
        "local": False,
        "check": lambda: len(list(TRANSCRIPTS_DIR.glob("*_zh.json"))) > 0,
        "description": "Run `notebooks/01_transcribe.ipynb` on Google Colab with GPU.",
        "colab_notebook": "01_transcribe.ipynb",
    },
    {
        "name": "Diarize",
        "icon": "3",
        "script": None,
        "local": False,
        "check": lambda: any(VOICE_REF_DIR.glob("SPEAKER_*_ref.wav")),
        "description": "Run `notebooks/00_diarize.ipynb` on Google Colab with GPU.",
        "colab_notebook": "00_diarize.ipynb",
    },
    {
        "name": "Translate",
        "icon": "4",
        "script": "translate.py",
        "local": True,
        "check": lambda: len(list(TRANSLATIONS_DIR.glob("*_en.json"))) > 0,
        "description": "Translate transcripts using DeepL + optional Gemini review.",
    },
    {
        "name": "Synthesize",
        "icon": "5",
        "script": None,
        "local": False,
        "check": lambda: len(list(TTS_DIR.glob("*_segments"))) > 0,
        "description": "Run `notebooks/03_synthesize.ipynb` on Google Colab with GPU.",
        "colab_notebook": "03_synthesize.ipynb",
    },
    {
        "name": "Align",
        "icon": "6",
        "script": "align_audio.py",
        "local": True,
        "check": lambda: len(list(ALIGNED_DIR.glob("*.wav"))) > 0,
        "description": "Align TTS segments to original timestamps with time-stretching and LUFS normalization.",
    },
    {
        "name": "Merge",
        "icon": "7",
        "script": "merge_video.py",
        "local": True,
        "check": lambda: len(list(OUTPUT_DIR.glob("*_EN.mp4"))) > 0,
        "description": "Merge aligned English audio into original video (video passthrough).",
    },
    {
        "name": "Subtitles",
        "icon": "8",
        "script": "generate_subtitles.py",
        "local": True,
        "check": lambda: len(list((TRANSLATIONS_DIR.parent / "subtitles").glob("*.srt"))) > 0 if (TRANSLATIONS_DIR.parent / "subtitles").exists() else False,
        "description": "Generate SRT subtitle files (bilingual or English-only).",
    },
]


def run_script(script_name: str):
    """Run a pipeline script and capture output."""
    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
    )
    return result


# Pipeline status overview
st.markdown("### Stage Status")

cols = st.columns(len(STAGES))
for i, (col, stage) in enumerate(zip(cols, STAGES)):
    done = stage["check"]()
    status = "done" if done else "pending"
    emoji = "✅" if done else "⬜"
    col.markdown(f"**{emoji} {stage['icon']}**")
    col.caption(stage["name"])

st.markdown("---")

# Detailed stage controls
for stage in STAGES:
    done = stage["check"]()
    status_icon = "✅" if done else "⏳"

    with st.expander(f"{status_icon} Stage {stage['icon']}: {stage['name']}", expanded=not done):
        st.markdown(stage["description"])

        if stage["local"]:
            # Local stage: run button
            if st.button(f"Run {stage['name']}", key=f"run_{stage['name']}"):
                with st.status(f"Running {stage['name']}...", expanded=True) as status_widget:
                    result = run_script(stage["script"])
                    if result.stdout:
                        st.code(result.stdout, language="text")
                    if result.returncode != 0:
                        st.error(f"Failed (exit code {result.returncode})")
                        if result.stderr:
                            st.code(result.stderr, language="text")
                        status_widget.update(label=f"{stage['name']} failed", state="error")
                    else:
                        status_widget.update(label=f"{stage['name']} complete", state="complete")
                        st.rerun()
        else:
            # GPU stage: Colab instructions
            st.info(f"This stage requires GPU. {stage['description']}")
            if st.button(f"Check for results", key=f"check_{stage['name']}"):
                if stage["check"]():
                    st.success("Results found! This stage is complete.")
                else:
                    st.warning("Results not found yet. Please run the notebook on Colab first.")

st.markdown("---")

# Quick actions
st.subheader("Quick Actions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Run all local stages"):
        local_stages = [s for s in STAGES if s["local"]]
        for stage in local_stages:
            if not stage["check"]():
                with st.status(f"Running {stage['name']}..."):
                    result = run_script(stage["script"])
                    if result.returncode == 0:
                        st.success(f"{stage['name']} complete")
                    else:
                        st.error(f"{stage['name']} failed")
                        break

with col2:
    if st.button("Re-run Align + Merge"):
        with st.status("Running alignment..."):
            result = run_script("align_audio.py")
            st.code(result.stdout, language="text")
        with st.status("Running merge..."):
            result = run_script("merge_video.py")
            st.code(result.stdout, language="text")
        st.success("Done! Check the output videos.")
        st.rerun()
