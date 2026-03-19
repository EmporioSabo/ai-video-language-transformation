"""Page 4: Quality Metrics Dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import TRANSLATIONS_DIR, ALIGNED_DIR
from metrics import compute_all_metrics

st.header("Quality Metrics")

if not TRANSLATIONS_DIR.exists() or not list(TRANSLATIONS_DIR.glob("*_en.json")):
    st.info("No translation data found. Run the pipeline first.")
    st.stop()

# Compute all metrics
aligned_dir = ALIGNED_DIR if ALIGNED_DIR.exists() else None
all_metrics = compute_all_metrics(TRANSLATIONS_DIR, aligned_dir)

if not all_metrics:
    st.warning("No metrics data available.")
    st.stop()

# Overview cards
st.subheader("Overview")
cols = st.columns(len(all_metrics))
for col, (stem, m) in zip(cols, all_metrics.items()):
    with col:
        st.markdown(f"**{stem}**")
        st.metric("Segments", m["overflow"]["total"])
        st.metric("Overflow", f"{m['overflow']['overflow_pct']}%")
        st.metric("Expansion", f"{m['timing']['expansion_ratio']}x")
        if m.get("lufs") is not None:
            st.metric("LUFS", m["lufs"])

st.markdown("---")

# Overflow histogram
st.subheader("TTS Duration / Available Window Ratio")

all_ratios = []
for stem, m in all_metrics.items():
    for r in m["overflow"]["ratios"]:
        all_ratios.append({"Video": stem, "Ratio": r})

if all_ratios:
    df_ratios = pd.DataFrame(all_ratios)

    fig = px.histogram(
        df_ratios,
        x="Ratio",
        color="Video",
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        labels={"Ratio": "TTS Duration / Available Window"},
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Perfect fit")
    fig.add_vline(x=1.2, line_dash="dot", line_color="orange", annotation_text="Max stretch (1.2x)")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    total_segs = len(all_ratios)
    fits = sum(1 for r in all_ratios if r["Ratio"] <= 1.0)
    stretchable = sum(1 for r in all_ratios if 1.0 < r["Ratio"] <= 1.2)
    truncated = sum(1 for r in all_ratios if r["Ratio"] > 1.2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Fits in window", f"{fits} ({fits/total_segs*100:.0f}%)")
    col2.metric("Time-stretched", f"{stretchable} ({stretchable/total_segs*100:.0f}%)")
    col3.metric("Truncated", f"{truncated} ({truncated/total_segs*100:.0f}%)")

st.markdown("---")

# Speaker distribution
st.subheader("Speaker Distribution")

speaker_data = []
for stem, m in all_metrics.items():
    for speaker, count in m["speakers"]["speakers"].items():
        speaker_data.append({"Video": stem, "Speaker": speaker, "Segments": count})

if speaker_data:
    df_speakers = pd.DataFrame(speaker_data)
    fig = px.bar(
        df_speakers,
        x="Video",
        y="Segments",
        color="Speaker",
        barmode="group",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Timing comparison
st.subheader("Timing Comparison")

timing_data = []
for stem, m in all_metrics.items():
    timing_data.append({
        "Video": stem,
        "Original (s)": m["timing"]["total_original_duration"],
        "TTS (s)": m["timing"]["total_tts_duration"],
        "Expansion": f"{m['timing']['expansion_ratio']}x",
        "Short Segments (<0.5s)": m["timing"]["short_segments"],
    })

df_timing = pd.DataFrame(timing_data)
st.dataframe(df_timing, use_container_width=True, hide_index=True)
