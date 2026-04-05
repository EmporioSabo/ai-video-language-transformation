"""Page 5: Full Pipeline — upload a video, server handles everything via RunPod GPU."""

import sys
import time
from pathlib import Path

import streamlit as st

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import job_manager
import runpod_client
from pipeline_server import start_pipeline_async

st.header("Full Pipeline")
st.caption(
    "Upload a Chinese video and the server will handle the entire pipeline: "
    "transcription, diarization, translation, voice cloning, and merging."
)

# ── Cost warning ──────────────────────────────────────────────────────────────

st.warning(
    "GPU processing costs ~$0.05-0.10 per video (RunPod serverless). "
    "A daily job limit is enforced to prevent runaway costs."
)

# Show today's usage
usage = runpod_client.get_daily_usage()
st.caption(
    f"Today: {usage['job_count']}/{usage['limit']} jobs, "
    f"{usage['total_gpu_seconds']:.0f}s GPU time"
)

# ── Upload ────────────────────────────────────────────────────────────────────

uploaded_video = st.file_uploader(
    "Upload a Chinese video",
    type=["mp4", "mkv", "avi", "mov"],
    key="full_pipeline_video",
)

if uploaded_video:
    with st.expander("Preview"):
        st.video(uploaded_video)

# ── Launch pipeline ───────────────────────────────────────────────────────────

if "active_job_id" not in st.session_state:
    st.session_state.active_job_id = None

if uploaded_video and st.session_state.active_job_id is None:
    if st.button("Start Full Pipeline", type="primary", use_container_width=True):
        job_id = job_manager.create_job(uploaded_video.getvalue(), uploaded_video.name)
        st.session_state.active_job_id = job_id
        start_pipeline_async(job_id)
        st.rerun()

# ── Progress dashboard ────────────────────────────────────────────────────────

STAGE_LABELS = {
    "created": "Initializing...",
    "extract": "Extracting audio",
    "transcribe": "Transcribing Chinese (GPU)",
    "diarize": "Identifying speakers (GPU)",
    "translate": "Translating to English",
    "synthesize": "Synthesizing English speech (GPU)",
    "align": "Aligning audio segments",
    "merge": "Merging audio into video",
    "subtitles": "Generating subtitles",
    "complete": "Done!",
    "failed": "Failed",
}

if st.session_state.active_job_id:
    job_id = st.session_state.active_job_id
    status = job_manager.get_job_status(job_id)

    if status.get("error") and status.get("job_id") is None:
        st.error(f"Job not found: {job_id}")
        st.session_state.active_job_id = None
    else:
        stage = status.get("stage", "created")
        progress = status.get("progress", 0)
        label = STAGE_LABELS.get(stage, stage)

        st.subheader(f"Job: {job_id}")

        # Progress bar
        st.progress(progress / 100, text=label)

        # Stage details
        col1, col2, col3 = st.columns(3)
        col1.metric("Stage", label)
        col2.metric("Progress", f"{progress}%")
        elapsed = time.time() - status.get("created_at", time.time())
        col3.metric("Elapsed", f"{elapsed:.0f}s")

        if stage == "complete":
            st.success("Your English video is ready!")
            result_path = status.get("result_path")
            if result_path and Path(result_path).exists():
                result_bytes = Path(result_path).read_bytes()
                st.download_button(
                    label="Download English video",
                    data=result_bytes,
                    file_name=Path(result_path).name,
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary",
                )

                # Check for subtitles
                srt_path = Path(result_path).parent / f"{Path(result_path).stem.replace('_EN', '')}_en.srt"
                if srt_path.exists():
                    st.download_button(
                        label="Download subtitles (SRT)",
                        data=srt_path.read_bytes(),
                        file_name=srt_path.name,
                        mime="text/srt",
                        use_container_width=True,
                    )

            if st.button("Start new job"):
                st.session_state.active_job_id = None
                st.rerun()

        elif stage == "failed":
            st.error(f"Pipeline failed: {status.get('error', 'Unknown error')}")
            if st.button("Dismiss and start over"):
                st.session_state.active_job_id = None
                st.rerun()

        else:
            # Auto-refresh while processing
            time.sleep(3)
            st.rerun()

# ── Job history ───────────────────────────────────────────────────────────────

st.divider()
with st.expander("Job History"):
    jobs = job_manager.list_jobs()
    if not jobs:
        st.info("No jobs yet.")
    else:
        for job in jobs[:10]:
            jid = job.get("job_id", "?")
            jstage = job.get("stage", "?")
            jfile = job.get("filename", "?")
            icon = {"complete": "done", "failed": "error"}.get(jstage, "hourglass_flowing_sand")
            st.write(f"**{jid}** — {jfile} — {STAGE_LABELS.get(jstage, jstage)}")
