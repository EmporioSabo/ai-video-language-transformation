"""AI Video Language Transformation — User-facing webapp (deployed on server)."""

import sys
import time
from pathlib import Path

import streamlit as st

# Add scripts/ and project root to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import auth
import job_manager
import runpod_client
from pipeline_server import start_pipeline_async

st.set_page_config(
    page_title="AI Video Language Transformation",
    page_icon="🎬",
    layout="wide",
)

# ── Supported languages ─────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "Chinese": "zh",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Dutch": "nl",
    "Turkish": "tr",
    "Polish": "pl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Greek": "el",
    "Czech": "cs",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Indonesian": "id",
}

# ── Auth gate ────────────────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎬 AI Video Language Transformation")
    st.write("Please sign in or create an account to continue.")
    st.divider()

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In", type="primary", use_container_width=True):
            ok, msg = auth.log_in(login_email, login_password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user_email = login_email
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_password2 = st.text_input("Confirm password", type="password", key="signup_password2")
        if st.button("Create Account", type="primary", use_container_width=True):
            if signup_password != signup_password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = auth.sign_up(signup_email, signup_password)
                if ok:
                    st.success(msg + " You can now log in.")
                else:
                    st.error(msg)

    st.stop()

# ── Logged-in UI ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.user_email}**")
    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.rerun()

st.title("🎬 AI Video Language Transformation")
st.markdown(
    "Upload a video in any supported language and the server will dub it into English automatically."
)

st.divider()

# ── Language selection ───────────────────────────────────────────────────────

lang_names = sorted(SUPPORTED_LANGUAGES.keys())
selected_lang_name = st.selectbox(
    "Source language",
    lang_names,
    index=lang_names.index("Chinese"),
    help="Language spoken in the video.",
)
selected_lang_code = SUPPORTED_LANGUAGES[selected_lang_name]

# ── TTS model selection ─────────────────────────────────────────────────────

TTS_MODELS = {
    "F5-TTS (RunPod GPU — better voice cloning)": "f5tts",
    "Voxtral (Mistral API — fast, no GPU)": "voxtral",
}

selected_tts_label = st.selectbox(
    "TTS Model",
    list(TTS_MODELS.keys()),
    index=0,
    help="Voxtral uses Mistral's API (fast, $0.016/1K chars). "
         "F5-TTS runs on RunPod GPU (better cross-lingual voice cloning).",
)
selected_tts_model = TTS_MODELS[selected_tts_label]

# ── Cost warning ─────────────────────────────────────────────────────────────

st.warning(
    "Processing costs ~$0.05-0.15 per video (RunPod GPU + TTS API). "
    "A daily job limit is enforced to prevent runaway costs."
)

usage = runpod_client.get_daily_usage()
st.caption(
    f"Today: {usage['job_count']}/{usage['limit']} jobs, "
    f"{usage['total_gpu_seconds']:.0f}s GPU time"
)

# ── Upload ───────────────────────────────────────────────────────────────────

uploaded_video = st.file_uploader(
    f"Upload a {selected_lang_name} video",
    type=["mp4", "mkv", "avi", "mov"],
    key="full_pipeline_video",
)

if uploaded_video:
    with st.expander("Preview"):
        st.video(uploaded_video)

# ── Launch pipeline ──────────────────────────────────────────────────────────

if "active_job_id" not in st.session_state:
    st.session_state.active_job_id = None

if uploaded_video and st.session_state.active_job_id is None:
    if st.button("Start Full Pipeline", type="primary", use_container_width=True):
        job_id = job_manager.create_job(
            uploaded_video.getvalue(), uploaded_video.name,
            language=selected_lang_code, tts_model=selected_tts_model,
        )
        st.session_state.active_job_id = job_id
        start_pipeline_async(job_id)
        st.rerun()

# ── Progress dashboard ───────────────────────────────────────────────────────

STAGE_LABELS = {
    "created": "Initializing...",
    "extract": "Extracting audio",
    "transcribe": "Transcribing audio (GPU)",
    "diarize": "Identifying speakers",
    "translate": "Translating to English",
    "synthesize": "Synthesizing English speech",
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

# ── Job history ──────────────────────────────────────────────────────────────

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
            jlang = job.get("language", "zh")
            st.write(f"**{jid}** — {jfile} ({jlang}) — {STAGE_LABELS.get(jstage, jstage)}")
