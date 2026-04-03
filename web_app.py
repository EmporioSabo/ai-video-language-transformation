"""
Streamlit web app — Chinese → English Video Translator
Upload a Chinese video, get an English-dubbed version with the speaker's cloned voice.

Run with:
    streamlit run web_app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

import auth
import pipeline_core as pipeline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chinese → English Video Translator",
    page_icon="🎬",
    layout="centered",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎬 Chinese → English Video Translator")
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

# ── Logged-in header ──────────────────────────────────────────────────────────

st.title("🎬 Chinese → English Video Translator")
st.caption("Upload a Chinese video · get the same video in English with the speaker's cloned voice")

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.user_email}**")
    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.rerun()

# ── Sidebar — API keys ────────────────────────────────────────────────────────

with st.sidebar:
    st.divider()
    st.header("⚙️ API Keys")

    deepl_key = st.text_input(
        "DeepL API Key",
        type="password",
        placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx",
        help="Free key at deepl.com/pro-api (500,000 chars/month)",
    )
    elevenlabs_key = st.text_input(
        "ElevenLabs API Key",
        type="password",
        placeholder="sk_...",
        help="Voice cloning — requires Starter plan ($5/month) minimum",
    )

    st.divider()
    st.info(
        "**First use:** the Whisper 'small' model (~500 MB) "
        "will be downloaded automatically."
    )
    st.caption(
        "⏱ Estimated time: 2–3× the video duration "
        "(e.g. 5-min video → ~15 min processing)"
    )

# ── Session state ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None  # {"bytes": bytes, "filename": str}

if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# ── File upload ───────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "📁 Upload your Chinese video",
    type=["mp4", "mkv", "avi", "mov"],
)

# Reset result when a new file is uploaded
if uploaded and uploaded.file_id != st.session_state.last_file_id:
    st.session_state.result = None
    st.session_state.last_file_id = uploaded.file_id

if uploaded:
    st.video(uploaded)

# ── Guards ────────────────────────────────────────────────────────────────────

if uploaded and not (deepl_key and elevenlabs_key):
    st.warning("⚠️ Enter your API keys in the sidebar to continue.")

# ── Transform button ──────────────────────────────────────────────────────────

if uploaded and deepl_key and elevenlabs_key:
    if st.button("🚀 Transform Video", type="primary", use_container_width=True):

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write uploaded video to disk
            video_path = Path(tmp_dir) / uploaded.name
            video_path.write_bytes(uploaded.getvalue())

            result_bytes = None

            with st.status("Processing...", expanded=True) as status:
                try:
                    # Step 1 — Extract audio
                    st.write("🎵 Extracting audio...")
                    whisper_audio = pipeline.extract_whisper_audio(video_path, tmp_dir)
                    voice_sample = pipeline.extract_voice_sample(video_path, tmp_dir)

                    # Step 2 — Transcribe
                    st.write("📝 Transcribing Chinese (this may take a few minutes)...")
                    segments = pipeline.transcribe(whisper_audio)
                    st.write(f"   → {len(segments)} segments detected")

                    # Step 3 — Translate
                    st.write("🌐 Translating to English via DeepL...")
                    translated = pipeline.translate(segments, deepl_key)

                    # Step 4 — Voice cloning + synthesis
                    st.write("🎤 Cloning voice + synthesizing English audio...")
                    eng_audio = pipeline.synthesize(
                        voice_sample, translated, elevenlabs_key, tmp_dir
                    )

                    # Step 5 — Merge
                    st.write("🎬 Merging audio + video...")
                    out_path = pipeline.merge(video_path, eng_audio, tmp_dir)

                    # Read result into memory before tempdir is deleted
                    result_bytes = out_path.read_bytes()
                    status.update(label="✅ Done!", state="complete")

                except subprocess.CalledProcessError as e:
                    status.update(label="❌ FFmpeg error", state="error")
                    st.error(f"FFmpeg error:\n{e.stderr.decode()}")
                except Exception as e:
                    status.update(label=f"❌ Error: {e}", state="error")
                    st.exception(e)

            # Store result in session state for download button
            if result_bytes:
                st.session_state.result = {
                    "bytes": result_bytes,
                    "filename": f"{Path(uploaded.name).stem}_EN.mp4",
                }

# ── Download button ───────────────────────────────────────────────────────────

if st.session_state.result:
    st.success("✅ Your English video is ready!")
    st.download_button(
        label="⬇️ Download English video",
        data=st.session_state.result["bytes"],
        file_name=st.session_state.result["filename"],
        mime="video/mp4",
        use_container_width=True,
        type="primary",
    )
