"""AI Video Language Transformation — Streamlit Interface."""

import streamlit as st
import sys
from pathlib import Path

# Add scripts/ and project root to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import auth

st.set_page_config(
    page_title="AI Video Language Transformation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────

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

# ── Logged-in UI ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.user_email}**")
    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.rerun()

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
