"""
Streamlit web app — Chinese → English Video Translator
Upload a Chinese video, get an English-dubbed version with the speaker's cloned voice.

Run with:
    streamlit run web_app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

import pipeline_core as pipeline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Traducteur Vidéo Chinois → Anglais",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Traducteur Vidéo Chinois → Anglais")
st.caption("Upload une vidéo chinoise · reçois la même vidéo en anglais avec la voix clonée du locuteur")

# ── Sidebar — API keys ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Clés API")

    deepl_key = st.text_input(
        "DeepL API Key",
        type="password",
        placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx",
        help="Clé gratuite sur deepl.com/fr/pro-api (500 000 chars/mois)",
    )
    elevenlabs_key = st.text_input(
        "ElevenLabs API Key",
        type="password",
        placeholder="sk_...",
        help="Clonage de voix — requiert le plan Starter ($5/mois) minimum",
    )

    st.divider()
    st.info(
        "**Première utilisation :** le modèle Whisper 'small' (~500 MB) "
        "sera téléchargé automatiquement."
    )
    st.caption(
        "⏱ Temps estimé : 2–3× la durée de la vidéo "
        "(ex : vidéo de 5 min → ~15 min de traitement)"
    )

# ── Session state ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None  # {"bytes": bytes, "filename": str}

if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# ── File upload ───────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "📁 Upload ta vidéo chinoise",
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
    st.warning("⚠️ Entre tes clés API dans la barre latérale pour continuer.")

# ── Transform button ──────────────────────────────────────────────────────────

if uploaded and deepl_key and elevenlabs_key:
    if st.button("🚀 Transformer la vidéo", type="primary", use_container_width=True):

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write uploaded video to disk
            video_path = Path(tmp_dir) / uploaded.name
            video_path.write_bytes(uploaded.getvalue())

            result_bytes = None

            with st.status("Traitement en cours...", expanded=True) as status:
                try:
                    # Step 1 — Extract audio
                    st.write("🎵 Extraction de l'audio...")
                    whisper_audio = pipeline.extract_whisper_audio(video_path, tmp_dir)
                    voice_sample = pipeline.extract_voice_sample(video_path, tmp_dir)

                    # Step 2 — Transcribe
                    st.write("📝 Transcription du chinois (peut prendre plusieurs minutes)...")
                    segments = pipeline.transcribe(whisper_audio)
                    st.write(f"   → {len(segments)} segments détectés")

                    # Step 3 — Translate
                    st.write("🌐 Traduction en anglais via DeepL...")
                    translated = pipeline.translate(segments, deepl_key)

                    # Step 4 — Voice cloning + synthesis
                    st.write("🎤 Clonage de la voix + synthèse en anglais...")
                    eng_audio = pipeline.synthesize(
                        voice_sample, translated, elevenlabs_key, tmp_dir
                    )

                    # Step 5 — Merge
                    st.write("🎬 Fusion audio + vidéo...")
                    out_path = pipeline.merge(video_path, eng_audio, tmp_dir)

                    # Read result into memory before tempdir is deleted
                    result_bytes = out_path.read_bytes()
                    status.update(label="✅ Transformation terminée !", state="complete")

                except subprocess.CalledProcessError as e:
                    status.update(label="❌ Erreur FFmpeg", state="error")
                    st.error(f"FFmpeg error:\n{e.stderr.decode()}")
                except Exception as e:
                    status.update(label=f"❌ Erreur : {e}", state="error")
                    st.exception(e)

            # Store result in session state for download button
            if result_bytes:
                st.session_state.result = {
                    "bytes": result_bytes,
                    "filename": f"{Path(uploaded.name).stem}_EN.mp4",
                }

# ── Download button ───────────────────────────────────────────────────────────

if st.session_state.result:
    st.success("✅ Ta vidéo en anglais est prête !")
    st.download_button(
        label="⬇️ Télécharger la vidéo en anglais",
        data=st.session_state.result["bytes"],
        file_name=st.session_state.result["filename"],
        mime="video/mp4",
        use_container_width=True,
        type="primary",
    )
