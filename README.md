# AI Video Language Transformation

Pipeline for transforming Chinese-language videos into English audio versions using AI-powered transcription, translation, voice cloning, and audio alignment. Includes a Streamlit web interface for managing the pipeline, reviewing results, and monitoring quality metrics.

## Pipeline

```
Source Video → Extract Audio (FFmpeg)
  → Transcribe Chinese (faster-whisper large-v3, Colab GPU)
  → Speaker Diarization (pyannote.audio, Colab GPU)
  → Translate to English (DeepL free API + optional Gemini review)
  → Synthesize English Speech with Voice Cloning (F5-TTS, Colab GPU)
  → Align Segments with Time-Stretching (pyrubberband + LUFS normalization)
  → Merge Audio into Video (FFmpeg)
  → [Optional] Burn Subtitles
```

## Project Structure

```
├── app/
│   ├── streamlit_app.py          # Main Streamlit entry point
│   └── pages/
│       ├── 1_Upload.py           # Video upload & preview
│       ├── 2_Pipeline.py         # Pipeline dashboard with run controls
│       ├── 3_Review.py           # Segment review, audio playback, translation editor
│       └── 4_Metrics.py          # Quality charts (overflow, LUFS, speakers)
├── notebooks/
│   ├── 00_diarize.ipynb          # Speaker diarization (Colab GPU)
│   ├── 01_transcribe.ipynb       # Whisper ASR (Colab GPU)
│   └── 03_synthesize.ipynb       # F5-TTS voice cloning (Colab GPU)
├── scripts/
│   ├── config.py                 # Paths, API keys, thresholds
│   ├── pipeline.py               # End-to-end orchestration (CLI + importable)
│   ├── download_videos.py        # Fetch source videos
│   ├── extract_audio.py          # FFmpeg audio extraction
│   ├── translate.py              # DeepL + optional Gemini translation
│   ├── align_audio.py            # Time-stretching + LUFS normalization
│   ├── merge_video.py            # FFmpeg final muxing
│   ├── generate_subtitles.py     # SRT subtitle generation
│   └── metrics.py                # Quality metrics computation
├── data/
│   ├── source/                   # Original videos
│   ├── audio/original/           # Extracted Chinese audio
│   ├── audio/voice_reference/    # Per-speaker voice samples for cloning
│   ├── transcripts/              # Whisper JSON output
│   ├── translations/             # Translated JSON with speaker labels
│   ├── tts/                      # Per-segment TTS WAV files
│   └── aligned/                  # Time-aligned English audio tracks
└── output/                       # Final English videos
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in API keys in .env
```

System dependency (for audio time-stretching):
```bash
# Fedora/RHEL
sudo dnf install rubberband
# Ubuntu/Debian
sudo apt install rubberband-cli
# macOS
brew install rubberband
```

## Usage

### Streamlit Web Interface

```bash
streamlit run app/streamlit_app.py
```

The web interface provides:
- **Upload** — Add source videos and preview them
- **Pipeline** — Run each stage with visual status tracking
- **Review** — Inspect segments, listen to TTS audio, edit translations
- **Metrics** — View overflow stats, LUFS levels, speaker distribution charts

### Command Line

GPU-intensive stages run in Google Colab notebooks:

| Stage | Where | Notebook / Script |
|-------|-------|-------------------|
| Transcription | Colab GPU | `notebooks/01_transcribe.ipynb` |
| Speaker Diarization | Colab GPU | `notebooks/00_diarize.ipynb` |
| Translation | Local | `python scripts/translate.py` |
| Speech Synthesis | Colab GPU | `notebooks/03_synthesize.ipynb` |
| Audio Alignment | Local | `python scripts/align_audio.py` |
| Video Merge | Local | `python scripts/merge_video.py` |

Or run local stages together:

```bash
cd scripts
python pipeline.py                    # Full local pipeline
python pipeline.py --stage translate  # Individual stage
python pipeline.py --stage align
python pipeline.py --stage merge
python pipeline.py --stage subtitles
```

## Key Design Decisions

- **Per-speaker voice cloning**: Speaker diarization (pyannote) identifies speakers across all videos, then F5-TTS clones each speaker's voice separately for natural-sounding output.
- **Cross-video speaker matching**: Voice embeddings ensure consistent speaker labels (SPEAKER_00/01) across all videos using cosine similarity.
- **Pitch-preserving time-stretching**: Segments that overflow their time window are sped up using pyrubberband (WSOLA algorithm, up to 1.2x) to preserve pitch. Only segments exceeding max speedup are truncated with fade-out.
- **LUFS loudness normalization**: Final audio is normalized to -14 LUFS (broadcast standard) using pyloudnorm for consistent volume.
- **Resumable synthesis**: TTS skips segments where WAV files already exist, allowing interrupted Colab sessions to continue.

## Configuration

All pipeline settings are in `scripts/config.py`. API keys are loaded from `.env` (see `.env.example`).

Key parameters:
- `MAX_SPEED_FACTOR = 1.20` — Maximum time-stretch speedup before truncation
- `TARGET_LUFS = -14` — Broadcast loudness standard
- `CROSSFADE_MS = 50` — Fade-out duration for truncated segments

## Tools Used

| Tool | Purpose |
|------|---------|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Chinese speech-to-text (large-v3 model) |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker diarization and voice embeddings |
| [DeepL API](https://www.deepl.com/docs-api) | Chinese → English translation |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | Zero-shot voice cloning and speech synthesis |
| [pyrubberband](https://github.com/bmcfee/pyrubberband) | Pitch-preserving audio time-stretching |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS loudness normalization |
| [pydub](https://github.com/jiaaro/pydub) | Audio segment placement and manipulation |
| [FFmpeg](https://ffmpeg.org/) | Audio extraction and video muxing |
| [Streamlit](https://streamlit.io/) | Web interface for pipeline management |
| [Plotly](https://plotly.com/python/) | Interactive quality metric charts |
