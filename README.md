---
title: AI Video Language Transformation
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.41.0"
app_file: webapp/streamlit_app.py
pinned: false
---

# AI Video Language Transformation

Pipeline for transforming Chinese-language videos into English audio versions using AI-powered transcription, translation, voice cloning, and audio alignment. Includes a Streamlit web interface for managing the pipeline, reviewing results, and monitoring quality metrics.

GitHub: [EmporioSabo/ai-video-language-transformation](https://github.com/EmporioSabo/ai-video-language-transformation)

## Pipeline

```
Source Video → Extract Audio (FFmpeg)
  → Transcribe Chinese (faster-whisper large-v3, Modal GPU)
  → Speaker Diarization (pyannote.audio, local CPU)
  → Translate to English (DeepL + optional Gemini review)
  → Synthesize English Speech with Voice Cloning (F5-TTS / Voxtral, Modal GPU)
  → Align Segments with Time-Stretching (pyrubberband + LUFS normalization)
  → Merge Audio into Video (FFmpeg)
  → [Optional] Burn Subtitles
```

## Project Structure

```
├── webapp/
│   ├── streamlit_app.py          # Main Streamlit entry point
│   └── pages/
│       ├── 1_Upload.py           # Video upload & preview
│       ├── 2_Pipeline.py         # Pipeline dashboard with run controls
│       ├── 3_Review.py           # Segment review, audio playback, translation editor
│       └── 4_Metrics.py          # Quality charts (overflow, LUFS, speakers)
├── notebooks/
│   ├── 00_diarize.ipynb          # Speaker diarization (Colab)
│   ├── 01_transcribe.ipynb       # Whisper ASR (Colab GPU)
│   ├── 02_translate.ipynb        # Translation (Colab)
│   └── 03_synthesize.ipynb       # F5-TTS voice cloning (Colab GPU)
├── scripts/
│   ├── config.py                 # Paths, API keys, thresholds
│   ├── pipeline.py               # End-to-end orchestration (CLI + importable)
│   ├── pipeline_server.py        # Server-side pipeline runner
│   ├── job_manager.py            # Async job tracking
│   ├── download_videos.py        # Fetch source videos
│   ├── extract_audio.py          # FFmpeg audio extraction
│   ├── diarize_local.py          # Local speaker diarization
│   ├── translate.py              # DeepL + optional Gemini translation
│   ├── translate_multilang.py    # Multilingual translation support
│   ├── synthesize_voxtral.py     # Voxtral-based speech synthesis
│   ├── transcribe_voxtral.py     # Voxtral-based transcription
│   ├── modal_client.py           # Client for Modal GPU endpoints
│   ├── runpod_client.py          # Client for RunPod GPU endpoints
│   ├── align_audio.py            # Time-stretching + LUFS normalization
│   ├── merge_video.py            # FFmpeg final muxing
│   ├── generate_subtitles.py     # SRT subtitle generation
│   └── metrics.py                # Quality metrics computation
├── modal_app.py                  # Modal serverless GPU endpoints (transcribe + synthesize)
├── 04_finetune.ipynb             # F5-TTS fine-tuning on Darija (Moroccan Arabic)
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
streamlit run webapp/streamlit_app.py
```

The web interface provides:
- **Upload** — Add source videos and preview them
- **Pipeline** — Run each stage with visual status tracking
- **Review** — Inspect segments, listen to TTS audio, edit translations
- **Metrics** — View overflow stats, LUFS levels, speaker distribution charts

### GPU Compute

GPU-intensive stages (transcription and synthesis) run on Modal serverless endpoints:

```bash
# Deploy Modal endpoints
modal deploy modal_app.py
# Copy the printed URLs to your .env as MODAL_TRANSCRIBE_URL and MODAL_SYNTHESIZE_URL
```

Colab notebooks are also available as an alternative for each stage:

| Stage | Notebook |
|-------|----------|
| Transcription | `notebooks/01_transcribe.ipynb` |
| Speaker Diarization | `notebooks/00_diarize.ipynb` |
| Translation | `notebooks/02_translate.ipynb` |
| Speech Synthesis | `notebooks/03_synthesize.ipynb` |

### Command Line

Local stages can be run directly:

```bash
cd scripts
python pipeline.py                    # Full local pipeline
python pipeline.py --stage translate  # Individual stage
python pipeline.py --stage align
python pipeline.py --stage merge
python pipeline.py --stage subtitles
```

## Key Design Decisions

- **Per-speaker voice cloning**: Speaker diarization (pyannote) identifies speakers, then F5-TTS clones each speaker's voice separately for natural-sounding output.
- **Pitch-preserving time-stretching**: Segments that overflow their time window are sped up using pyrubberband (WSOLA algorithm, up to 1.2x) to preserve pitch. Only segments exceeding max speedup are truncated with fade-out.
- **LUFS loudness normalization**: Final audio is normalized to -14 LUFS (broadcast standard) using pyloudnorm for consistent volume.
- **Resumable synthesis**: TTS skips segments where WAV files already exist, allowing interrupted sessions to continue.
- **Serverless GPU**: Modal endpoints handle transcription and synthesis on demand, with no persistent GPU cost between runs.

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
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker diarization |
| [DeepL API](https://www.deepl.com/docs-api) | Chinese → English translation |
| [Gemini](https://deepmind.google/technologies/gemini/) | Translation review for technical terminology |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | Zero-shot voice cloning and speech synthesis |
| [Voxtral](https://mistral.ai/news/voxtral) | Speech synthesis |
| [pyrubberband](https://github.com/bmcfee/pyrubberband) | Pitch-preserving audio time-stretching |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS loudness normalization |
| [pydub](https://github.com/jiaaro/pydub) | Audio segment placement and manipulation |
| [FFmpeg](https://ffmpeg.org/) | Audio extraction and video muxing |
| [Modal](https://modal.com/) | Serverless GPU infrastructure |
| [Streamlit](https://streamlit.io/) | Web interface for pipeline management |
| [Plotly](https://plotly.com/python/) | Interactive quality metric charts |

## Darija Fine-Tuning

`04_finetune.ipynb` documents an experiment fine-tuning F5-TTS on Moroccan Arabic (Darija) using the [DarijaTTS-clean](https://huggingface.co/datasets/KandirResearch/DarijaTTS-clean) dataset. The notebook covers vocabulary extension (adding Arabic characters to F5-TTS's base vocab), embedding layer resizing, and training on 20,000 samples. See the accompanying [blog post](blog_article.md) for a full writeup of the approach and findings.
