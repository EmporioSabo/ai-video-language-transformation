# AI Video Language Transformation

Pipeline for transforming Chinese-language GitHub tutorial videos into English audio versions using AI-powered transcription, translation, voice cloning, and audio alignment.

## Pipeline

```
Source Video → Extract Audio (FFmpeg)
  → Transcribe Chinese (faster-whisper large-v3, Colab GPU)
  → Speaker Diarization (pyannote.audio, Colab GPU)
  → Translate to English (DeepL free API + optional Gemini review)
  → Synthesize English Speech with Voice Cloning (F5-TTS, Colab GPU)
  → Align Segments to Original Timestamps (pydub)
  → Merge Audio into Video (FFmpeg)
  → [Optional] Burn Subtitles
```

## Project Structure

```
├── notebooks/
│   ├── 00_diarize.ipynb        # Speaker diarization (Colab GPU)
│   ├── 01_transcribe.ipynb     # Whisper ASR (Colab GPU)
│   └── 03_synthesize.ipynb     # F5-TTS voice cloning (Colab GPU)
├── scripts/
│   ├── config.py               # Paths, API keys, thresholds
│   ├── download_videos.py      # Fetch source videos
│   ├── extract_audio.py        # FFmpeg audio extraction
│   ├── translate.py            # DeepL + optional Gemini translation
│   ├── align_audio.py          # Segment alignment to timestamps
│   ├── merge_video.py          # FFmpeg final muxing
│   ├── generate_subtitles.py   # SRT subtitle generation
│   └── pipeline.py             # End-to-end orchestration
├── data/
│   ├── source/                 # Original videos
│   ├── audio/original/         # Extracted Chinese audio
│   ├── audio/voice_reference/  # Per-speaker voice samples for cloning
│   ├── transcripts/            # Whisper JSON output
│   ├── translations/           # Translated JSON with speaker labels
│   ├── tts/                    # Per-segment TTS WAV files
│   └── aligned/                # Time-aligned English audio tracks
└── output/                     # Final English videos
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in API keys in .env
```

## Usage

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
- **No time-stretching**: Segments are placed at original timestamps and truncated with fade-out if they overflow. Speed adjustments during synthesis degraded audio quality, so natural-speed generation is used instead.
- **Resumable synthesis**: TTS skips segments where WAV files already exist, allowing interrupted sessions to continue.

## Configuration

All pipeline settings are in `scripts/config.py`. API keys are loaded from `.env` (see `.env.example`).

## Tools Used

| Tool | Purpose |
|------|---------|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Chinese speech-to-text (large-v3 model) |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker diarization and voice embeddings |
| [DeepL API](https://www.deepl.com/docs-api) | Chinese → English translation |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | Zero-shot voice cloning and speech synthesis |
| [pydub](https://github.com/jiaaro/pydub) | Audio segment alignment |
| [FFmpeg](https://ffmpeg.org/) | Audio extraction and video muxing |
