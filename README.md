# AI Video Language Transformation

Pipeline for transforming Chinese-language instructional videos into English audio versions with optional bilingual subtitles.

## Pipeline

```
Source Video → Extract Audio (FFmpeg)
  → Transcribe Chinese (Whisper large-v3)
  → Translate to English (DeepL + Gemini review)
  → Synthesize English Speech with Voice Cloning (F5-TTS)
  → Align & Time-Stretch Segments
  → Merge Audio into Video (FFmpeg)
  → [Optional] Subtitles
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in API keys in .env
```

## Usage

```bash
cd scripts

# Full pipeline
python pipeline.py

# Individual stages
python pipeline.py --stage download
python pipeline.py --stage extract
python pipeline.py --stage translate
python pipeline.py --stage align
python pipeline.py --stage merge
python pipeline.py --stage subtitles
```

GPU-intensive stages (transcription, speech synthesis) run in Google Colab:
- `notebooks/01_transcribe.ipynb` — Whisper ASR
- `notebooks/03_synthesize.ipynb` — F5-TTS voice cloning

## Configuration

All pipeline settings are in `scripts/config.py`. API keys are loaded from `.env` (see `.env.example`).
