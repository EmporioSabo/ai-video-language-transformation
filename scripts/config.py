"""Centralized configuration for the AI Video Language Transformation pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──
ROOT = Path(__file__).resolve().parent.parent

# ── Directory paths ──
DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
AUDIO_ORIGINAL_DIR = DATA_DIR / "audio" / "original"
VOICE_REF_DIR = DATA_DIR / "audio" / "voice_reference"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TRANSLATIONS_DIR = DATA_DIR / "translations"
TTS_DIR = DATA_DIR / "tts"
ALIGNED_DIR = DATA_DIR / "aligned"
OUTPUT_DIR = ROOT / "output"

# ── Source videos ──
GOOGLE_DRIVE_FOLDER_ID = "1fjbpwxk-Ub-SnierKNZ7kS8jUbMf_rZN"

VIDEO_FILES = {
    "video1_pr_within_team": "video1_pr_within_team.mp4",
    "video2_pr_across_teams": "video2_pr_across_teams.mp4",
    "video3_pr_conflicts": "video3_pr_conflicts.mp4",
}

# ── Audio extraction ──
WHISPER_SAMPLE_RATE = 16000  # Hz, mono, for Whisper input
REFERENCE_SAMPLE_RATE = 44100  # Hz, for voice reference / final output

# ── ASR (faster-whisper) ──
WHISPER_MODEL = "large-v3"
WHISPER_LANGUAGE = "zh"
WHISPER_BEAM_SIZE = 5
WHISPER_WORD_TIMESTAMPS = True

# ── Translation ──
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TARGET_LANGUAGE = "EN-US"  # DeepL target language code

GITHUB_GLOSSARY = {
    "拉取请求": "pull request",
    "合并": "merge",
    "分支": "branch",
    "派生": "fork",
    "克隆": "clone",
    "提交": "commit",
    "推送": "push",
    "拉取": "pull",
    "冲突": "conflict",
    "解决冲突": "resolve conflict",
    "仓库": "repository",
    "主分支": "main branch",
    "功能分支": "feature branch",
    "代码审查": "code review",
    "批准": "approve",
    "请求更改": "request changes",
    "工作流": "workflow",
    "远程": "remote",
    "暂存": "staging",
    "回退": "revert",
    "变基": "rebase",
}

# ── TTS (F5-TTS) ──
VOICE_REF_DURATION_SEC = 15  # Minimum clean voice sample for cloning

# ── Alignment ──
MIN_SPEED_FACTOR = 0.85  # Slowest allowed TTS playback
MAX_SPEED_FACTOR = 1.20  # Fastest allowed TTS playback
CROSSFADE_MS = 50  # Crossfade between segments
SILENCE_PADDING_MS = 200  # Default pause between sentences

# ── Audio post-processing ──
TARGET_LUFS = -14  # Broadcast loudness standard

# ── Output ──
OUTPUT_VIDEO_SUFFIX = "_EN"
OUTPUT_FORMAT = "mp4"
