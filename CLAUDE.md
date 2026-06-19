# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language video/audio transcription tool that converts media files to text (.txt) and subtitles (.srt) using `faster-whisper` (local Whisper model with CTranslate2). Supports GPU (CUDA) acceleration. Purely local, no network API calls.

The UI is in Chinese. The project uses a Hugging Face China mirror (`hf-mirror.com`) for model downloads.

## Running

All commands assume the project root as working directory. The Python venv is at `.venv/`.

**GUI mode (the only mode):**
```
.venv\Scripts\pythonw.exe transcribe_gui.pyw
```

**Install dependencies:**
```
.venv\Scripts\pip.exe install -r requirements.txt
```

**Run tests:**
```
.venv\Scripts\python.exe -m pytest test_transcribe_core.py -v
```

**Build Windows exe (PyInstaller onefile, 内置 medium 模型):**
```
.venv\Scripts\python.exe build.py
```
产物在 `dist/语音转文字.exe`。前置条件：先用 medium 模型运行一次触发下载到 HF cache。

## Architecture

Three Python files with clear separation of concerns:

- **`transcribe_core.py`** — Core module. Environment setup (HF mirror, CUDA DLL paths), device detection, model loading with GPU fallback + bundled-model path resolution, file discovery, and `transcribe_file()` which executes transcription and returns results. No UI code.
- **`transcribe_gui.pyw`** — tkinter GUI. Imports `transcribe_core`. Runs transcription in a background daemon thread. Features: file/folder selection, model/device/language/VAD settings with persistence, skip-processed-files toggle, real-time progress with percentage, stop button, per-file error isolation.
- **`build.py`** — Packaging script. Copies medium model from HF cache to `bundled_models/`, then calls PyInstaller to produce a single exe.
- **`test_transcribe_core.py`** — Unit tests for subtitle splitting/merging/timing functions.

## Key Configuration

- HF mirror: `https://hf-mirror.com` (set in `transcribe_core.py`, override via `HF_ENDPOINT` env var)
- Default VAD params: `min_silence_duration_ms=700`, `speech_pad_ms=300`, `threshold=0.50` — all adjustable via GUI
- CUDA DLL paths are prepended to `PATH` on Windows in `transcribe_core.py`
- Default Whisper model: `medium`; default language: auto-detect; default variant: 简体中文
- Supported formats: mp4, mkv, avi, mov, webm, flv, wmv, m4v, ts, mp3, wav, flac, ogg, m4a, aac, wma, opus, ape, aiff
- Version: `__version__` in `transcribe_gui.pyw` (currently `1.0.0`)
- Settings persistence: `%APPDATA%/transcribe/settings.json` (saved on start/close)
- Log file: `%TEMP%/transcribe_gui.log` (RotatingFileHandler, 2MB×2)
- Bundled model: PyInstaller exe 内置 medium 模型（`bundled_models/medium/`），其他模型首次使用从 HF 下载
- OpenCC: 简体走 t2s、繁体走 s2t 后处理，统一字符变体（`opencc-python-reimplemented`）
