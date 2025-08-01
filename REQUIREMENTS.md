# VHS Video Summarizer

This tool processes long, digitized VHS `.mp4` video files to generate a **high-level narrative summary** of their contents. It combines **video scene analysis** with **audio transcription** to help users quickly understand what’s happening in legacy footage — without watching the entire video.

---

## 🎯 Purpose

Many old VHS videos have been digitized but not reviewed due to their length and unknown content. This tool provides an efficient way to extract meaningful insights by analyzing:

- **Visual content** (scene-level representative frames)
- **Audio dialogue or ambient sound** (via Whisper)

The output is a 2–3 paragraph **natural language summary** that captures the main story or activity in the video.

---

## 📥 Inputs

- A single `.mp4` file (e.g. `my-video.mp4`)
- A vision-capable LLM (e.g. `gemma3:12b`) configured in the `llm` CLI
- Optional: Whisper model for audio transcription (`llm video-frames transcribe`)

---

## 📤 Outputs

- `frame_captions.txt`: Timestamped scene-by-scene visual descriptions
- `whisper_transcript.txt`: Full audio transcript of the video (optional)
- `video_summary.md`: Final narrative summary of what the video shows

---

## 🧠 How It Works

### 1. Scene Detection

- Uses [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) to detect visual scene changes.
- Extracts one frame per scene (usually the midpoint).

### 2. Frame Captioning

- Each frame is described using a **vision-language model** (e.g. Gemma3:12b).
- Each caption is paired with its timestamp for context.

### 3. Audio Transcription (Optional)

- Transcribes the video audio using Whisper.
- Adds semantic detail that visual inspection alone might miss.

### 4. Narrative Summary Generation

- Passes the list of timestamped frame descriptions and the transcript to a language model (e.g. GPT-4).
- Generates a 2–3 paragraph **story-style summary** of the video’s contents.

---

## ⚙️ Dependencies

- `uv` package manager for Python dependency management
- `scenedetect[opencv]` (via `uv add` or `uv tool install`)
- [`llm`](https://github.com/simonw/llm) CLI with:
  - `llava` plugin for vision models
  - Whisper access for transcription
  - A configured LLM such as `gemma3:12b` and/or `gpt-4`

---

## 🧪 Example Usage

```bash
python summarize_video_with_scenes.py
