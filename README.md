# 🎓 Lecture Video Maker

An AI-powered tool that converts any topic into a full lecture video with voiceover, stock visuals, and subtitles — all from a beautiful web UI or the terminal.

---

## ✨ Features

- **AI Script Generation** — Uses your local [Ollama](https://ollama.com) model to write a multi-segment lecture script
- **Natural Voiceover** — Microsoft Edge-TTS synthesizes speech (no API costs)
- **Stock Visuals** — Automatically fetches relevant videos and images from [Pexels](https://www.pexels.com/api/)
- **Subtitle Bar** — Pillow renders animated subtitle overlays (no libass needed)
- **Web UI** — Full browser-based interface with live progress tracking and video preview
- **CLI Mode** — Fully interactive terminal alternative

---

## 📋 Requirements

### System
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) (must be in PATH)
- [Ollama](https://ollama.com) running locally with at least one model pulled

### Python Packages
```bash
pip install edge-tts requests pillow numpy
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/lecture-video-maker.git
cd lecture-video-maker
```

### 2. Install Python dependencies
```bash
pip install edge-tts requests pillow numpy
```

### 3. Start Ollama and pull a model
```bash
ollama serve          # in a separate terminal
ollama pull llama3    # or mistral, gemma2, etc.
```

### 4. Get a free Pexels API key
Sign up at https://www.pexels.com/api/ — it's free.

### 5. Launch the Web UI
```bash
python lecture_gen.py
```
Opens automatically at **http://localhost:8080**

### Or use CLI mode
```bash
python lecture_gen.py --cli
```

---

## 🖥️ Web UI Walkthrough

1. **Settings Card** — Enter your Pexels API key, Ollama URL, and select a voice
2. **Topic & Parameters** — Set your lecture topic, choose duration and media preference
3. **Edit Script** — Review/edit the AI-generated narration per segment before rendering
4. **Render** — Watch live progress with log output
5. **Download** — Preview and download the final `.mp4`

---

## 📁 Output Structure

```
lecture_output/
├── <topic>_<timestamp>.mp4       ← final lecture video
└── <timestamp>/
    ├── script.txt                ← full generated script
    ├── audio/seg_00.mp3 …        ← TTS audio per segment
    ├── media/visual_00.mp4 …     ← downloaded Pexels media
    ├── subs/seg_00/frame_*.png   ← subtitle frames
    └── segments/seg_00.mp4 …     ← per-segment rendered clips
```

---

## ⚙️ Configuration

Settings are saved to `config.json` and persist between runs.

| Setting | Default | Description |
|---|---|---|
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `pexels_api_key` | `""` | Your Pexels API key |
| `edge_tts_voice` | `en-US-GuyNeural` | TTS voice ID |
| `output_dir` | `lecture_output` | Output directory |

### Available Voices

| Voice ID | Description |
|---|---|
| `en-US-GuyNeural` | US Male |
| `en-US-JennyNeural` | US Female |
| `en-GB-RyanNeural` | UK Male |
| `en-GB-SoniaNeural` | UK Female |
| `en-IN-NeerjaNeural` | Indian Female |
| `en-IN-PrabhatNeural` | Indian Male |
| `en-AU-NatashaNeural` | Australian Female |
| `en-AU-WilliamNeural` | Australian Male |

---

## 🛠️ Troubleshooting

**`ffmpeg` not found** → Install from https://ffmpeg.org/download.html and add to PATH

**Ollama not running** → Run `ollama serve` in a terminal, then `ollama pull <model>`

**Pexels returns no results** → Script falls back to a solid-colour background automatically

**Script generation returns invalid JSON** → Try a larger/smarter model (llama3, mistral, gemma2)

**Video too short/long** → Adjust the duration slider and/or the target model

---

## 📜 License

MIT
