# 🎬 VidRush AI Video Engine

**An end-to-end autonomous AI video production system** that takes a text prompt, analyzes a video library using computer vision, generates professional voiceover, and produces perfectly synchronized videos.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/mustafabasar7/vidrush)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-orange)
![MoviePy](https://img.shields.io/badge/Video-MoviePy%20v2-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🚀 Live Demo

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/mustafabasar7/vidrush)**

Enter your own API keys in the UI to enable full AI features, or run in demo mode without keys.

## ✨ Features

- **Vision-Based Clip Selection**: Gemini 2.0 Flash "sees" video content via keyframe analysis
- **Professional TTS**: Google Cloud TTS or free Edge-TTS fallback
- **Perfect Audio-Video Sync**: Video loops/trims to match audio duration exactly
- **Proxy Editing System**: 4K→480p downscaling for 10x faster rendering
- **No API Keys Required**: Demo mode works without any keys (Edge-TTS + first video selection)
- **Bring Your Own Keys**: Enter API keys directly in the UI

## 📐 System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PROMPT    │ ──▶ │   VISION    │ ──▶ │    AUDIO    │ ──▶ │  ASSEMBLY   │ ──▶ │   OUTPUT    │
│   (Text)    │     │  (Gemini)   │     │   (TTS)     │     │  (MoviePy)  │     │   (Video)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     📝                 👁️                  🎙️                 ✂️                  🎬
```

## 🛠️ Local Installation

```bash
# Clone the repository
git clone https://github.com/mustafabasar7/Portfolio.git
cd Portfolio/vidrush

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
# Opens at http://localhost:7860
```

## 🔑 API Keys (Optional)

| Key | Purpose | Get it from |
|-----|---------|-------------|
| Gemini API | Vision analysis | [aistudio.google.com](https://aistudio.google.com/) |
| Google Cloud TTS | Professional voice | [console.cloud.google.com](https://console.cloud.google.com/) |

Without keys, the app uses:
- **Demo mode**: First video auto-selected
- **Edge-TTS**: Free Microsoft text-to-speech

## 📁 Adding Your Videos

Place `.mp4` files in the same folder as `app.py`. The engine will auto-detect them.

## 🚀 Deploy Your Own

### Hugging Face Spaces (Recommended)
1. Fork this repo
2. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
3. Upload `app.py` and `requirements.txt`
4. Add sample videos to the Space

### Docker
```bash
docker build -t vidrush .
docker run -p 7860:7860 vidrush
```

## 📄 License

MIT License - free to use and modify.

## 👤 Author

**Mustafa Başar**
- GitHub: [@mustafabasar7](https://github.com/mustafabasar7)
- Email: mustafabasar7@gmail.com

---

*Built with ❤️ using Gemini AI, MoviePy, and Gradio*
