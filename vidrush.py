import os
import json
import time
import subprocess
import datetime
import sys
import argparse
import re
import asyncio

import google.generativeai as genai
from dotenv import load_dotenv
import edge_tts
import requests
import base64
from PIL import Image

# MoviePy v2 imports (from Context7 documentation)
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip, concatenate_videoclips
from moviepy.video.fx import Loop, Resize

# Load environment variables from .env
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

class VidrushEngine:
    def __init__(self, stock_folder="."):
        self.stock_folder = stock_folder
        self.assets_dir = os.path.join(stock_folder, "assets")
        self.temp_dir = os.path.join(stock_folder, "temp")
        self.thumb_dir = os.path.join(self.assets_dir, "thumbnails")
        self.proxy_dir = os.path.join(self.assets_dir, "proxies")
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)
        os.makedirs(self.proxy_dir, exist_ok=True)
        self.ffmpeg_path = self._find_ffmpeg()
        
        # Google TTS Config via API KEY
        self.google_api_key = os.getenv("GOOGLE_API_KEY") 
        if self.google_api_key:
            print("Using Google Cloud TTS (REST API) with API Key...")
        else:
            print("GOOGLE_API_KEY not found in .env, falling back to Edge-TTS.")

    def _find_ffmpeg(self):
        """Checks if ffmpeg is in PATH or in local directory."""
        if subprocess.run(["where.exe", "ffmpeg"], capture_output=True).returncode == 0:
            return "ffmpeg"
        local_ffmpeg = os.path.join(os.getcwd(), "ffmpeg.exe")
        if os.path.exists(local_ffmpeg):
            return local_ffmpeg
        return "ffmpeg"

    def _wrap_text(self, text, width=25):
        """Wraps text into lines of a maximum length."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        lines.append(" ".join(current_line))
        return "\n".join(lines)

    def _update_progress(self, progress, value, desc):
        """Safely updates Gradio progress bar without crashing."""
        if progress is None:
            return
        try:
            progress(value, desc=desc)
        except Exception:
            print(f"[Progress] {desc}")

    async def _extract_keyframes(self, progress=None):
        """Extracts high-quality I-frames for AI analysis. Non-blocking."""
        stock_videos = self.get_stock_videos()
        indexed_content = []
        
        if not stock_videos:
            return []

        total = len(stock_videos)
        for i, v in enumerate(stock_videos):
            self._update_progress(progress, (i + 0.5) / total, f"Görsel çıkarılıyor: {v}")
            
            v_path = os.path.join(self.stock_folder, v)
            thumb_name = f"{v}.jpg"
            thumb_path = os.path.join(self.thumb_dir, thumb_name)
            
            try:
                if not os.path.exists(thumb_path):
                    cmd = [
                        self.ffmpeg_path, "-y", "-ss", "0.5", "-i", v_path,
                        "-vframes", "1", "-vf", "scale=320:-1",
                        thumb_path
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=15)
                
                if os.path.exists(thumb_path):
                    img = Image.open(thumb_path)
                    indexed_content.append({"filename": v, "image": img, "path": thumb_path})
            except Exception as e:
                print(f"Error indexing {v}: {e}")
                
        return indexed_content

    async def generate_script(self, prompt, progress=None):
        """Uses Gemini 2.0 Flash VISION to analyze images and write a synchronized script."""
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        self._update_progress(progress, 0.1, "Video kütüphanesi taranıyor...")
        video_indexes = await self._extract_keyframes(progress)
        
        if not video_indexes:
            return [], []

        self._update_progress(progress, 0.3, "Gemini Vision kurgu planlıyor...")
        parts = [f"Sen profesyonel video yönetmenisin. Kütüphanemdeki videoları görsellere bakarak analiz et. Konu: '{prompt}'"]
        for idx, item in enumerate(video_indexes):
            parts.append(f"KLİP {idx} (Dosya: {item['filename']}):")
            parts.append(item['image'])
            
        parts.append("""
        Senaryoyu kurgula. 'video_file' tam adını yaz. 'reasoning' kısmına görsel analizini ekle.
        Return ONLY JSON array of: {"text": "...", "video_file": "...", "reasoning": "..."}
        """)
        
        try:
            response = model.generate_content(parts)
            text = response.text
            # Robust JSON extraction using regex
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                content = text.strip()
            
            # Clean common AI noise
            content = content.replace("```json", "").replace("```", "").replace("JSON", "").strip()
            scenes = json.loads(content)
            return scenes, [item['path'] for item in video_indexes]
        except Exception as e:
            print(f"AI Script Error: {e}")
            # Emergency Fallback
            fallback = [{"text": f"Görsel analiz sonucunda kurgulanan video. (Prompt: {prompt})", 
                         "video_file": video_indexes[0]['filename'], 
                         "reasoning": "AI yanıtı işlenemedi, kütüphanedeki ilk video seçildi."}]
            return fallback, [item['path'] for item in video_indexes]

    async def generate_audio(self, scenes):
        """Generates High-Quality audio using Google TTS REST API or Edge-TTS."""
        for i, scene in enumerate(scenes):
            audio_path = os.path.join(self.temp_dir, f"scene_{i}.mp3")
            
            if self.google_api_key:
                print(f"Synthesizing scene {i} with Google Cloud TTS...")
                url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_api_key}"
                payload = {
                    "input": {"text": scene['text']},
                    "voice": {"languageCode": "tr-TR", "name": "tr-TR-Standard-A"},
                    "audioConfig": {"audioEncoding": "MP3"}
                }
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    audio_content = response.json().get("audioContent")
                    with open(audio_path, "wb") as out:
                        out.write(base64.b64decode(audio_content))
                else:
                    print(f"Google TTS Error: {response.text}")
                    VOICE = "tr-TR-AhmetNeural"
                    communicate = edge_tts.Communicate(scene['text'], VOICE)
                    await communicate.save(audio_path)
            else:
                VOICE = "tr-TR-AhmetNeural"
                communicate = edge_tts.Communicate(scene['text'], VOICE)
                await communicate.save(audio_path)
            
            audio_clip = AudioFileClip(audio_path)
            scene['duration'] = audio_clip.duration
            scene['audio_path'] = audio_path
            audio_clip.close()
            
        return scenes

    def _format_srt_time(self, seconds):
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def produce_srt(self, scenes):
        """Generates an SRT file for the entire video."""
        srt_path = os.path.join(self.temp_dir, "subtitles.srt")
        current_time = 0.0
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, scene in enumerate(scenes):
                start = self._format_srt_time(current_time)
                end = self._format_srt_time(current_time + scene['duration'])
                f.write(f"{i+1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{scene['text']}\n\n")
                current_time += scene['duration']
        return srt_path

    def get_stock_videos(self):
        """Lists available mp4 files in the stock folder, excluding output files."""
        return [f for f in os.listdir(self.stock_folder) 
                if f.endswith(".mp4") and not f.startswith("vidrush_")]

    def _get_proxy_video(self, filename, progress=None):
        """Creates or returns a low-res proxy version of a large video for fast rendering."""
        original_path = os.path.join(self.stock_folder, filename)
        proxy_path = os.path.join(self.proxy_dir, f"proxy_{filename}")
        
        if not os.path.exists(proxy_path):
            self._update_progress(progress, None, f"Proxy üretiliyor: {filename}...")
            cmd = [
                self.ffmpeg_path, "-y", "-ss", "0", "-i", original_path,
                "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
                "-an", "-t", "60",
                proxy_path
            ]
            subprocess.run(cmd, capture_output=True)
        return proxy_path

    def assemble_video(self, scenes, progress=None):
        """High-Sync assembly using MoviePy v2 syntax (from Context7 docs)."""
        final_clips = []
        total = len(scenes)
        
        TARGET_H = 960
        TARGET_W = 540
        
        for i, scene in enumerate(scenes):
            self._update_progress(progress, (i+1)/total, f"Sahne kurgulanıyor {i+1}/{total}...")
            
            video_file = scene.get('video_file')
            if not video_file:
                print(f"Warning: Scene {i} has no video_file, skipping.")
                continue
                
            video_path = self._get_proxy_video(video_file, progress)
            audio_path = scene.get('audio_path')
            
            if not audio_path or not os.path.exists(audio_path):
                print(f"Warning: Scene {i} has no audio, skipping.")
                continue
            
            try:
                # Load assets
                v_clip = VideoFileClip(video_path)
                a_clip = AudioFileClip(audio_path)
                
                # MoviePy v2: Use .resized() method directly
                v_clip = v_clip.resized(height=TARGET_H)
                w, h = v_clip.size
                
                # Crop to vertical format
                if w > TARGET_W:
                    v_clip = v_clip.cropped(x_center=w/2, y_center=h/2, width=TARGET_W, height=TARGET_H)
                
                # PERFECT SYNC: Match video duration to audio duration
                target_duration = a_clip.duration
                
                if v_clip.duration < target_duration:
                    # MoviePy v2 Loop (from Context7): Use Loop effect with duration parameter
                    v_clip = v_clip.with_effects([Loop(duration=target_duration)])
                else:
                    v_clip = v_clip.with_duration(target_duration)
                
                # Note: Ken Burns effect removed for stability. Add back if needed:
                # v_clip = v_clip.resized(lambda t: 1 + 0.03 * t / target_duration)
                
                # Attach Audio
                v_clip = v_clip.with_audio(a_clip)
                
                # Professional Subtitles
                font_path = "C:/Windows/Fonts/arialbd.ttf"
                if not os.path.exists(font_path): 
                    font_path = "C:/Windows/Fonts/arial.ttf"
                
                try:
                    wrapped_text = self._wrap_text(scene['text'], width=20)
                    txt_clip = TextClip(
                        text=wrapped_text,
                        font=font_path,
                        font_size=32,
                        color='yellow',
                        bg_color='black',
                        text_align="center",
                        method='caption',
                        size=(TARGET_W - 40, None)
                    ).with_duration(target_duration).with_position(('center', TARGET_H - 200))
                    
                    v_clip = CompositeVideoClip([v_clip, txt_clip])
                except Exception as e:
                    print(f"Subtitle error on scene {i}: {e}")
                
                # Add crossfade transition
                if i > 0:
                    v_clip = v_clip.with_effects([])  # Placeholder for future transitions
                
                final_clips.append(v_clip)
                
            except Exception as e:
                print(f"Error assembling scene {i}: {e}")
                continue

        if not final_clips:
            print("ERROR: No clips were assembled!")
            return None

        print("🎬 Final Render starting...")
        final_video = concatenate_videoclips(final_clips, method="compose")
        
        output_name = "vidrush_pro_sync.mp4"
        final_video.write_videofile(
            output_name, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            threads=os.cpu_count(),
            logger=None
        )
        
        # Cleanup
        for c in final_clips: 
            try:
                c.close()
            except:
                pass
        final_video.close()
        
        return output_name


async def main():
    import gradio as gr
    
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=None, help="Video topic for CLI mode")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode")
    args = parser.parse_args()
    
    engine = VidrushEngine()

    async def run_process(prompt, progress=None):
        if not prompt:
            return None, "Lütfen bir prompt girin.", []
        
        scenes, thumbs = await engine.generate_script(prompt, progress)
        
        if not scenes:
            return None, "Senaryo oluşturulamadı.", thumbs
            
        engine._update_progress(progress, 0.5, "Sesler üretiliyor...")
        scenes = await engine.generate_audio(scenes)
        engine.produce_srt(scenes)
        
        engine._update_progress(progress, 0.7, "Video montajı yapılıyor...")
        output_video = engine.assemble_video(scenes, progress)
        
        if not output_video:
            return None, "Video birleştirilemedi.", thumbs
        
        reasoning_md = "### 🧠 AI Kurgu Analizi & Senkronizasyon Mantığı\n\n"
        for i, scene in enumerate(scenes):
            reasoning_md += f"**Sahne {i+1}:** {scene.get('video_file')}\n"
            reasoning_md += f"> *{scene.get('reasoning', 'Eşleşen görsel seçildi.')}*\n\n"
        
        return output_video, reasoning_md, thumbs

    # CLI Mode
    if (args.prompt or args.cli):
        prompt = args.prompt or "Kahve tadımı"
        output, _, _ = await run_process(prompt, None)
        print(f"Done! Video saved as {output}")
        return

    # Gradio UI Mode (from Context7 docs)
    def ui_wrapper(prompt, progress=gr.Progress()):
        """Wrapper that safely passes progress to async function."""
        async def async_run():
            return await run_process(prompt, progress)
        return asyncio.run(async_run())

    def index_library(progress=gr.Progress()):
        """Index the video library with progress feedback."""
        async def async_index():
            engine._update_progress(progress, 0.1, "Kütüphane taranıyor...")
            indexed = await engine._extract_keyframes(progress)
            thumbs = [item['path'] for item in indexed]
            return thumbs
        return asyncio.run(async_index())

    with gr.Blocks(title="Vidrush AI Video Sync", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎥 Vidrush AI Video Sync")
        gr.Markdown("Yapay zeka videolarınızı analiz eder ve konunuza en uygun senaryoyu kurgulayarak seslendirir.")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(label="Video Konusu / Prompt", placeholder="Örn: Espresso nasıl yapılır?", lines=3)
                with gr.Row():
                    index_btn = gr.Button("🔍 Kütüphaneyi Tara", variant="secondary")
                    generate_btn = gr.Button("🚀 Videoyu Oluştur", variant="primary")
                
                gr.Markdown("### 🖼️ AI Video Kütüphanesi (Keyframes)")
                gallery = gr.Gallery(label="Analiz Edilen Kareler", columns=3, height="auto")
                
                index_btn.click(index_library, outputs=[gallery])

            with gr.Column(scale=1):
                video_output = gr.Video(label="Oluşturulan Video")
                reasoning_output = gr.Markdown(label="AI Analizi")
        
        generate_btn.click(ui_wrapper, inputs=[prompt_input], outputs=[video_output, reasoning_output, gallery])

    print("\n" + "="*50)
    print("🚀 VIDRUSH BAŞLATILIYOR...")
    print(f"📁 Video Klasörü: {os.path.abspath(engine.stock_folder)}")
    print(f"🎬 Bulunan Video Sayısı: {len(engine.get_stock_videos())}")
    print("="*50 + "\n")
    
    print("🚀 VIDRUSH AÇILIYOR: http://127.0.0.1:7860")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    asyncio.run(main())
