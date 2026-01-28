import os
import json
import time
import subprocess
import datetime
import sys
import argparse
import re
import asyncio

from google import genai
from google.genai import types
from dotenv import load_dotenv
import edge_tts
import requests
import base64
from PIL import Image

# MoviePy v2 imports
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip, concatenate_videoclips
from moviepy.video.fx import Loop

# Load environment variables from .env (fallback)
load_dotenv()

class VidRusherEngine:
    def __init__(self, stock_folder=".", gemini_key=None, google_tts_key=None):
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
        
        # API Keys (from UI or environment)
        self.gemini_api_key = gemini_key or os.getenv("GEMINI_API_KEY", "")
        self.google_api_key = google_tts_key or os.getenv("GOOGLE_API_KEY", "")
        
        self.genai_client = None
        if self.gemini_api_key:
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
            print("Gemini API Client configured")
        
        if self.google_api_key:
            print("Google Cloud TTS configured")
        else:
            print("Using Edge-TTS (free fallback)")

    def _find_ffmpeg(self):
        """Checks if ffmpeg is in PATH or in local directory."""
        try:
            if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0:
                return "ffmpeg"
        except:
            pass
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
        """Safely updates Gradio progress bar."""
        if progress is None:
            return
        try:
            progress(value, desc=desc)
        except Exception:
            print(f"[Progress] {desc}")

    async def _extract_keyframes(self, progress=None):
        """Extracts keyframes for AI analysis."""
        stock_videos = self.get_stock_videos()
        indexed_content = []
        
        if not stock_videos:
            return []

        total = len(stock_videos)
        for i, v in enumerate(stock_videos):
            self._update_progress(progress, (i + 0.5) / total, f"Extracting: {v}")
            
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
        """Uses Gemini Vision to analyze and script."""
        if not self.genai_client:
            return [{"text": f"Demo: {prompt}", "video_file": self.get_stock_videos()[0] if self.get_stock_videos() else None, "reasoning": "No API key - using first video"}], []
            
        self._update_progress(progress, 0.1, "Scanning video library...")
        video_indexes = await self._extract_keyframes(progress)
        
        if not video_indexes:
            return [], []

        self._update_progress(progress, 0.3, "AI analyzing content...")
        
        contents = [f"You are a professional video editor. Analyze these video frames and create a script for: '{prompt}'"]
        for idx, item in enumerate(video_indexes):
            contents.append(f"CLIP {idx} (File: {item['filename']}):")
            # Convert PIL image to bytes for google-genai
            import io
            img_byte_arr = io.BytesIO()
            item['image'].save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            
        contents.append("""
        Create a video script. Use exact 'video_file' names. Add 'reasoning' for each selection.
        Return ONLY a JSON array of objects with exactly these keys: "text", "video_file", "reasoning".
        """)
        
        try:
            response = self.genai_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=contents
            )
            text = response.text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                content = text.strip()
            
            content = content.replace("```json", "").replace("```", "").strip()
            scenes = json.loads(content)
            return scenes, [item['path'] for item in video_indexes]
        except Exception as e:
            print(f"AI Error: {e}")
            fallback = [{"text": f"Generated content for: {prompt}", 
                         "video_file": video_indexes[0]['filename'], 
                         "reasoning": "AI response parsing failed or API error, using first video."}]
            return fallback, [item['path'] for item in video_indexes]

    async def generate_audio(self, scenes):
        """Generates voiceover using TTS."""
        for i, scene in enumerate(scenes):
            audio_path = os.path.join(self.temp_dir, f"scene_{i}.mp3")
            
            if self.google_api_key:
                print(f"Google TTS: Scene {i}")
                url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_api_key}"
                payload = {
                    "input": {"text": scene['text']},
                    "voice": {"languageCode": "en-US", "name": "en-US-Neural2-D"},
                    "audioConfig": {"audioEncoding": "MP3"}
                }
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    audio_content = response.json().get("audioContent")
                    with open(audio_path, "wb") as out:
                        out.write(base64.b64decode(audio_content))
                else:
                    print(f"TTS Error: {response.text}, falling back to Edge-TTS")
                    VOICE = "en-US-GuyNeural"
                    communicate = edge_tts.Communicate(scene['text'], VOICE)
                    await communicate.save(audio_path)
            else:
                VOICE = "en-US-GuyNeural"
                communicate = edge_tts.Communicate(scene['text'], VOICE)
                await communicate.save(audio_path)
            
            audio_clip = AudioFileClip(audio_path)
            scene['duration'] = audio_clip.duration
            scene['audio_path'] = audio_path
            audio_clip.close()
            
        return scenes

    def get_stock_videos(self):
        """Lists available mp4 files."""
        return [f for f in os.listdir(self.stock_folder) 
                if f.endswith(".mp4") and not f.startswith("vidrusher_")]

    def _get_proxy_video(self, filename, progress=None):
        """Creates low-res proxy for fast rendering."""
        original_path = os.path.join(self.stock_folder, filename)
        proxy_path = os.path.join(self.proxy_dir, f"proxy_{filename}")
        
        if not os.path.exists(proxy_path):
            self._update_progress(progress, None, f"Creating proxy: {filename}...")
            cmd = [
                self.ffmpeg_path, "-y", "-ss", "0", "-i", original_path,
                "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
                "-an", "-t", "60",
                proxy_path
            ]
            subprocess.run(cmd, capture_output=True)
        return proxy_path

    def assemble_video(self, scenes, progress=None):
        """Assembles final video with MoviePy v2."""
        final_clips = []
        total = len(scenes)
        
        TARGET_H = 720
        TARGET_W = 405
        
        for i, scene in enumerate(scenes):
            self._update_progress(progress, (i+1)/total, f"Assembling scene {i+1}/{total}...")
            
            video_file = scene.get('video_file')
            if not video_file:
                continue
                
            video_path = self._get_proxy_video(video_file, progress)
            audio_path = scene.get('audio_path')
            
            if not audio_path or not os.path.exists(audio_path):
                continue
            
            try:
                v_clip = VideoFileClip(video_path)
                a_clip = AudioFileClip(audio_path)
                
                v_clip = v_clip.resized(height=TARGET_H)
                w, h = v_clip.size
                
                if w > TARGET_W:
                    v_clip = v_clip.cropped(x_center=w/2, y_center=h/2, width=TARGET_W, height=TARGET_H)
                
                target_duration = a_clip.duration
                
                if v_clip.duration < target_duration:
                    v_clip = v_clip.with_effects([Loop(duration=target_duration)])
                else:
                    v_clip = v_clip.with_duration(target_duration)
                
                v_clip = v_clip.with_audio(a_clip)
                
                try:
                    wrapped_text = self._wrap_text(scene['text'], width=20)
                    txt_clip = TextClip(
                        text=wrapped_text,
                        font_size=24,
                        color='white',
                        bg_color='rgba(0,0,0,0.7)',
                        text_align="center",
                        size=(TARGET_W - 20, None)
                    ).with_duration(target_duration).with_position(('center', TARGET_H - 150))
                    
                    v_clip = CompositeVideoClip([v_clip, txt_clip])
                except Exception as e:
                    print(f"Subtitle error: {e}")
                
                final_clips.append(v_clip)
                
            except Exception as e:
                print(f"Error on scene {i}: {e}")
                continue

        if not final_clips:
            return None

        print("Rendering final video...")
        final_video = concatenate_videoclips(final_clips, method="compose")
        
        output_name = os.path.join(self.temp_dir, "vidrusher_output.mp4")
        final_video.write_videofile(
            output_name, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            threads=os.cpu_count() or 4,
            logger=None
        )
        
        for c in final_clips: 
            try:
                c.close()
            except:
                pass
        final_video.close()
        
        return output_name


# ==================== GRADIO UI ====================

def create_demo():
    import gradio as gr
    
    # Global engine instance (will be recreated with API keys)
    current_engine = [None]
    
    def initialize_engine(gemini_key, google_key, stock_path):
        """Initialize engine with user-provided API keys and path."""
        current_engine[0] = VidRusherEngine(
            stock_folder=stock_path if stock_path else ".",
            gemini_key=gemini_key if gemini_key else None,
            google_tts_key=google_key if google_key else None
        )
        video_count = len(current_engine[0].get_stock_videos())
        return f"Engine initialized! Found {video_count} videos in: {current_engine[0].stock_folder}"
    
    def process_video(prompt, gemini_key, google_key, stock_path, upload_files, progress=gr.Progress()):
        """Main video generation function."""
        if not prompt:
            return None, "Please enter a prompt!", []
        
        # Determine working folder: use uploads if provided, otherwise stock_path
        working_folder = stock_path if stock_path else "."
        
        if upload_files:
            # Create a unique temp folder for this upload session
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            upload_dir = os.path.join(".", "temp", f"upload_{timestamp}")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Copy uploaded files to the temp upload dir
            import shutil
            for f_obj in upload_files:
                shutil.copy(f_obj.name, os.path.join(upload_dir, os.path.basename(f_obj.name)))
            
            working_folder = upload_dir
            print(f"Using uploaded files in: {working_folder}")

        # Initialize engine with the chosen folder
        engine = VidRusherEngine(
            stock_folder=working_folder,
            gemini_key=gemini_key if gemini_key else None,
            google_tts_key=google_key if google_key else None
        )
        
        async def async_run():
            scenes, thumbs = await engine.generate_script(prompt, progress)
            
            if not scenes:
                return None, "No script generated.", thumbs
            
            engine._update_progress(progress, 0.5, "Generating audio...")
            scenes = await engine.generate_audio(scenes)
            
            engine._update_progress(progress, 0.7, "Assembling video...")
            output_video = engine.assemble_video(scenes, progress)
            
            if not output_video:
                return None, "Video assembly failed.", thumbs
            
            reasoning_md = "### AI Analysis & Reasoning\n\n"
            for i, scene in enumerate(scenes):
                reasoning_md += f"**Scene {i+1}:** `{scene.get('video_file')}`\n"
                reasoning_md += f"> *{scene.get('reasoning', 'Best match selected.')}*\n\n"
            
            return output_video, reasoning_md, thumbs
        
        return asyncio.run(async_run())
    
    def index_videos(gemini_key, stock_path, progress=gr.Progress()):
        """Index video library and show keyframes."""
        engine = VidRusherEngine(stock_folder=stock_path if stock_path else ".", gemini_key=gemini_key)
        
        async def async_index():
            engine._update_progress(progress, 0.1, "Scanning library...")
            indexed = await engine._extract_keyframes(progress)
            return [item['path'] for item in indexed]
        
        return asyncio.run(async_index())

    # Build Gradio Interface
    with gr.Blocks(
        title="VidRusher AI Video Engine",
        theme=gr.themes.Soft(),
        css="""
        .api-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 10px; padding: 20px; }
        .main-title { text-align: center; background: linear-gradient(90deg, #00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        """
    ) as demo:
        
        gr.Markdown("# VidRusher AI Video Engine", elem_classes="main-title")
        gr.Markdown("**Turn Your Ideas into Videos Automatically:** Type what you want to create. Our AI analyzes your footage, chooses the right scenes, adds a narrator's voice, and delivers a fully edited video in seconds.")
        
        with gr.Accordion("API Configuration (Optional)", open=True):
            gr.Markdown("### Security Note")
            gr.Markdown("*Your API keys are processed only in-memory for this session. They are never stored, logged, or shared. Our code is [Open Source](https://github.com/mustafabasar7/vidrusher).*")
            with gr.Row():
                gemini_input = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Get from: aistudio.google.com",
                    type="password"
                )
                google_tts_input = gr.Textbox(
                    label="Google Cloud TTS Key (Optional)",
                    placeholder="Falls back to Edge-TTS if empty",
                    type="password"
                )
                stock_path_input = gr.Textbox(
                    label="Video Library Path",
                    placeholder="Path to folder containing .mp4 files (Default: .)",
                    value="."
                )
            status_output = gr.Textbox(label="Status", interactive=False)
            init_btn = gr.Button("Initialize Engine", variant="secondary")
            init_btn.click(initialize_engine, inputs=[gemini_input, google_tts_input, stock_path_input], outputs=[status_output])
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Video Topic / Prompt",
                    placeholder="e.g., How to make the perfect espresso",
                    lines=3
                )
                with gr.Row():
                    index_btn = gr.Button("Scan Library", variant="secondary")
                    generate_btn = gr.Button("Generate Video", variant="primary")
                
                gr.Markdown("### Video Library (Keyframes)")
                gallery = gr.Gallery(label="Analyzed Frames", columns=3, height="auto")
                
                with gr.Tab("Upload Your Videos"):
                    video_upload = gr.File(
                        label="Upload mp4 clips",
                        file_count="multiple",
                        file_types=[".mp4"]
                    )
                    gr.Markdown("*If you upload videos here, the system will use them instead of the Library Path.*")
                
                index_btn.click(index_videos, inputs=[gemini_input, stock_path_input], outputs=[gallery])

            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video")
                reasoning_output = gr.Markdown(label="AI Analysis")
        
        generate_btn.click(
            process_video, 
            inputs=[prompt_input, gemini_input, google_tts_input, stock_path_input, video_upload], 
            outputs=[video_output, reasoning_output, gallery]
        )
        
        gr.Markdown("---")
        gr.Markdown("**Built by [Mustafa Başar](https://github.com/mustafabasar7)** | Gemini 2.0 Flash + MoviePy + Gradio")
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
