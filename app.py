import gradio as gr
import torch
import librosa
import scipy.io.wavfile
import os
import uuid
import threading
import requests
import time
import subprocess
import shutil
import spaces
from transformers import AutoProcessor, AutoModelForCTC, VitsModel, AutoTokenizer

# ==========================================
# 1. CONFIGURATION
# ==========================================
SPACE_URL = "https://hayyoai1000me-sidaama-voice-code-her.hf.space"

# ==========================================
# 2. LOAD MODELS (FORCED CPU)
# ==========================================
print("Loading models...")
device = "cpu"

asr_model_name = "badrex/Ethio-ASR-sidaama"
asr_processor = AutoProcessor.from_pretrained(asr_model_name)
asr_model = AutoModelForCTC.from_pretrained(asr_model_name).to(device)

tts_model_name = "facebook/mms-tts-sid"
tts_model = VitsModel.from_pretrained(tts_model_name).to(device)
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)

print("✅ Models loaded successfully!")

# ==========================================
# 3. AI FUNCTIONS
# ==========================================
def core_transcribe(audio_path):
    safe_wav_path = f"/tmp/{uuid.uuid4().hex}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", safe_wav_path, "-y"],
            check=True, capture_output=True
        )
        speech, _ = librosa.load(safe_wav_path, sr=16000)
        inputs = asr_processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad(): logits = asr_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return asr_processor.batch_decode(predicted_ids)[0]
    finally:
        if os.path.exists(safe_wav_path): os.remove(safe_wav_path)

@spaces.GPU
def transcribe(audio_path):
    if audio_path is None: return "Please record or upload audio."
    try:
        text = core_transcribe(audio_path)
        if os.path.exists(audio_path): os.remove(audio_path)
        return text
    except Exception as e: return f"Error: {e}"

@spaces.GPU
def speak(text):
    if not text or not text.strip(): return None
    try:
        inputs = tts_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad(): output = tts_model(**inputs).waveform
        filepath = f"/tmp/tts_{uuid.uuid4().hex}.wav"
        scipy.io.wavfile.write(filepath, rate=tts_model.config.sampling_rate, data=output.cpu().numpy().T)
        return filepath
    except Exception as e: return None

# ==========================================
# 4. THE CAPTIONING PIPELINE
# ==========================================
def caption_video(input_video):
    if input_video is None: return None, None, "Please upload a video."
    job_id = uuid.uuid4().hex
    audio_path = f"/tmp/{job_id}.wav"
    srt_path = f"/tmp/{job_id}.srt"
    output_video = f"/tmp/{job_id}_captioned.mp4"
    final_video = f"/tmp/{job_id}_final.mp4"
    final_srt = f"/tmp/{job_id}_final.srt"

    try:
        subprocess.run(["ffmpeg", "-i", input_video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"], check=True, capture_output=True)
        probe_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_video}"'
        duration = float(subprocess.check_output(probe_cmd, shell=True).decode('utf-8').strip())

        transcription = core_transcribe(audio_path)

        words = transcription.split()
        word_count = len(words)
        target_chunks = max(1, int(duration / 3))
        chunk_size = max(1, word_count // target_chunks)
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, word_count, chunk_size)]
        time_per_chunk = duration / len(chunks)

        srt_content = ""
        for i, chunk in enumerate(chunks):
            start = i * time_per_chunk
            end = (i + 1) * time_per_chunk
            def format_time(seconds):
                hrs = int(seconds // 3600); mins = int((seconds % 3600) // 60); secs = int(seconds % 60); ms = int((seconds % 1) * 1000)
                return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"
            srt_content += f"{i+1}\n{format_time(start)} --> {format_time(end)}\n{chunk}\n\n"

        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)

        subprocess.run([
            "ffmpeg", "-i", input_video,
            "-vf", f"subtitles={srt_path}:force_style='FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=20'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", output_video, "-y"
        ], check=True, capture_output=True)

        shutil.copy(output_video, final_video)
        shutil.copy(srt_path, final_srt)
        return final_video, final_srt, "✅ Captions generated successfully!"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"
    finally:
        for f in [audio_path, srt_path, output_video]:
            if os.path.exists(f): os.remove(f)

# ==========================================
# 5. KEEP-ALIVE
# ==========================================
def keep_alive():
    while True:
        time.sleep(600)
        try: requests.get(SPACE_URL)
        except: pass

threading.Thread(target=keep_alive, daemon=True).start()

# ==========================================
# 6. GRADIO UI — ICONS ONLY
# ==========================================
with gr.Blocks(title="Hayyo Voice") as app:
    with gr.Tab("🎤"):
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", show_label=False)
        asr_button = gr.Button("🎤")
        text_output = gr.Textbox(show_label=False, lines=3)
        asr_button.click(fn=transcribe, inputs=audio_input, outputs=text_output)
    with gr.Tab("🔊"):
        text_input = gr.Textbox(show_label=False, lines=3)
        tts_button = gr.Button("🔊")
        audio_output = gr.Audio(show_label=False)
        tts_button.click(fn=speak, inputs=text_input, outputs=audio_output)

app.launch(allowed_paths=["/tmp"])
