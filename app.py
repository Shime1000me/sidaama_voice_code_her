import spaces  # keep first on ZeroGPU
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
import binascii
from transformers import AutoProcessor, AutoModelForCTC, VitsModel, AutoTokenizer

SPACE_URL = "https://hayyoai1000me-sidaama-voice-code-her.hf.space"

print("Loading models...")
device = "cpu"  # no GPU is requested, so calls use no ZeroGPU quota

asr_model_name = "badrex/Ethio-ASR-sidaama"
asr_processor = AutoProcessor.from_pretrained(asr_model_name)
asr_model = AutoModelForCTC.from_pretrained(asr_model_name).to(device)

tts_model_name = "facebook/mms-tts-sid"
tts_model = VitsModel.from_pretrained(tts_model_name).to(device)
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
print("✅ Models loaded successfully!")

# ZeroGPU stops a Space at startup if it finds no @spaces.GPU function.
# This placeholder satisfies that check. Nothing calls it.
@spaces.GPU
def _zerogpu_placeholder():
    pass

def core_transcribe(audio_path):
    safe_wav_path = f"/tmp/{uuid.uuid4().hex}.wav"
    try:
        subprocess.run(["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", safe_wav_path, "-y"], check=True, capture_output=True)
        speech, _ = librosa.load(safe_wav_path, sr=16000)
        inputs = asr_processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad(): logits = asr_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = asr_processor.batch_decode(predicted_ids)[0]
        if isinstance(decoded, list): decoded = decoded[0]
        return decoded.strip()
    finally:
        if os.path.exists(safe_wav_path): os.remove(safe_wav_path)

def transcribe(audio_path):
    if audio_path is None: return "Please record or upload audio."
    try:
        text = core_transcribe(audio_path)
        if os.path.exists(audio_path): os.remove(audio_path)
        return text
    except Exception as e: return f"Error: {e}"

def speak(text):
    if not text or not text.strip(): return None
    try:
        inputs = tts_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad(): output = tts_model(**inputs).waveform
        filepath = f"/tmp/tts_{uuid.uuid4().hex}.wav"
        scipy.io.wavfile.write(filepath, rate=tts_model.config.sampling_rate, data=output.cpu().numpy().T)
        
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        
        # Convert to HEX instead of Base64
        hex_audio = binascii.hexlify(audio_bytes).decode('utf-8')
        
        if os.path.exists(filepath): os.remove(filepath)
        
        return hex_audio
    except Exception as e: 
        print(f"TTS Error: {e}")
        return None

def keep_alive():
    while True:
        time.sleep(600)
        try: requests.get(SPACE_URL)
        except: pass

threading.Thread(target=keep_alive, daemon=True).start()

with gr.Blocks(title="Hayyo Voice") as app:
    with gr.Tab("🎤"):
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", show_label=False)
        asr_button = gr.Button("🎤")
        text_output = gr.Textbox(show_label=False, lines=3)
        asr_button.click(fn=transcribe, inputs=audio_input, outputs=text_output)
    with gr.Tab("🔊"):
        text_input = gr.Textbox(show_label=False, lines=3)
        tts_button = gr.Button("🔊")
        audio_output = gr.Textbox(show_label=False, lines=3) 
        tts_button.click(fn=speak, inputs=text_input, outputs=audio_output)

app.launch()
