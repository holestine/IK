import gradio as gr
from transformers import pipeline
import numpy as np
import torch
from audio import Audio


device = "cuda:0" if torch.cuda.is_available() else "cpu"
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=device)
a = Audio()

def transcribe(audio):
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    text = transcriber({"sampling_rate": sr, "raw": y})["text"]  
    a.communicate(text)

    return text


demo = gr.Interface(
    transcribe,
    gr.Audio(sources="microphone"),
    "text",
)

demo.launch()