from openai import OpenAI
from audio import Audio
from key import openai_key
import threading
import numpy as np
from transformers import pipeline
import torch

financial_prompt = "You are a financial assistant for question-answering tasks. Use the following pieces of \
    retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three \
    sentences maximum and keep the answer concise. If the question is not clear ask follow up questions \n\n"

class ChatBot:

    def __init__(self, mic_id=None, enable_speakers=False, threaded=False, prompt=financial_prompt, data_file='data.txt') -> None:
        '''
        Initialize the chatbot
        mic_id:          The index of the mic to enable
        enable_speakers: Wether or not audio will be played
        threaded:        Plays back audio in seperate thread, can interfere with speech detector
        '''
        
        # Get Open AI client
        self.__client = OpenAI(api_key=openai_key)

        # Whether or not to use speakers
        self.__enable_speakers = enable_speakers

        # Whether or not to thread playback
        self.__threaded = threaded

        # Initialize audio library
        self.audio = Audio()

        # Initialize mic
        if (mic_id is not None):
            self.initialize_microphone(mic_id)

        # Get the data for the LLM
        with open(data_file, 'r', encoding="utf-8") as f:
            data = ''.join(line for line in f)

        # Prompt to initialize LLM
        self.llm_prompt = {'role':'system', 'content':f""" \
                            "{prompt}"
                            "{data}"
                            """}
        
    def get_completion_from_messages(self, messages, model="gpt-4-turbo", temperature=0):
        '''
        Send the message to the specified OpenAI model
        '''
        response = self.__client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return response.choices[0].message.content
    
    def respond(self, prompt, history=[]):
        '''
        Get a response based on the current history
        '''
        context = [self.llm_prompt]
        for interaction in history:
            context.append({'role':'user', 'content':f"{interaction[0]}"})
            context.append({'role':'assistant', 'content':f"{interaction[1]}"})

        context.append({'role':'user', 'content':f"{prompt}"})
        response = self.get_completion_from_messages(context)

        if (self.__enable_speakers):
            # With threads
            if (self.__threaded):
                speaker_thread = threading.Thread(target=self.audio.communicate, args=(response,))
                speaker_thread.start()
            # Without threads
            else:
                self.audio.communicate(response)
       
        return response
    
    def initialize_microphone(self, mic_id):
        ''' 
        Initialize microphone object with the indicated ID. For best results a headset with a mic is recommended.
        '''
        self.audio.initialize_microphone(mic_id)

    def recognize_speech_from_mic(self):
        '''
        Listens for speech
        return: The text of the captured speech
        '''
        return self.audio.recognize_speech_from_mic()
    
    def communicate(self, message):
        '''
        Plays a message on the speakers
        message: the message
        '''
        self.audio.communicate(message)

    def get_prompt_from_audio(self, audio):
        '''
        Converts audio captured from gradio to text
        audio: object containing sampling frequency and raw audio data
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=device)

        sr, y = audio
        
        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)
            
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        prompt = transcriber({"sampling_rate": sr, "raw": y})["text"]  

        return prompt


if __name__ == "__main__":
    # Full Audio
    #chatbot = ChatBot(mic_id=2, enable_speakers=True)

    # No Audio
    chatbot = ChatBot(enable_speakers=False)

    history = []
    human_prompt = ""
    while human_prompt != 'goodbye':
        response = chatbot.respond(human_prompt, history)
        history.append([human_prompt, response])
        human_prompt = input(f"\n{response}\n\n")
