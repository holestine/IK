import os
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
import numpy as np
from transformers import pipeline
import torch

# If having trouble with ffmpeg, setting these may help
#AudioSegment.converter = "C:\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
#AudioSegment.ffmpeg    = "C:\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
#AudioSegment.ffprobe   = "C:\\ffmpeg\\ffmpeg\\bin\\ffprobe.exe"

class Audio:

    def __init__(self) -> None:
        """ Initialize speech recognition object
        """
        self.recognizer = sr.Recognizer()

        # Disable mic by default
        self.microphone = None

    def initialize_microphone(self, device_index):
        """ Initialize microphone object with appropriate device

        device_index: int indicating the index of the microphone
        """
        self.microphone = sr.Microphone(device_index)

    def communicate(self, phrase='You forgot to pass the text'):
        """ Audio approach that saves to a file and then plays it. Could be sped up by doing a sentence at a time.

        phrase: the string to convert to speech
        """

        try: 
            temp_file = 'temp.mp3'
            gTTS(phrase).save(temp_file)
            speech = AudioSegment.from_mp3(temp_file)
            play(speech)
            os.remove(temp_file)
        except BaseException as error:
            # Option without temporary mp3 but it's more robotic
            print(error)
            engine = pyttsx3.init()
            engine.say(phrase)
            engine.runAndWait()

    def recognize_speech_from_mic(self):
        """ Transcribes speech from a microphone

        Returns a dictionary with the following keys:
            "success":       A boolean indicating whether or not the request was successful
            "error":         'None' if successful, otherwise a string containing an error message
            "transcription": A string containing the transcribed text or 'None' if speech was unrecognizable
        """

        # Adjust the recognizer sensitivity for ambient noise and listen to the microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            speech = self.recognizer.listen(source)

        # Initialize response object
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }

        # Try to recognize the speech and handle exceptions accordingly
        try:
            response["transcription"] = self.recognizer.recognize_google(speech)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # Speech was unintelligible
            response["success"] = False
            response["error"] = "Unable to recognize speech"

        return response
    
    def get_prompt_from_gradio_audio(self, speech):
        '''
        Converts audio captured from gradio to text. See https://www.gradio.app/guides/real-time-speech-recognition for more info.
        audio: object containing sampling frequency and raw audio data
        
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=device)

        rate, y = speech
        
        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)
            
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        prompt = transcriber({"sampling_rate": rate, "raw": y})["text"]

        return prompt

    def get_prompt_from_file(self, file):

        speech = sr.AudioFile(file)
        with speech as source:
            speech = self.recognizer.record(source)
        text = self.recognizer.recognize_google(speech)
        return text
        
if __name__ == "__main__":
    audio = Audio()
    
    for i, mic in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{i}: {mic}")

    audio.initialize_microphone(1)

    audio.communicate("Hello")

    print(audio.recognize_speech_from_mic())

