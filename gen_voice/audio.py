import os
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3

class Audio:

    def __init__(self) -> None:
        # Initialize speech recognition object
        self.recognizer = sr.Recognizer()
        self.mic_enabled = False

    def initialize_microphone(self, device_index):
        # Initialize microphone object with appropriate device
        self.microphone = sr.Microphone(device_index)
        self.mic_enabled = True

    def communicate(self, phrase):
        temp_file = 'temp.mp3'
        gTTS(phrase).save(temp_file)
        audio = AudioSegment.from_mp3(temp_file)
        play(audio)
        os.remove(temp_file)

        # Option without temporary mp3 but it's more robotic
        #engine = pyttsx3.init()
        #engine.say(phrase)
        #engine.runAndWait()

    def recognize_speech_from_mic(self):
        """ Transcribe speech from the microphone

        Returns a dictionary with three keys:
        "success": a boolean indicating whether or not the API request was successful
        "error": `None` if no error occured, otherwise a string containing an error message if the API could not be reached or speech was unrecognizable
        "transcription": A string containing the transcribed text or `None` if speech was unrecognizable
        """

        # Adjust the recognizer sensitivity for ambient noise and listen to the microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        # Initialize response object
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }

        # Try to recognize the speech if a RequestError or UnknownValueError exception is caught update the response object accordingly
        try:
            response["transcription"] = self.recognizer.recognize_google(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["success"] = False
            response["error"] = "Unable to recognize speech"

        return response

if __name__ == "__main__":
    audio = Audio()
    
    sr.Microphone.list_microphone_names()

    audio.initialize_microphone(2)

    audio.communicate("Hello")

    print(audio.recognize_speech_from_mic())

