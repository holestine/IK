from openai import OpenAI
from audio import Audio
from key import openai_key
import threading
from extract_web_data import extract_webpage_data

class ChatBot:

    def __init__(self, mic_id=None, enable_speakers=False, threaded=False) -> None:
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
        with open('data.txt', 'r', encoding="utf-8") as f:
            data = ''.join(line for line in f)

        # Prompt to initialize LLM
        self.llm_prompt = {'role':'system', 'content':f""" \
                            You are a financial assistant for question-answering tasks. \
                            Use the following pieces of retrieved context to answer \
                            the question. If you don't know the answer, say that you \
                            don't know. Use three sentences maximum and keep the \
                            answer concise. \
                            If the question is not clear ask follow up questions \n\n \
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
        context.append({'role':'assistant', 'content':f"{response}"})

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
