from audio import Audio
from chatbot import ChatBot
import gradio as gr

# Initialize the chatbot
chatbot = ChatBot(2)


with gr.Blocks() as demo:
    gr.ChatInterface(chatbot.respond)
    

demo.launch()
