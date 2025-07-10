import gradio as gr
import requests

API_URL = "http://app:8000/chat"

def chatbot(prompt):
    res = requests.post(API_URL, json={"prompt": prompt})
    return res.json()["response"]

iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="Internal Company Chatbot",
    description="Ask your question to the chatbot powered by Ollama.",
)

iface.launch(server_name="0.0.0.0", server_port=7860)
