from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': req.prompt}])
    return {"response": response['message']['content']}
