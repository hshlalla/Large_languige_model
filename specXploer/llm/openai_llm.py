from .base import BaseLLM
from langchain_openai import ChatOpenAI

class OpenAILLM(BaseLLM):
    def __init__(self, api_key, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model, openai_api_key=api_key)

    def invoke(self, prompt: str, **kwargs):
        return self.llm.invoke(prompt)