from .base import BaseLLM
import requests

class PerplexityLLM(BaseLLM):
    def __init__(self, api_key, model="sonar"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.perplexity.ai/chat/completions"

    def invoke(self, prompt: str, system_prompt: str = "", response_format=None, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if response_format:
            payload["response_format"] = response_format
        resp = requests.post(self.url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]