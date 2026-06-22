"""A thin OpenAI-compatible chat client.

Works with any OpenAI-compatible endpoint, e.g. a local vLLM server
(`vllm serve <model>`) or the official OpenAI / DeepSeek API. Configure it
with environment variables or constructor arguments:

    KZIP_BASE_URL   default "http://localhost:8000/v1"
    KZIP_API_KEY    default "EMPTY"  (vLLM ignores it)
    KZIP_MODEL      default "Qwen3-4B"
"""

import os

from openai import OpenAI


class LLM:
    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or os.environ.get("KZIP_MODEL", "Qwen3-4B")
        base_url = base_url or os.environ.get("KZIP_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.environ.get("KZIP_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, prompt, system="You are a helpful AI assistant.", max_tokens=2048, temperature=0.0):
        """Run a single chat completion and return the generated text."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
