"""
Cloud LLM client — NOT used during local development.

Fill this in only when you're ready to deploy a hosted demo (e.g. before
pushing a public GitHub demo link). Implement generate() using the OpenAI or
Anthropic SDK, reading the API key from settings. Until then, LLM_PROVIDER
should stay "ollama" in .env and this file can stay a stub.
"""

from app.llm.base import LLMClient
from app.config import settings


class CloudClient(LLMClient):
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        # TODO (only when deploying): call OpenAI/Anthropic API here using
        # settings.openai_api_key or settings.anthropic_api_key.
        raise NotImplementedError(
            "Cloud LLM not wired up yet — set LLM_PROVIDER=ollama in .env for now."
        )
