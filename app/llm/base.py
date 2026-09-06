"""
Abstract LLM client. Every agent (sql_generator, verifier, ambiguity_checker,
etc.) talks to THIS interface, never to Ollama or an API directly.

That's what makes "local now, cloud later" a one-line config change instead of
a rewrite: add a new class here that implements generate(), and get_llm_client()
in factory.py returns it based on settings.llm_provider.
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        """Return the model's text completion for the given prompt."""
        raise NotImplementedError
