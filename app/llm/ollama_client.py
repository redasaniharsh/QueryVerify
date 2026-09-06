"""
Local LLM client, talks to Ollama running on this machine.
Nothing here ever calls the internet.
"""

import ollama
from app.llm.base import LLMClient
from app.config import settings

OLLAMA_TIMEOUT_SECONDS = 180


class LLMTimeoutError(RuntimeError):
    """Raised when an Ollama call times out or otherwise cannot complete.

    Caught by the repair loop (sql generation), the verifier, and the ambiguity
    checker, and converted into error_feedback or a graceful response instead
    of crashing the pipeline.
    """


class OllamaClient(LLMClient):
    def __init__(self, model: str):
        self.model = model
        self.host = settings.ollama_host

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        client = ollama.Client(host=self.host, timeout=OLLAMA_TIMEOUT_SECONDS)
        try:
            response = client.chat(
                model=self.model,
                messages=(
                    ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}]
                ),
                options={"temperature": temperature},
            )
        except Exception as exc:
            raise LLMTimeoutError(
                f"Ollama call to '{self.model}' failed after "
                f"{OLLAMA_TIMEOUT_SECONDS}s timeout: {exc}"
            ) from exc
        return response["message"]["content"]
