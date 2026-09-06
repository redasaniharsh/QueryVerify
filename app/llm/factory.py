"""
This function is the single switch point. Every agent calls
get_llm_client("sql") or get_llm_client("reasoning") and gets back whichever
client is configured — it never knows or cares if that's local or cloud.
"""

from app.llm.base import LLMClient
from app.llm.ollama_client import OllamaClient
from app.llm.cloud_client import CloudClient
from app.config import settings


def get_llm_client(role: str = "sql") -> LLMClient:
    """
    role="sql"        -> the SQL-generation model (code-specialized)
    role="reasoning"   -> the verification / ambiguity / explanation model
                          (lighter, general-purpose)
    """
    if settings.llm_provider == "ollama":
        model = settings.ollama_sql_model if role == "sql" else settings.ollama_reasoning_model
        return OllamaClient(model=model)

    # Cloud path — only reached if LLM_PROVIDER is changed away from "ollama"
    return CloudClient(model=settings.cloud_sql_model)
