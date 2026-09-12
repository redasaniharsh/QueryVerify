"""
Central configuration.

This is the ONE place that decides local-vs-cloud LLM. Everything else in the
app asks this file "which client do I use" and never hardcodes a provider.

Now:      LLM_PROVIDER=ollama   -> fully local, nothing leaves the machine.
Later:    LLM_PROVIDER=openai   -> swap to a cloud model for a hosted demo,
          (or anthropic)           without touching agent/pipeline code.
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM provider switch
    llm_provider: str = "ollama"

    # Local (Ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_sql_model: str = "qwen2.5-coder:7b-instruct"
    ollama_reasoning_model: str = "qwen2.5:3b-instruct"

    # Cloud (used only if llm_provider != "ollama")
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cloud_sql_model: str = "gpt-4o-mini"

    # Database
    database_url: str = "sqlite:///./data/sample.db"

    # Agent behavior
    max_repair_attempts: int = 3
    self_consistency_samples: int = 3

    class Config:
        # QV_ENV_FILE lets a caller swap the config file without touching the
        # real .env (e.g. run the suite against a SQL Server database:
        #   $env:QV_ENV_FILE = ".env.mssql").
        env_file = os.environ.get("QV_ENV_FILE", ".env")


settings = Settings()

if settings.llm_provider != "ollama":
    raise RuntimeError(
        'This build is local-only. LLM_PROVIDER must be "ollama" — '
        "cloud providers are intentionally disabled until deployment."
    )
