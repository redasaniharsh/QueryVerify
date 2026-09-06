# QueryVerify

[![CI](https://github.com/redasaniharsh/QueryVerify/actions/workflows/ci.yml/badge.svg)](https://github.com/redasaniharsh/QueryVerify/actions/workflows/ci.yml)

A self-verifying Natural-Language-to-SQL analyst agent.

Takes a plain-English question → generates SQL with a local LLM → runs it safely →
checks whether the result actually answers the question → repairs itself if not →
returns an answer with a confidence score and plain-English explanation.

## Why this exists

Most NL-to-SQL tools stop at "did the query run." QueryVerify checks whether the
result actually answers what was asked, and fixes itself when it doesn't.

## Architecture

```
User question
   -> Orchestrator (FastAPI)
   -> Ambiguity Check          (asks for clarification if the question is vague)
   -> Schema Introspection     (reads real table/column names from the DB)
   -> SQL Generation           (local LLM via Ollama)
   -> Sandboxed Execution      (runs the query safely, read-only)
   -> Semantic Verifier        (checks result vs. original question intent)
        -> if mismatch: loop back to SQL Generation (self-repair, retry budget)
        -> if match: Confidence Report (score + plain-English explanation)
```

## Tech stack

- Backend: Python, FastAPI
- LLM: Ollama (local) — see `app/config.py` for model names
- Database: SQLite (dev) / PostgreSQL (target)
- Frontend: Streamlit
- LLM provider is swappable: local (Ollama) now, cloud-ready later — see
  `app/llm/` for the abstraction that makes that swap a config change, not a
  code change.

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull qwen2.5-coder:7b
   ollama pull llama3.2:3b
   ```
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and adjust if needed.
4. Seed the sample database:
   ```
   python scripts/setup_db.py
   ```
5. Run the backend:
   ```
   uvicorn app.main:app --reload
   ```
6. Run the frontend (separate terminal):
   ```
   streamlit run frontend/streamlit_app.py
   ```

## Run with Docker

Both the backend and frontend run as containers; Ollama stays **native on the
host** (GPU/device passthrough to Docker on Windows is unreliable and adds
complexity you don't need here).

Prerequisites:

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
   and running (WSL2 backend on Windows).
2. Ollama running natively on the host, with both models pulled:
   ```
   ollama pull qwen2.5-coder:7b-instruct
   ollama pull qwen2.5:3b-instruct
   ```

Then:

```
docker compose up --build
```

and open http://localhost:8501.

How it wires together:

- `backend` (FastAPI, port 8000) uses `OLLAMA_HOST=http://host.docker.internal:11434`
  to reach Ollama on your Windows host. `extra_hosts: ["host.docker.internal:host-gateway"]`
  is included defensively (Windows/macOS resolve `host.docker.internal` already;
  Linux needs the explicit mapping).
- `frontend` (Streamlit, port 8501) calls the backend over the internal Docker
  network via `QV_API_URL=http://backend:8000/ask`.
- `./data` is bind-mounted into **both** containers, so `sample.db` (analytics DB)
  and `chat_history.db` (conversation history) persist across restarts — nothing
  is wiped when a container stops.
- The backend is also published on host port `8000:8000` so you can hit the API
  directly (e.g. `curl http://localhost:8000/docs`) for debugging.

To stop: `docker compose down` (data stays on disk in `./data`).

## Automated CI

Automated CI covers deterministic safety guards (destructive-SQL blocking,
injection protection) on every push. Full end-to-end testing (SQL generation,
verification, self-repair) requires a local Ollama instance and is run manually
— see `scripts/edge_case_tests.py`.

## Project status

Research/design complete (OJT Review-1). Build in progress — see `docs/` (create
this folder for your build notes/logbook entries as you go) for weekly progress.

## Swapping to a cloud LLM later

Set `LLM_PROVIDER=openai` (or `anthropic`) and the matching API key in `.env`.
No application code changes needed — see `app/llm/base.py`.
