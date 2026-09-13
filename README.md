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
   -> Sandboxed Execution      (runs the query safely, read-only, with a row cap)
   -> Semantic Verifier        (checks result vs. original question intent)
        -> if mismatch: loop back to SQL Generation (self-repair, retry budget)
        -> if match: Confidence Report (score + plain-English explanation)
```

## Tech stack

- Backend: Python, FastAPI
- LLM: Ollama (local) — see `app/config.py` for model names
- Database: **SQL Server** (`mssql+pyodbc`, ODBC Driver 17). SQLite is used only
  for internal per-session CSV uploads and local chat history — never for the
  sample dataset.
- Frontend: Streamlit
- LLM provider is swappable: local (Ollama) now, cloud-ready later — see
  `app/llm/` for the abstraction that makes that swap a config change, not a
  code change.

## Setup

Prerequisites:

1. [Ollama](https://ollama.com) installed, with the models pulled:
   ```
   ollama pull qwen2.5-coder:7b-instruct
   ollama pull qwen2.5:3b-instruct
   ```
2. **SQL Server** reachable locally with the ODBC Driver 17 installed:
   - SQL Server Express (or Developer/Standard) running as a named instance
     `SQLEXPRESS` with **Windows authentication** (the default), OR any SQL
     Server reachable by host + port with a SQL login.
   - Install [Microsoft ODBC Driver 17 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
     on the same machine.

Then:

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and confirm the `DATABASE_URL` matches your
   SQL Server. The default assumes local SQL Server Express with Windows auth:
   ```
   DATABASE_URL=mssql+pyodbc://.\SQLEXPRESS/QueryVerifyTest?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes
   ```
   SQL auth against a fixed TCP port (required for Docker, see below):
   ```
   DATABASE_URL=mssql+pyodbc://sa:<password>@<host>:1433/QueryVerifyTest?driver=ODBC+Driver+17+for+SQL+Server
   ```
3. Seed the sample database (creates `QueryVerifyTest` + `QueryVerifyTitanic`
   with the analytics schema and data):
   ```
   python scripts/setup_db_sqlserver.py
   ```
4. Run the backend:
   ```
   uvicorn app.main:app --reload
   ```
5. Run the frontend (separate terminal):
   ```
   streamlit run frontend/streamlit_app.py
   ```

Point the browser at http://localhost:8501 and ask away.

## Run with Docker

The backend and frontend run as containers; Ollama and SQL Server both stay
**native on the host** (device/GPU passthrough to Docker on Windows is
unreliable and adds complexity you don't need here).

Prerequisites:

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
   and running (WSL2 backend on Windows).
2. Ollama running natively on the host, with both models pulled:
   ```
   ollama pull qwen2.5-coder:7b-instruct
   ollama pull qwen2.5:3b-instruct
   ```
3. SQL Server running natively on the host, configured so the container can
   reach it over the network:
   - **TCP/IP enabled** for SQL Server (SQL Server Configuration Manager →
     Protocols for `SQLEXPRESS` → TCP/IP → Enabled) on port **1433** (a fixed
     port, not the named-instance dynamic port).
   - A SQL login (e.g. `sa`) with its password. Windows `trusted_connection`
     auth does **not** work from a Linux container, so the container must use
     SQL auth. Set the password via the `MSSQL_SA_PASSWORD` variable (see below).

Then:

```
$env:MSSQL_SA_PASSWORD="YourPassword"; docker compose up --build
```

and open http://localhost:8501.

How it wires together:

- `backend` (FastAPI, port 8000) reaches Ollama (port 11434) and SQL Server
  (port 1433) on your Windows host through `host.docker.internal`.
  `extra_hosts: ["host.docker.internal:host-gateway"]` is included defensively
  (Windows/macOS resolve `host.docker.internal` already; Linux needs the
  explicit mapping).
- `frontend` (Streamlit, port 8501) calls the backend over the internal Docker
  network via `QV_API_URL=http://backend:8000/ask`.
- `./data` is bind-mounted into **both** containers so per-session upload SQLite
  files and `chat_history.db` (conversation history) persist across restarts —
  nothing is wiped when a container stops. The sample dataset lives in SQL
  Server, not in `./data`.
- The backend is also published on host port `8000:8000` so you can hit the API
  directly (e.g. `curl http://localhost:8000/docs`) for debugging.

To stop: `docker compose down` (data stays on disk in `./data`).

## Automated CI

Automated CI covers deterministic safety guards (destructive-SQL blocking,
injection protection) plus backend-agnostic UI behavior on every push. Full
end-to-end testing (SQL generation, verification, self-repair, SQL Server
properties) requires a local Ollama + SQL Server and is run manually — see
`scripts/test_sql_properties_mssql.py`, `scripts/titanic_case_tests.py`, and
`scripts/edge_case_tests.py`.

## Project status

Research/design complete (OJT Review-1). Build in progress — see `docs/` (create
this folder for your build notes/logbook entries as you go) for weekly progress.

## Swapping to a cloud LLM later

Set `LLM_PROVIDER=openai` (or `anthropic`) and the matching API key in `.env`.
No application code changes needed — see `app/llm/base.py`.