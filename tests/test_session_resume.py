"""Reload/resume behavior: the app must keep the active conversation in the
URL (?conv=<id>) and restore it on the first run of a fresh session, so a real
browser hard-refresh (Ctrl+Shift+R — where session_state does NOT survive but
the address bar does) lands back on the same conversation instead of a blank
"New chat".

Simulated with Streamlit's AppTest: run 1 is a normal session that asks a
question (creating a conversation and writing ?conv=); run 2 seeds a NEW
AppTest with the same query_params, which is exactly what a reload produces.

Deliberately backend-agnostic: when the /ask backend is reachable the
conversation ends in an answer, and when it is not (e.g. plain `pytest tests`
in CI, where the uvicorn + Ollama stack is not running) it ends in an error
message — but either way the conversation is created (conv_id + ?conv=<id>),
persisted, and restored on reload, which is the behavior under test.
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "frontend"))

import chat_history  # noqa: E402

APP_PATH = str(_REPO_ROOT / "frontend" / "streamlit_app.py")

TEST_Q = "how many rows are in the dataset"


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point the app at a throwaway conversation table so the test never touches
    data/chat_history.db (monkeypatch keeps the shared chat_history module wired
    to the temp file for the whole test process)."""
    db = tmp_path / "chat_history.db"
    monkeypatch.setattr(chat_history, "DB_PATH", db)
    chat_history.init_db()
    return db


def _ask(at: AppTest, question: str) -> None:
    at.text_input(key="qv_text").set_value(question)
    senders = [b for b in at.button if b.label == "➤"]
    assert senders, "send button not found in the rendered app"
    senders[0].click()
    at.run()


def test_reload_resumes_last_conversation(isolated_db):
    # Run 1: a normal session. Fresh boot -> no conversation, no conv param.
    at1 = AppTest.from_file(APP_PATH, default_timeout=180)
    at1.run()
    assert not at1.exception
    assert at1.session_state["conv_id"] is None
    assert "conv" not in at1.query_params

    # Ask a question -> /ask answer lands, conversation created, URL synced.
    _ask(at1, TEST_Q)

    assert not at1.exception
    conv_id = at1.session_state["conv_id"]
    assert conv_id is not None
    assert at1.query_params.get("conv") == [str(conv_id)]
    assert len(at1.session_state["messages"]) == 2
    # CI runs without the /ask backend: the terminal message is an "answer"
    # when the backend is up, or an "error" when it is unreachable. Either way
    # the conversation was created, persisted, and URL-synced — which is what
    # the reload semantics depend on.
    assert at1.session_state["messages"][-1]["kind"] in ("answer", "error")

    # Run 2: a brand-new AppTest seeded with ?conv=<id> == simulated reload.
    at2 = AppTest.from_file(APP_PATH, default_timeout=180)
    at2.query_params = {"conv": str(conv_id)}
    at2.run()

    assert not at2.exception
    assert at2.session_state["conv_id"] == conv_id
    assert len(at2.session_state["messages"]) == len(at1.session_state["messages"])
    told = [(m["kind"], m.get("content")) for m in at1.session_state["messages"]]
    reloaded = [(m["kind"], m.get("content")) for m in at2.session_state["messages"]]
    assert reloaded == told
    assert at2.query_params.get("conv") == [str(conv_id)]


def test_stale_conversation_link_falls_back_to_clean_chat(isolated_db):
    # A deleted/missing conversation id in the URL must not error: the app
    # drops the stale parameter and boots into a clean new chat.
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.query_params = {"conv": "999999"}
    at.run()

    assert not at.exception
    assert at.session_state["conv_id"] is None
    assert at.session_state["messages"] == []
    assert "conv" not in at.query_params