"""SQLite-backed conversation history for the QueryVerify chat UI.

Conversations live in data/chat_history.db (separate from the main
analytics sample.db). Each message stores its exact session payload as
JSON so reloading a conversation reproduces the SQL block, result
dataframe and confidence pill exactly as they were shown.
"""

import datetime as _dt
import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    sql TEXT,
    result TEXT,
    confidence TEXT,
    confidence_score REAL,
    explanation TEXT,
    payload TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
"""

_lock = threading.Lock()

# Words that leave awkward dangling titles when used as a cut point.
_DANGLING = {
    "by", "of", "for", "to", "in", "at", "on", "with", "and", "or",
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
}

_TITLE_WORDS = 6


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def init_db():
    with _lock:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def make_title(question: str) -> str:
    """Short 5-6 word title from the first question (e.g. 'Natural phrasing
    request' style). Dangling prepositions at the cut point are dropped."""
    words = question.split()
    title_words = words[:_TITLE_WORDS]
    while title_words and title_words[-1].lower().strip("…").rstrip("?!.,:;") in _DANGLING:
        title_words.pop()
    title = " ".join(title_words)
    if len(words) > len(title_words):
        title += "…"
    return title.strip() or (question[:24] + ("…" if len(question) > 24 else ""))


def create_conversation(title: str) -> int:
    init_db()
    ts = _now()
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) "
                "VALUES (?, ?, ?)",
                (title, ts, ts),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def save_message(conv_id: int, msg: dict) -> None:
    init_db()
    payload = json.dumps(msg, ensure_ascii=False)
    kind = msg.get("kind", "")
    content = msg.get("content") if msg.get("content") is not None else ""
    data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
    sql = data.get("sql") if kind == "answer" else None
    result = json.dumps(data.get("result"), ensure_ascii=False) if kind == "answer" and data.get("result") else None
    confidence = data.get("confidence") if kind == "answer" else None
    confidence_score = data.get("confidence_score") if kind == "answer" else None
    explanation = data.get("explanation") if kind == "answer" else None
    ts = _now()
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, kind, content, sql, "
                "result, confidence, confidence_score, explanation, payload, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (conv_id, msg.get("role", "assistant"), kind, content, sql,
                 result, confidence, confidence_score, explanation, payload, ts),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, conv_id)
            )
            conn.commit()
        finally:
            conn.close()


def list_conversations() -> list[dict]:
    init_db()
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, id DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def _payload_to_message(row: sqlite3.Row) -> dict:
    try:
        return json.loads(row["payload"])
    except (TypeError, ValueError):
        pass
    kind = row["kind"]
    if kind == "answer":
        return {
            "role": row["role"],
            "kind": "answer",
            "data": {
                "sql": row["sql"],
                "result": json.loads(row["result"]) if row["result"] else None,
                "confidence": row["confidence"],
                "confidence_score": row["confidence_score"],
                "explanation": row["explanation"],
            },
        }
    return {"role": row["role"], "kind": kind, "content": row["content"]}


def load_messages(conv_id: int) -> list[dict]:
    init_db()
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conv_id,),
            ).fetchall()
            return [_payload_to_message(r) for r in rows]
        finally:
            conn.close()


def delete_conversation(conv_id: int) -> None:
    init_db()
    with _lock:
        conn = _connect()
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.commit()
        finally:
            conn.close()