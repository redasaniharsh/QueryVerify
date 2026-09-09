"""
Streamlit frontend for QueryVerify — chat-style conversation with
clarification follow-up (no retyping) and a dynamic status indicator
while the backend is working. Supports typed or spoken (audio) questions;
audio is transcribed fully locally with Whisper (faster-whisper), the
transcript becomes the user message, and goes through the exact same
/ask pipeline as a typed question.

Visual layer: the page chrome (hero, chat bubbles, input bar, confidence
badge, thinking animation) is styled with a single CSS block injected
via st.markdown(unsafe_allow_html=True) on top of the dark theme defined
in .streamlit/config.toml. No application logic lives in the CSS.
"""

import hashlib
import io
import logging
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import faster_whisper
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger("queryverify.frontend")

_FRONTEND_DIR = Path(__file__).resolve().parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))
import chat_history  # local module: SQLite persistence for conversations

# Backend endpoint. Overridable (QV_API_URL) so the containerized frontend can
# reach the backend service by Docker DNS name (http://backend:8000/ask).
API_URL = os.environ.get("QV_API_URL", "http://127.0.0.1:8000/ask")

# Bring-your-own-data: uploaded CSVs are loaded into a NEW, separate SQLite file
# per browser session (data/user_upload_<session_id>.db). The sample DB is never
# touched. Files are temporary: replaced on a new upload and swept once they are
# older than UPLOAD_TTL_SECONDS (the browser tab is the only real lifecycle).
_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOAD_MAX_BYTES = int(os.environ.get("QV_UPLOAD_MAX_MB", "50")) * 1024 * 1024
UPLOAD_TTL_SECONDS = int(os.environ.get("QV_UPLOAD_TTL_HOURS", "2")) * 3600

# Whisper model size for local transcription (small/base are fast enough for
# short clips on CPU; "small" transcribes Hindi far better than "base").
WHISPER_MODEL = os.environ.get("QV_WHISPER_MODEL", "small")

# Languages this project's pipeline understands. Anything else Whisper "hears"
# on a question clip is a mis-detection (or out-of-scope speech) and must not
# be shipped into /ask as garbage.
SUPPORTED_SPEECH_LANGS = {"en", "hi"}
MIN_LANG_PROB = float(os.environ.get("QV_WHISPER_MIN_LANG_PROB", "0.5"))
MIN_AVG_LOGPROB = float(os.environ.get("QV_WHISPER_MIN_AVG_LOGPROB", "-0.65"))
MAX_NO_SPEECH_PROB = float(os.environ.get("QV_WHISPER_MAX_NO_SPEECH_PROB", "0.8"))
# Optional hard language hint ("en"/"hi"); auto-detect + English-fallback logic
# is skipped entirely when this is set.
WHISPER_LANG_HINT = os.environ.get("QV_WHISPER_LANG") or None
WHISPER_DEBUG = os.environ.get("QV_WHISPER_DEBUG", "0") == "1"

STATUS_MESSAGES = [
    "Reading your database schema...",
    "Generating SQL...",
    "Double-checking the answer...",
    "Cross-referencing multiple attempts...",
]

# --------------------------------------------------------------------------
# Visual assets (pure CSS layer — no behaviour).
# --------------------------------------------------------------------------

# Seamless WhatsApp-style doodle tile: data/SQL/analytics glyphs (database
# cylinder, magnifier, chart bars, check-in-circle), drawn dark-on-dark at very
# low opacity so it reads as a faint wallpaper behind the chat area.
_PATTERN_TILE = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140' "
    "viewBox='0 0 140 140'>"
    "<g fill='none' stroke='rgba(255,255,255,0.05)' stroke-width='2' "
    "stroke-linecap='round' stroke-linejoin='round'>"
    # database cylinder
    "<ellipse cx='30' cy='24' rx='17' ry='7'/>"
    "<path d='M13 24 v34 a17 7 0 0 0 34 0 V24'/>"
    "<path d='M13 44 a17 7 0 0 0 34 0'/>"
    # magnifying glass
    "<circle cx='94' cy='22' r='13'/>"
    "<path d='M104 30 L116 42'/>"
    # chart bars + baseline
    "<path d='M24 90 v16 M34 90 v10 M44 90 v22'/>"
    "<path d='M16 120 h40'/>"
    # check inside a circle
    "<circle cx='92' cy='92' r='16'/>"
    "<path d='M85 92 l5 5 l9 -10'/>"
    "</g></svg>"
)
_PATTERN_URL = "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(
    _PATTERN_TILE
)

# QueryVerify logo mark: rounded accent tile, a small database cylinder and a
# verification check (used as the assistant avatar and hero mark).
_LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40' "
    "viewBox='0 0 40 40'>"
    "<defs><linearGradient id='qv' x1='0' y1='0' x2='1' y2='1'>"
    "<stop offset='0' stop-color='#4f8cff'/>"
    "<stop offset='1' stop-color='#2456d6'/>"
    "</linearGradient></defs>"
    "<rect width='40' height='40' rx='11' fill='url(#qv)'/>"
    "<ellipse cx='20' cy='14' rx='7' ry='3' fill='#fff'/>"
    "<path d='M13 14 v10 a7 3 0 0 0 14 0 V14' stroke='#fff' fill='none'/>"
    "<path d='M13 19 a7 3 0 0 0 14 0' stroke='#fff' fill='none'/>"
    "<path d='M15 27 l4 4 l7 -8' stroke='#fff' stroke-width='3' "
    "stroke-linecap='round' stroke-linejoin='round' fill='none'/>"
    "</svg>"
)
_LOGO_URL = "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(_LOGO_SVG)

_CSS = """
<style>
/* ---------- theme tokens ---------- */
.stApp{
  --qv-accent:#3b82f6;
  --qv-accent-2:#60a5fa;
  --qv-bg:#0a0f1c;
  --qv-bubble:#1a2438;
  --qv-border:rgba(255,255,255,.07);
  --qv-text:#e8edf7;
  --qv-muted:#93a1b8;
}

/* ---------- WhatsApp-style doodle wallpaper ---------- */
.stApp{
  background-color:var(--qv-bg);
  background-image:url("__PATTERN__");
  background-size:140px 140px;
  background-attachment:fixed;
}
.stApp [data-testid="stHeader"]{background:transparent; z-index:5;}
#MainMenu, [data-testid="stToolbar"]{display:none;}

/* ---------- hero ---------- */
.hero{display:flex; align-items:center; gap:14px; padding:22px 4px 4px;}
.hero-mark{width:46px; height:46px; border-radius:13px;
  box-shadow:0 8px 26px rgba(59,130,246,.35);}
.hero-title{font-size:27px; font-weight:800; letter-spacing:-.5px; line-height:1.1;}
.hero-tag{font-size:14px; color:var(--qv-muted); margin-top:2px;}
.hero-divider{height:1px; margin:16px 0 4px;
  background:linear-gradient(90deg,transparent,rgba(96,165,250,.4),transparent);}

/* ---------- chat bubbles ---------- */
.stChatMessage{display:flex; gap:10px; align-items:flex-start; margin:12px 0;}
.stChatMessage > div:first-child{
  width:38px; height:38px; border-radius:11px; flex:0 0 38px;
  display:flex; align-items:center; justify-content:center;
  background:rgba(96,165,250,.10); font-size:20px; line-height:1;
}
.stChatMessage > div:first-child img{
  width:38px; height:38px; border-radius:11px;
  box-shadow:0 3px 10px rgba(0,0,0,.35);
}
.stChatMessage [data-testid="stChatMessageContent"]{
  border-radius:16px; padding:11px 14px;
  box-shadow:0 3px 12px rgba(0,0,0,.25);
  line-height:1.5; min-width:0;
}
.stChatMessage:has([aria-label*="Chat message from user"]){flex-direction:row-reverse;}
.stChatMessage:has([aria-label*="Chat message from user"]) [data-testid="stChatMessageContent"]{
  flex:0 1 auto; max-width:74%; color:#fff;
  background:linear-gradient(135deg,#2563eb,#1d4ed8);
  border-bottom-right-radius:5px;
}
.stChatMessage:has([aria-label*="Chat message from assistant"]) [data-testid="stChatMessageContent"]{
  flex:1 1 auto; max-width:94%; color:var(--qv-text);
  background:var(--qv-bubble);
  border-bottom-left-radius:5px;
}
.stChatMessage [data-testid="stChatMessageContent"] p{margin:0 0 8px;}
.stChatMessage [data-testid="stChatMessageContent"] p:last-child{margin-bottom:0;}
.stChatMessage [data-testid="stChatMessageContent"] .stCaption,
.stChatMessage [data-testid="stChatMessageContent"] .stCaption p{
  color:var(--qv-muted); font-size:13px;
}

/* ---------- smooth message entrance (newest bubble only) ---------- */
.stChatMessage:not(:has(~ .stChatMessage)) [data-testid="stChatMessageContent"]{
  animation:qvIn .38s ease-out both;
}
@keyframes qvIn{
  from{opacity:0; transform:translateY(9px) scale(.985);}
  to{opacity:1; transform:none;}
}

/* ---------- thinking indicator (pulsing dots) ---------- */
.think{display:inline-flex; align-items:center; gap:11px;
  padding:10px 15px; border-radius:16px; border-bottom-left-radius:5px;
  background:var(--qv-bubble); border:1px solid rgba(59,130,246,.35);}
.think .dots{display:inline-flex; gap:5px;}
.think .dot{width:7px; height:7px; border-radius:50%; background:var(--qv-accent-2);
  animation:blink 1.2s infinite;}
.think .dot:nth-child(2){animation-delay:.2s;}
.think .dot:nth-child(3){animation-delay:.4s;}
@keyframes blink{0%,60%,100%{opacity:.25; transform:translateY(0);}
  30%{opacity:1; transform:translateY(-2px);}}
.think .think-msg{color:var(--qv-muted); font-size:13px;}

/* ---------- confidence badge pill ---------- */
.conf-pill{display:inline-flex; align-items:center; gap:7px; margin:8px 0 4px;
  padding:4px 12px; border-radius:999px; font-size:13px; font-weight:600;}
.conf-ico{width:17px; height:17px; border-radius:50%; flex:0 0 17px;
  display:inline-flex; align-items:center; justify-content:center; font-size:11px;}
.conf-pill.high{background:rgba(34,197,94,.14); color:#4ade80;
  border:1px solid rgba(34,197,94,.35);}
.conf-pill.high .conf-ico{background:#22c55e; color:#04170b;}
.conf-pill.medium{background:rgba(250,204,21,.12); color:#facc15;
  border:1px solid rgba(250,204,21,.35);}
.conf-pill.medium .conf-ico{background:#facc15; color:#171105;}
.conf-pill.low{background:rgba(248,113,113,.13); color:#f87171;
  border:1px solid rgba(248,113,113,.4);}
.conf-pill.low .conf-ico{background:#ef4444; color:#1c0505;}

/* ---------- chat input: one pill bar (text + mic + send) ---------- */
[data-testid="stMainBlockContainer"]{padding-bottom:130px;}
[data-testid="stForm"]{
  position:fixed; bottom:18px; left:50%; transform:translateX(-50%);
  width:min(740px, 93vw); height:auto; min-height:0; flex:0 0 auto;
  z-index:30;
  padding:8px 10px 8px 6px; border-radius:26px;
  background:rgba(10,15,28,.86); backdrop-filter:blur(10px);
  border:1px solid var(--qv-border);
  box-shadow:0 12px 40px rgba(0,0,0,.5);
  align-self:end;
}
[data-testid="stForm"] [data-testid="stVerticalBlock"]{gap:0; height:auto; min-height:0;}
[data-testid="stForm"] [data-testid="stHorizontalBlock"]{gap:8px; align-items:center;}
[data-testid="stForm"] [data-testid="stColumn"]{height:auto; min-height:0;}
[data-testid="stForm"] [data-testid="stTextInput"]{
  background:transparent; border:none; box-shadow:none;}
[data-testid="stForm"] [data-baseweb="input"],
[data-testid="stForm"] [data-testid="stTextInput"] > div{
  background:transparent; border:none; box-shadow:none !important;}
[data-testid="stForm"] [data-baseweb="input"] input,
[data-testid="stForm"] input{color:var(--qv-text);}
/* send button hugs the pill's right edge, round + accent */
[data-testid="stForm"] [data-testid="stElementContainer"]{width:100%;}
[data-testid="stForm"] [data-testid="stFormSubmitButton"]{
  display:flex; justify-content:flex-end; align-items:center; height:44px; width:100%;}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button{
  width:38px !important; height:38px !important; min-height:38px !important;
  max-height:38px !important; border-radius:50% !important; padding:0 !important;
  box-sizing:border-box; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#3b82f6,#2563eb); color:#fff;
  border:none; font-size:16px; line-height:1;}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover{
  filter:brightness(1.12);}
/* mic icon docked inside the pill, just left of the send button. Both circles
   are exactly 38px, centered in a 38px fixed box that shares the pill's
   vertical centerline (bottom:30px -> same sweep as the 62px pill at bottom:18px). */
[data-testid="stAudioInput"]{
  position:fixed; bottom:30px; z-index:31;
  right:calc(50vw - min(370px, 46.5vw) + 56px);
  width:38px; height:38px; padding:0; margin:0;
  display:flex; align-items:center; justify-content:center; box-sizing:border-box;}
[data-testid="stAudioInput"] > div{
  width:38px; height:38px; padding:0 !important; margin:0;
  display:flex; align-items:center; justify-content:center; overflow:visible;}
[data-testid="stAudioInput"] [data-testid="stElementToolbar"]{display:none !important;}
[data-testid="stAudioInput"] button{
  width:38px !important; height:38px !important; min-height:38px !important;
  max-height:38px !important; border-radius:50% !important; padding:0 !important;
  margin:0 !important; box-sizing:border-box; overflow:hidden;
  display:flex; align-items:center; justify-content:center;
  background:#1b2740 !important; border:1px solid rgba(255,255,255,.06) !important;
  color:#9db4e0 !important; line-height:1;}
[data-testid="stAudioInput"] button:hover{
  background:#233153 !important; color:#fff !important;}
[data-testid="stAudioInput"] svg{
  width:17px; height:17px; flex:0 0 17px; display:block;}
[data-testid="stAudioInput"] [data-testid="stWidgetLabel"]{display:none;}
[data-testid="stAudioInput"] > div > div:nth-child(3),
[data-testid="stAudioInput"] [data-testid="stAudioInputWaveformTimeCode"]{display:none !important;}

/* ---------- misc polish ---------- */
[data-testid="stSidebar"]{
  background:rgba(10,15,28,.6); border-right:1px solid rgba(255,255,255,.06);}
.sb-brand{font-size:17px; font-weight:800; letter-spacing:-.3px;
  color:var(--qv-text); margin:6px 2px 12px;}
.sb-hdr{font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:.09em; color:var(--qv-muted); margin:16px 4px 6px;}
.sb-emptystate{font-size:12px; color:var(--qv-muted); margin:2px 4px 8px;}
.sb-feedback{font-size:12px; color:var(--qv-muted); margin:2px 4px 8px;}
.sb-feedback b{color:var(--qv-text); font-weight:800;}
.feedback-thanks{font-size:12px; color:var(--qv-muted); margin:4px 2px 2px;}
.feedback-thumbs [data-testid="stButton"] button{padding:2px 10px; font-size:14px;}
[data-testid="stSidebar"] button{
  background:rgba(96,165,250,.06); color:var(--qv-accent-2);
  border:1px solid rgba(96,165,250,.35); border-radius:10px;
  justify-content:center;}
[data-testid="stSidebar"] button[kind="secondary"]{
  background:transparent; color:#cdd7ea;
  border:1px solid rgba(255,255,255,.08);
  justify-content:flex-start; font-weight:500; font-size:14px;
  padding:6px 10px; text-align:left;}
[data-testid="stSidebar"] button[kind="secondary"]:hover{
  background:rgba(96,165,250,.08); color:#fff;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{line-height:1.5;}
[data-testid="stCodeBlock"]{border-radius:12px;
  border:1px solid var(--qv-border); background:#0d1424;}
.stDataFrame{background:rgba(255,255,255,.02); border-radius:12px;
  margin:6px 0 2px;}
.stAlert{border-radius:12px;}
/* Keep the sidebar expanded at all viewport widths. Streamlit 1.54
   auto-collapses the sidebar (translateX(-100%) + 1px) when the window is
   approx <=770px wide (docked DevTools / split screen). Collapse is applied
   by swapping an emotion class on stSidebar, so an !important override on the
   stable data-testid selector beats it and survives re-renders. */
[data-testid="stSidebar"][data-testid="stSidebar"][data-testid="stSidebar"]{
  width:300px !important; max-width:300px !important; min-width:300px !important;
  transform:none !important; visibility:visible !important;
  margin-right:0 !important;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"][data-testid="stSidebarContent"][data-testid="stSidebarContent"]{
  width:100% !important; max-width:100% !important; min-width:100% !important;}
/* At the widths where Streamlit would auto-collapse, keep the fixed input pill
   and mic inside the area right of the never-collapsed 300px sidebar. */
@media (max-width: 770px){
  [data-testid="stForm"]{
    width:min(440px, calc(100vw - 336px));
    transform:translateX(calc(-50% + 150px));}
}
</style>
""".replace("__PATTERN__", _PATTERN_URL)

st.set_page_config(page_title="QueryVerify", layout="centered")

st.markdown(_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero">
      <img class="hero-mark" src="{_LOGO_URL}" alt="QueryVerify logo"/>
      <div>
        <div class="hero-title">QueryVerify</div>
        <div class="hero-tag">Ask your database anything — verified before you trust it</div>
      </div>
    </div>
    <div class="hero-divider"></div>
    """,
    unsafe_allow_html=True,
)

chat_history.init_db()

if "conv_id" not in st.session_state:
    st.session_state.conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qv_session_id" not in st.session_state:
    st.session_state["qv_session_id"] = secrets.token_hex(4)
if "qv_db_mode" not in st.session_state:
    st.session_state["qv_db_mode"] = "Sample Data"
if "qv_db_path" not in st.session_state:
    st.session_state["qv_db_path"] = None
if "qv_db_schema" not in st.session_state:
    st.session_state["qv_db_schema"] = None
if "qv_upload_sig" not in st.session_state:
    st.session_state["qv_upload_sig"] = None


def _new_chat():
    st.session_state.conv_id = None
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)
    st.session_state.pop("qv_audio_processed", None)


def _open_conversation(conv_id):
    st.session_state.conv_id = conv_id
    st.session_state.messages = chat_history.load_messages(conv_id)
    pending = None
    if st.session_state.messages:
        last = st.session_state.messages[-1]
        pending = last.get("pending_question")
    if pending:
        st.session_state["pending_question"] = pending
    else:
        st.session_state.pop("pending_question", None)
    st.session_state.pop("qv_audio_processed", None)


def _persist_message(msg):
    """Store a message row, creating the conversation from its first question
    if this chat has no id yet (auto-titled from the question)."""
    cid = st.session_state.get("conv_id")
    if cid is None:
        title = chat_history.make_title(
            (msg.get("content") or msg.get("pending_question") or "Conversation")
        )
        cid = chat_history.create_conversation(title)
        st.session_state.conv_id = cid
    msg["message_id"] = chat_history.save_message(cid, msg)


class UserUploadError(Exception):
    """Raised when an uploaded file fails validation or cannot be loaded."""


def _sanitize_table_name(name: str) -> str:
    """Filename stem -> a safe SQLite table name (lowercase, [a-z0-9_])."""
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9_]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "data"
    if stem[0].isdigit():
        stem = "t_" + stem
    return stem


def _sweep_uploads(max_age_s: float, keep: Path | None = None) -> None:
    """Delete session-only upload databases older than max_age_s. Never touches
    the file currently in use (keep) and never the sample DB."""
    if not _UPLOAD_DIR.exists():
        return
    now = time.time()
    keep_resolved = Path(keep).resolve() if keep else None
    for f in _UPLOAD_DIR.glob("user_upload_*.db"):
        if keep_resolved and f.resolve() == keep_resolved:
            continue
        try:
            if now - f.stat().st_mtime > max_age_s:
                f.unlink()
        except OSError:
            pass  # file in use / already gone; harmless


def _build_uploaded_db(files) -> tuple[Path, list]:
    """Validate + load uploaded CSVs into this session's private SQLite file.

    Returns (db_path, schema) where schema is [(table, [cols], row_count)].
    On any error raises UserUploadError; the previous upload (if any) is kept
    untouched until the new set is fully built.
    """
    if not files:
        raise UserUploadError("No files selected.")

    loaded = []  # (table_name, DataFrame) — parse everything BEFORE writing
    for uploaded_file in files:
        fname = uploaded_file.name
        if not fname.lower().endswith(".csv"):
            raise UserUploadError(f"{fname}: only CSV files are supported.")
        if (uploaded_file.size or 0) > UPLOAD_MAX_BYTES:
            raise UserUploadError(
                f"{fname}: file is too large "
                f"(max {UPLOAD_MAX_BYTES // (1024 * 1024)} MB)."
            )
        try:
            df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        except Exception as exc:
            raise UserUploadError(f"Could not parse {fname}: {exc}")
        if df.shape[1] == 0:
            raise UserUploadError(f"{fname}: the CSV has no columns.")
        loaded.append((_sanitize_table_name(fname), df))

    db_path = _UPLOAD_DIR / f"user_upload_{st.session_state['qv_session_id']}.db"
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # replace this session's previous build

    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        for table_name, df in loaded:
            df.to_sql(table_name, conn, index=False, if_exists="replace")
        conn.commit()
    except sqlite3.Error as exc:
        raise UserUploadError(f"Could not write uploaded data to SQLite: {exc}")
    finally:
        conn.close()

    schema = [(table_name, list(df.columns), len(df)) for table_name, df in loaded]
    _sweep_uploads(UPLOAD_TTL_SECONDS, keep=db_path)
    return db_path, schema


def _render_schema_block(schema) -> None:
    """Compact 'what can I ask about' list after a successful upload."""
    lines = ['<div class="sb-emptystate"><b>Uploaded schema:</b></div>']
    for table_name, cols, n_rows in schema:
        cols_txt = ", ".join(cols) if cols else "(no columns)"
        lines.append(
            f'<div class="sb-emptystate">• <code>{table_name}</code> '
            f"({cols_txt}) &middot; {n_rows} rows</div>"
        )
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def _active_database() -> str | None:
    """Which database the current session's questions target (None = sample)."""
    if (
        st.session_state.get("qv_db_mode") == "My Uploaded Data"
        and st.session_state.get("qv_db_path")
    ):
        return Path(st.session_state["qv_db_path"]).name
    return None


if not st.session_state.get("_qv_swept_once"):
    _sweep_uploads(UPLOAD_TTL_SECONDS)
    st.session_state["_qv_swept_once"] = True


with st.sidebar:
    st.markdown('<div class="sb-brand">QueryVerify</div>', unsafe_allow_html=True)
    if st.button("＋ New chat", key="qv_new_chat", use_container_width=True):
        _new_chat()
        st.rerun()

    st.markdown('<div class="sb-hdr">Recents</div>', unsafe_allow_html=True)
    conversations = chat_history.list_conversations()
    if conversations:
        for conv in conversations:
            if st.button(
                conv["title"],
                key=f"qv_conv_{conv['id']}",
                use_container_width=True,
                type="secondary",
            ):
                _open_conversation(conv["id"])
                st.rerun()
    else:
        st.markdown(
            '<div class="sb-emptystate">No conversations yet.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get("conv_id") is not None or st.session_state.messages:
        if st.button(
            "Delete this conversation",
            key="qv_clear",
            use_container_width=True,
            type="secondary",
        ):
            cid = st.session_state.get("conv_id")
            if cid is not None:
                chat_history.delete_conversation(cid)
            _new_chat()
            st.rerun()

    _feedback_stats = chat_history.feedback_stats()
    if _feedback_stats["total"] > 0:
        _helpful_pct = round(100 * _feedback_stats["ups"] / _feedback_stats["total"])
        st.markdown("---")
        st.markdown(
            f'<div class="sb-feedback"><b>{_helpful_pct}%</b> of rated answers '
            "marked helpful</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div class="sb-hdr">Use your own data</div>',
        unsafe_allow_html=True,
    )
    st.file_uploader(
        "Upload CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="qv_csv_upload",
        help=(
            "Load your own CSV data to ask questions about it. One SQLite table "
            "is created per file, named after the file. The sample dataset is "
            "never modified."
        ),
    )
    _uploaded_files = st.session_state.get("qv_csv_upload")
    if _uploaded_files:
        _sig = [(f.name, f.size) for f in _uploaded_files]
        if _sig != st.session_state["qv_upload_sig"]:
            try:
                _db_path, _schema = _build_uploaded_db(_uploaded_files)
                st.session_state["qv_db_path"] = str(_db_path)
                st.session_state["qv_db_schema"] = _schema
                st.session_state["qv_upload_sig"] = _sig
                st.session_state["qv_db_mode"] = "My Uploaded Data"
                st.success(
                    f"Loaded {len(_schema)} table(s) from "
                    f"{len(_uploaded_files)} file(s)."
                )
            except UserUploadError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Failed to load uploaded data: {exc}")

    _have_upload = st.session_state.get("qv_db_path") is not None
    st.radio(
        "Data source",
        ["Sample Data", "My Uploaded Data"],
        key="qv_db_mode",
        disabled=not _have_upload,
        help=(
            "Which database questions are routed to. Sample Data is the fixed "
            "demo dataset; My Uploaded Data is this session's temporary file."
        ),
    )
    if _have_upload:
        st.markdown(
            '<div class="sb-emptystate">Session-only: the file lives in '
            "data/ and is deleted when replaced by a new upload or once it "
            "goes stale after the tab closes.</div>",
            unsafe_allow_html=True,
        )
        _render_schema_block(st.session_state["qv_db_schema"])


@st.cache_resource
def _load_whisper():
    """Load the local Whisper model once per session (cached via st.cache_resource).
    Runs fully offline: no cloud transcription API is ever called."""
    return faster_whisper.WhisperModel(
        WHISPER_MODEL, device="cpu", compute_type="int8"
    )


class UnclearAudioError(Exception):
    """Raised when Whisper's words are not trustworthy enough to send to /ask
    (bad mic, loud noise, non-en/hi speech, or a confident-but-hostile
    language). The UI turns this into a friendly retry prompt."""


def _run_whisper(model, audio_bytes, language):
    """One transcription pass. Segments are materialized so we can report the
    model's own confidence signals (avg_logprob / no_speech_prob) alongside
    the text. condition_on_previous_text=False suits short one-shot clips."""
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=language,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segs = list(segments)
    text = "".join(s.text for s in segs).strip()
    stats = {
        "avg_logprob": (sum(s.avg_logprob for s in segs) / len(segs)) if segs else 0.0,
        "max_no_speech_prob": max((s.no_speech_prob for s in segs), default=1.0),
    }
    return text, info, stats


def _transcribe(audio_bytes: bytes, language: str | None = None) -> tuple[str, dict]:
    """Transcribe recorded audio locally; returns (text, meta).

    language: explicit "en"/"hi" hint, or None for auto-detect. When
    auto-detecting, a language outside {en, hi} (Whisper frequently "hears"
    German/Japanese on short, slightly noisy English clips) or a very low
    language confidence triggers ONE retry with language="en" — the expected
    case for this project — instead of passing a mistranslated transcript into
    the pipeline. QV_WHISPER_LANG force-hints and skips the fallback.

    Raises UnclearAudioError when the result is still not confident enough,
    so garbage text never reaches /ask.
    """
    model = _load_whisper()
    hint = language or WHISPER_LANG_HINT

    text, info, stats = _run_whisper(model, audio_bytes, hint)
    retried_en = False
    misfire = None
    if hint is None and (
        info.language not in SUPPORTED_SPEECH_LANGS
        or info.language_probability < MIN_LANG_PROB
    ):
        misfire = (info.language, info.language_probability)
        text, info, stats = _run_whisper(model, audio_bytes, "en")
        retried_en = True

    meta = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "top_languages": [
            (lang, round(prob, 3))
            for lang, prob in (info.all_language_probs or [])[:3]
        ],
        "avg_logprob": round(stats["avg_logprob"], 3),
        "max_no_speech_prob": round(stats["max_no_speech_prob"], 3),
        "duration": round(info.duration, 2),
        "hint_used": hint,
        "retried_en": retried_en,
        **(
            {"misfire_language": misfire[0], "misfire_probability": round(misfire[1], 3)}
            if misfire
            else {}
        ),
    }
    logger.info("Whisper transcription meta=%s", meta)

    if not text:
        raise UnclearAudioError("no speech detected")
    weak = (
        info.language_probability < MIN_LANG_PROB
        or stats["avg_logprob"] < MIN_AVG_LOGPROB
        or stats["max_no_speech_prob"] > MAX_NO_SPEECH_PROB
    )
    if weak:
        raise UnclearAudioError(
            f"language={info.language} ({info.language_probability:.0%}), "
            f"avg_logprob={stats['avg_logprob']:.2f}, "
            f"no_speech={stats['max_no_speech_prob']:.0%}"
        )
    return text, meta


def _call_backend(question):
    """Run requests.post in a background thread while the main thread cycles
    the dynamic status text. Streamlit reruns the whole script per interaction,
    so all widget calls stay on the main thread; the status lives in ONE
    st.empty() placeholder, written only here and always cleared before the
    response is processed (finally), so no status text can outlive the answer.

    Note: Streamlit binds a st.empty() placeholder to its position in the
    script, and repeated .markdown() on the same instance REPLACES (probe:
    app/agents + AppTest) — the loop must never open a second placeholder for
    the same status, and must stop the instant the request completes.
    """
    holder = st.empty()
    done = threading.Event()
    bucket = {}
    database = _active_database()

    def _post():
        try:
            payload = {"question": question}
            if database:
                payload["database"] = database
            resp = requests.post(API_URL, json=payload, timeout=300)
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")
            bucket["resp"] = resp
        except Exception as exc:  # network/down server, or a 4xx with detail
            bucket["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_post, daemon=True)
    thread.start()
    i = 0
    try:
        while not done.is_set():
            holder.markdown(
                f'<div class="think"><span class="dots">'
                f'<span class="dot"></span><span class="dot"></span>'
                f'<span class="dot"></span></span>'
                f'<span class="think-msg">{STATUS_MESSAGES[i % len(STATUS_MESSAGES)]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            i += 1
            time.sleep(0.5)
    finally:
        holder.empty()  # clear the moment the request completes (or is interrupted)

    if "exc" in bucket:
        raise bucket["exc"]
    resp = bucket["resp"]
    resp.raise_for_status()
    return resp.json()


_CONF_ICONS = {"high": "✓", "medium": "⚠", "low": "✕"}
_USER_AVATAR = "🧑"
_ASSISTANT_AVATAR = _LOGO_URL

_TRACE_ICONS = {
    "guard": "⛔",
    "absent_entity": "🔍",
    "schema_read": "📖",
    "ambiguity": "❓",
    "generation": "✏️",
    "execution": "▶️",
    "verification": "✅",
    "consistency": "🔄",
    "final": "🎯",
}


def _confidence_pill(label, score):
    icon = _CONF_ICONS.get(label, "✕")
    st.markdown(
        f'<span class="conf-pill {label}"><span class="conf-ico">{icon}</span>'
        f"{score:.0%} confidence</span>",
        unsafe_allow_html=True,
    )


def _render_trace(data):
    """Collapsed expandable timeline of what the pipeline did for this answer."""
    trace = data.get("trace") or []
    if not trace:
        return
    with st.expander("Show how this was answered", expanded=False):
        for entry in trace:
            step = entry.get("step", "")
            detail = entry.get("detail", "")
            dur = entry.get("duration_s", 0)
            icon = _TRACE_ICONS.get(step, "•")
            if step == "verification" and "Rejected" in detail:
                icon = "❌"
            dur_txt = f" · {dur:.1f}s" if dur >= 0.05 else ""
            st.markdown(f"{icon} {detail}{dur_txt}")


# --------------------------------------------------------------------------
# Automatic result visualization. Only renders when the result table has a
# clean, chartable shape; a chart is never forced onto data it doesn't fit.
# --------------------------------------------------------------------------

_YEAR_RE = re.compile(r"^\d{4}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")


def _coerce_number(value):
    """float(value) for int/float/numeric-string values, else None (bool off)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace(" ", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _all_numbers(rows, col):
    values = [r.get(col) for r in rows]
    non_empty = [v for v in values if v is not None and v != ""]
    if not non_empty:
        return False
    return all(_coerce_number(v) is not None for v in non_empty)


def _all_dates(rows, col):
    values = [r.get(col) for r in rows]
    non_empty = [v for v in values if v is not None and v != ""]
    if not non_empty:
        return False
    return all(
        isinstance(v, str)
        and (_YEAR_RE.fullmatch(v.strip()) or _DATE_RE.fullmatch(v.strip()))
        for v in non_empty
    )


def _chart_plan(rows):
    """Decide whether/how to visualize a result table.

    Returns ("bar"|"line", x_col, y_col) or None when the table does not
    warrant a chart:
      * fewer than 2 rows        -> None (one number is not a chart)
      * anything but 2 columns   -> None (don't force a shape onto it)
      * date + number, 2+ rows   -> ("line", date_col, number_col)
      * category + number, 2+ rows -> ("bar", category_col, number_col)
      * number + number (or any other pairing) -> None
    """
    if not rows or len(rows) < 2:
        return None
    columns = list(rows[0].keys())
    if len(columns) != 2:
        return None

    def _is_value_col(col):
        # A date-like column is an axis (trend), never a chartable value.
        return _all_numbers(rows, col) and not _all_dates(rows, col)

    value_cols = [c for c in columns if _is_value_col(c)]
    if len(value_cols) != 1:
        return None
    y = value_cols[0]
    x = columns[1] if columns[0] == y else columns[0]

    if _all_dates(rows, x):
        return "line", x, y
    if not _all_numbers(rows, x):
        return "bar", x, y
    return None


def _render_chart(data):
    """Render a chart for a successful answer, when the result shape fits.

    Placed between the results dataframe and the confidence pill. The
    'Visualized:' label makes clear this is auto-generated from the same
    rows shown above, not a separate artifact.
    """
    rows = (data.get("result") or {}).get("rows") or []
    plan = _chart_plan(rows)
    if plan is None:
        return
    kind, x_col, y_col = plan
    df = pd.DataFrame(rows)
    if kind == "line":
        df = df.copy()
        df[x_col] = pd.to_datetime(df[x_col])
    st.caption("_Visualized:_")
    if kind == "bar":
        st.bar_chart(df, x=x_col, y=y_col)
    else:
        st.line_chart(df, x=x_col, y=y_col)


def _render_answer(data):
    translated = data.get("translated_question")
    if translated:
        st.caption(f'_Understood as: "{translated}"_')

    if data.get("blocked"):
        st.error(data.get("explanation", "Blocked by the read-only guard."))
        _render_trace(data)
        return
    if not data.get("success"):
        st.error(data.get("explanation", "Could not answer this question."))
        _render_trace(data)
        return

    sql = data.get("sql")
    if sql:
        st.markdown("**Generated SQL**")
        st.code(sql, language="sql")

    st.markdown("**Results**")
    rows = (data.get("result") or {}).get("rows") or []
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.info("No rows returned.")

    _render_chart(data)

    label = data.get("confidence", "low")
    score = data.get("confidence_score", 0.0)
    _confidence_pill(label, score)

    explanation = data.get("explanation") or ""
    if explanation:
        st.caption(explanation)

    _render_trace(data)


def _render_feedback(msg):
    """Small 👍/👎 rating below a successful answer. Encourages a single vote:
    once a rating is recorded (session or DB), the buttons are replaced by a
    one-line confirmation and cannot be clicked again."""
    msg_id = msg.get("message_id")
    if not msg_id:
        return
    if msg.get("feedback"):
        st.markdown(
            '<div class="feedback-thanks">Thanks for the feedback</div>',
            unsafe_allow_html=True,
        )
        return
    col_up, col_down = st.columns([1, 1])
    with col_up:
        if st.button("👍", key=f"qv_fb_{msg_id}_up", help="Helpful"):
            chat_history.set_feedback(msg_id, "up")
            msg["feedback"] = "up"
            st.rerun()
    with col_down:
        if st.button("👎", key=f"qv_fb_{msg_id}_down", help="Not helpful"):
            chat_history.set_feedback(msg_id, "down")
            msg["feedback"] = "down"
            st.rerun()


def _render_message(msg):
    role = "assistant" if msg["role"] == "assistant" else "user"
    avatar = _ASSISTANT_AVATAR if role == "assistant" else _USER_AVATAR
    with st.chat_message(role, avatar=avatar):
        if msg["kind"] == "user":
            st.markdown(msg["content"])
        elif msg["kind"] == "flag":
            st.markdown(msg["content"])
            st.caption("_You can answer the follow-up below — no need to retype the original question._")
            _render_trace(msg.get("data") or {})
        elif msg["kind"] == "answer":
            _render_answer(msg["data"])
            if (msg.get("data") or {}).get("success") and msg.get("message_id"):
                _render_feedback(msg)
        else:
            st.error(msg["content"])


for msg in st.session_state.messages:
    _render_message(msg)

# --- Chat input: one continuous pill bar (text + mic + send). The form owns
# the text field + send button; the audio recorder is a separate widget OUTSIDE
# the form (so a finished recording reruns the script and is processed on its
# own, exactly as before) that CSS docks into the right side of the same pill.
with st.form("qv_input", clear_on_submit=True, border=False):
    col_text, col_send = st.columns(
        [0.8, 0.2], vertical_alignment="center", gap="small"
    )
    with col_text:
        prompt = st.text_input(
            "Your question",
            placeholder="Ask a question about your data...",
            key="qv_text",
            label_visibility="collapsed",
        )
    with col_send:
        st.form_submit_button("➤", use_container_width=False)

audio_file = st.audio_input(
    "Record a voice question", key="qv_audio", label_visibility="collapsed"
)

if prompt:
    pass  # typed question; nothing extra to do
elif audio_file is not None:
    # Transcribe the recorded audio once (fingerprinted on content so reruns
    # of the same recording never re-transcribe), then feed the transcript
    # into the exact same pipeline as a typed question.
    audio_hash = hashlib.sha256(audio_file.getvalue()).hexdigest()
    if st.session_state.get("qv_audio_processed") != audio_hash:
        with st.spinner("Transcribing audio..."):
            try:
                transcript, meta = _transcribe(audio_file.getvalue())
            except UnclearAudioError as exc:
                logger.info("Whisper rejected clip: %s", exc)
                if "no speech detected" in str(exc):
                    st.error("No speech detected in the recording. Please try again.")
                else:
                    st.error(
                        "Could not clearly understand the audio — please try again. "
                        f"({exc})"
                    )
                st.stop()
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                st.stop()
        if not transcript:
            st.error("No speech detected in the recording. Please try again.")
            st.stop()
        st.session_state["qv_audio_processed"] = audio_hash
        st.session_state["qv_audio_meta"] = meta
        if WHISPER_DEBUG or meta.get("retried_en"):
            if meta.get("retried_en"):
                st.caption(
                    f"_Auto-detect heard `{meta['misfire_language']}` "
                    f"({meta['misfire_probability']:.0%}) — re-transcribed as English._"
                )
            else:
                st.caption(f"_Voice meta: {meta}_")
        prompt = transcript

if prompt:
    pending = st.session_state.get("pending_question")
    if pending:
        full_question = f"{pending} — clarification: {prompt}"
        display = f"_(answering the follow-up)_ {prompt}"
    else:
        full_question = prompt
        display = prompt

    user_msg = {"role": "user", "kind": "user", "content": display}
    st.session_state.messages.append(user_msg)
    _persist_message(user_msg)
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(display)

    st.session_state.messages.append({"role": "assistant", "kind": "spinner"})
    msg_index = len(st.session_state.messages) - 1
    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        try:
            data = _call_backend(full_question)
        except Exception as exc:
            err_msg = {
                "role": "assistant",
                "kind": "error",
                "content": f"Failed to reach the backend: {exc}",
            }
            st.session_state.messages[msg_index] = err_msg
            _persist_message(err_msg)
            st.error(f"Failed to reach the backend: {exc}")
            st.stop()

        if data.get("needs_clarification"):
            # Keep the ORIGINAL question so the next chat message is treated as
            # the answer to this clarification (combined, sent to /ask).
            st.session_state["pending_question"] = full_question
            flag_msg = {
                "role": "assistant",
                "kind": "flag",
                "content": data["clarifying_question"],
                "pending_question": full_question,
                "data": data,
            }
            st.session_state.messages[msg_index] = flag_msg
            _persist_message(flag_msg)
            st.markdown(data["clarifying_question"])
            st.caption("_You can answer the follow-up below — no need to retype the original question._")
        else:
            st.session_state.pop("pending_question", None)
            answer_msg = {
                "role": "assistant",
                "kind": "answer",
                "data": data,
            }
            st.session_state.messages[msg_index] = answer_msg
            _persist_message(answer_msg)
            _render_answer(data)
            if data.get("success") and answer_msg.get("message_id"):
                _render_feedback(answer_msg)