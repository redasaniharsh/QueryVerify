"""
SQLAlchemy engines, created per database on demand.

There is deliberately NO single module-level global engine: /ask decides which
database this request targets (the fixed sample by default, or a session-scoped
user upload) and calls get_engine(database) for exactly that one file. Engines
are cached by resolved path and use NullPool so each connect() opens/closes its
own SQLite handle — nothing holds the file open, so uploaded databases can be
cleaned up while the backend is running.

Uploaded databases are only ever accepted from the project data/ directory
(bare file names like user_upload_<session_id>.db), never from arbitrary paths.
"""

from pathlib import Path
import re

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from app.config import settings

# Repository data directory: app/db/connection.py -> repo/data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Only these files may be addressed as an uploaded database. Rejects path
# traversal / arbitrary reads; the sample DB is addressed via database=None.
UPLOAD_NAME_RE = re.compile(r"^user_upload_[A-Za-z0-9_-]+\.db$")

_ENGINES: dict[str, Engine] = {}


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def default_engine():
    """Engine for the sample database, from settings.database_url as before."""
    return create_engine(settings.database_url)


def resolve_db(database: str | None) -> Path:
    """Validate a request's database selector and return its absolute path.

    None / "" / "sample" -> the configured sample DB;
    a bare "user_upload_<id>.db" file name -> DATA_DIR / that file;
    anything else -> ValueError (client error).
    """
    if not database:
        url_database = settings.database_url.split("///", 1)
        if len(url_database) == 2 and url_database[1]:
            # Relative sqlite URLs resolve against the process CWD exactly as
            # create_engine(settings.database_url) always did.
            return (Path.cwd() / Path(url_database[1])).resolve()
        return (DATA_DIR / "sample.db").resolve()
    name = str(database).replace("\\", "/").rsplit("/", 1)[-1]
    if not UPLOAD_NAME_RE.match(name):
        raise ValueError(
            f"Unsupported database: {database!r}. Only user_upload_*.db files "
            "inside the data/ directory are allowed."
        )
    return (DATA_DIR / name).resolve()


def get_engine(database: str | None = None):
    """Return an engine for the given database (default: the sample DB).

    When the configured DATABASE_URL is a server database (e.g. SQL Server),
    the default engine is built straight from that URL, exactly as
    default_engine() would. Session-scoped SQLite uploads still resolve through
    resolve_db() so their file lives in data/ and is never locked.
    """
    if not database and not settings.database_url.startswith("sqlite"):
        key = f"url::{settings.database_url}"
        engine = _ENGINES.get(key)
        if engine is None:
            engine = create_engine(settings.database_url)
            _ENGINES[key] = engine
        return engine

    path = resolve_db(database)
    if not path.exists():
        raise ValueError(
            f"Database not found: {path.name} — upload a CSV first, or the "
            "session it belonged to has ended."
        )
    key = str(path)
    engine = _ENGINES.get(key)
    if engine is None:
        engine = create_engine(
            _sqlite_url(path),
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        _ENGINES[key] = engine
    return engine


def clear_engine_cache() -> None:
    """Drop all cached engines (tests / teardown)."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()
