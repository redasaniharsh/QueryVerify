"""
SQLAlchemy engine setup, read from settings.database_url.

TODO (Week 1):
- create_engine(settings.database_url)
- a get_connection() helper the executor.py agent uses for sandboxed,
  read-only execution (e.g. open a connection with a query timeout, and for
  SQLite you can open in read-only mode via a URI: "file:./data/sample.db?mode=ro")
"""

from sqlalchemy import create_engine
from app.config import settings

engine = create_engine(settings.database_url)
