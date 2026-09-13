"""
Runs the generated SQL safely and returns either a result or an error.
"""

import sqlparse
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


def _reject_reason(sql: str) -> str | None:
    """Return why a query is rejected, or None if it is a single read-only SELECT.

    ONE leading WITH clause (a CTE) is allowed as long as the terminating
    command is still a single SELECT. Depth-0 token scan only: sqlparse renders
    each CTE definition list as one generic token, so the only DML keywords seen
    at this level are the real command verbs."""
    parsed = sqlparse.parse(sql)
    if not parsed:
        return "Empty query"
    if len(parsed) > 1:
        return "Only a single SELECT statement is allowed"
    saw_with = False
    for tok in parsed[0].tokens:
        if tok.is_whitespace or (tok.ttype is not None and sqlparse.tokens.Comment in tok.ttype):
            continue
        if tok.ttype is sqlparse.tokens.Keyword.CTE:
            saw_with = True
            continue
        if tok.ttype is sqlparse.tokens.Keyword.DML:
            if tok.value.upper() != "SELECT":
                return f"Only SELECT statements are allowed (found {tok.value.upper()})"
            return None
        if not saw_with:
            return "Only SELECT statements are allowed"
        # saw_with + generic CTE-definition token: keep scanning for the
        # terminating SELECT command.
    return "Only SELECT statements are allowed (a WITH query must end in SELECT)"


def is_read_only(sql: str) -> bool:
    """Basic guard: only allow a single read-only SELECT statement through."""
    return _reject_reason(sql) is None


QUERY_TIMEOUT = 30  # seconds


def _normalize_nulls(rows: list[dict]) -> list[dict]:
    """Deterministic display fix: render any None/null value as 'Unknown'."""
    return [
        {k: ("Unknown" if v is None else v) for k, v in row.items()}
        for row in rows
    ]


def execute_sql(sql: str, engine) -> dict:
    """Execute a SQL query with a timeout and return a serializable result dict."""
    rejection = _reject_reason(sql)
    if rejection:
        return {"success": False, "error": rejection}

    def _run():
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = _normalize_nulls([dict(zip(columns, row)) for row in result.fetchall()])
            return {"success": True, "columns": columns, "rows": rows}

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            return future.result(timeout=QUERY_TIMEOUT)
    except FuturesTimeout:
        return {"success": False, "error": f"Query timed out after {QUERY_TIMEOUT}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
