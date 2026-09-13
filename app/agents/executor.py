"""
Runs the generated SQL safely and returns either a result or an error.
"""

import sqlparse
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from app.config import settings


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


def _capped_result(columns: list, raw_rows: list, max_rows: int) -> dict:
    """Build a serializable result dict, truncating oversized sets with a
    clear notice. Never hides the truncation: the UI shows the notice so the
    user knows the answer is partial and is told how to narrow it."""
    truncated = len(raw_rows) > max_rows
    kept = raw_rows[:max_rows]
    out = {
        "success": True,
        "columns": columns,
        "rows": _normalize_nulls([dict(zip(columns, row)) for row in kept]),
    }
    if truncated:
        out["truncated"] = True
        out["total_rows"] = None
        out["notice"] = (
            f"The query returned more than {max_rows} rows. Showing the first "
            f"{max_rows} — narrow the question (filter, LIMIT/SELECT TOP, or "
            "aggregate) to see the exact total."
        )
    return out


def execute_sql(sql: str, engine) -> dict:
    """Execute a SQL query with a timeout and return a serializable result dict.

    Result sets are capped at settings.max_result_rows; anything larger is
    truncated with a notice (never fetched into one giant list, never hung)."""
    rejection = _reject_reason(sql)
    if rejection:
        return {"success": False, "error": rejection}

    max_rows = settings.max_result_rows

    def _run():
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            # fetch max_rows+1 so we can tell whether more rows exist without
            # ever pulling an unbounded set into memory.
            raw = list(result.fetchmany(max_rows + 1))
            return _capped_result(columns, raw, max_rows)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            return future.result(timeout=QUERY_TIMEOUT)
    except FuturesTimeout:
        return {"success": False, "error": f"Query timed out after {QUERY_TIMEOUT}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
