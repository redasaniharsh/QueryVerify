"""
Ties generation + execution + verification together into the retry loop,
with self-consistency cross-check on first verified match.
"""

import re
from time import perf_counter

from sqlalchemy import inspect, text

from app.config import settings
from app.agents.sql_generator import generate_sql
from app.agents.executor import execute_sql
from app.agents.verifier import verify_result
from app.llm.ollama_client import LLMTimeoutError
from app.utils.trace import add as trace_add


_ID_SUFFIXES = ("_id", "_key", "_number", "_code", "_sku", "_no")

# "by X" breakdown detection: which dimension columns form the groups of a full
# breakdown question, and (via _expected_group_count) how many groups each has.
_DIM_COLUMN = {
    "product category": "category",
    "product subcategory": "subcategory",
    "product line": "product_line",
    "marital status": "marital_status",
    "product name": "product_name",
    "category": "category",
    "categories": "category",
    "subcategory": "subcategory",
    "subcategories": "subcategory",
    "country": "country",
    "countries": "country",
    "gender": "gender",
    "product": "product_name",
    "products": "product_name",
}

_BREAKDOWN_DIM_RE = re.compile(
    r"\b(?:(?:(?:grouped|broken)\s+)?(?:by|per)|(?:for\s+)?each)\s+"
    r"(product category|product subcategory|product line|marital status|"
    r"product name|category|categories|subcategory|subcategories|"
    r"country|countries|gender|product|products)\b",
    re.IGNORECASE,
)

_DISTINCT_GROUP_CACHE: dict[tuple[str, str], int] = {}


def _table_owning_column(engine, column: str) -> str | None:
    """First table that owns a column with this name (schema-internal names only)."""
    inspector = inspect(engine)
    for table in inspector.get_table_names():
        if column in {c["name"].lower() for c in inspector.get_columns(table)}:
            return table
    return None


def _expected_group_count(engine, column: str) -> int | None:
    """How many groups a 'by <column>' breakdown should produce, read once from
    the dimension table that owns the column and cached per (table, column)."""
    table = _table_owning_column(engine, column)
    if table is None:
        return None
    key = (table, column)
    if key not in _DISTINCT_GROUP_CACHE:
        with engine.connect() as conn:
            n = conn.execute(
                text(
                    f'SELECT COUNT(DISTINCT COALESCE("{column}", \'Unknown\')) '
                    f'FROM "{table}"'
                )
            ).scalar()
        _DISTINCT_GROUP_CACHE[key] = int(n or 0)
    return _DISTINCT_GROUP_CACHE[key]


def _breakdown_suspicion(question: str, result_rows: int, engine) -> tuple[str, int] | None:
    """Return (grouping_column, expected_group_count) when the question asks for a
    full 'by X' breakdown but the result has significantly fewer rows than the
    dimension allows (the accidental-LIMIT signature), else None. Fires only when
    the row count is well below the expected group count, so legitimate filtered
    subsets that still return most groups are left alone."""
    q = question.lower()
    m = _BREAKDOWN_DIM_RE.search(q)
    if not m:
        return None
    col = _DIM_COLUMN.get(m.group(1).lower())
    if col is None:
        return None
    expected = _expected_group_count(engine, col)
    if expected is None or expected < 2:
        return None
    if result_rows < max(2, expected // 2):
        return col, expected
    return None


def _is_id_column(name: str) -> bool:
    """Identifier-like columns (primary/surrogate/business keys) that only
    identify a row and carry no answer content: id, *_id, *_key, *_number,
    *_code, *_sku, *_no. Generalized so any table's key columns count, not
    just the customer pair."""
    low = name.lower()
    if low == "id" or low == "sku":
        return True
    return any(low.endswith(s) for s in _ID_SUFFIXES)


def _row_signatures(rows: list[dict], id_key: str | None) -> set:
    """Identify each row by its entity: the id value if present, otherwise the
    group-by label value(s) (e.g. country name) with their numeric totals, and
    only as a last resort the rank position. Identifier columns are dropped
    from the comparison entirely, so two identical answers that merely project
    different key columns (customer_id vs customer_number vs none) compare
    equal instead of being counted as disagreements. This makes comparisons
    robust to row order and column-projection differences while still catching
    genuinely different answers (different labels, totals, or rankings)."""
    sig = set()
    for rank, row in enumerate(rows, start=1):
        if id_key is not None and row.get(id_key) is not None:
            identity = f"id:{row[id_key]}"
        else:
            labels = tuple(
                sorted(
                    str(v)
                    for k, v in row.items()
                    if isinstance(v, str) and v and not _is_id_column(k)
                )
            )
            identity = f"label:{labels}" if labels else f"rank:{rank}"
        numerics = tuple(
            sorted(
                v
                for k, v in row.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and not _is_id_column(k)
            )
        )
        sig.add((identity, numerics))
    return sig


def _identity_key(all_columns: list[set]) -> str | None:
    """Pick the strongest identifier column shared by every compared result
    (prefer true ids/keys over *_number / *_code), or None, in which case
    rows fall back to their non-identifier label values."""
    if not all_columns:
        return None
    common = set.intersection(*[set(cols) for cols in all_columns])

    def rank(c: str) -> int:
        low = c.lower()
        if low == "id" or low.endswith("_id"):
            return 0
        if low.endswith("_key"):
            return 1
        return 2

    candidates = sorted((c for c in common if _is_id_column(c)), key=lambda c: (rank(c), c))
    return candidates[0] if candidates else None


def _check_consistency(question: str, schema_context: str, engine, original_result: dict, trace: list | None = None) -> tuple[float, str]:
    """Generate additional samples and compare the ranked entities (id or
    group-by label + numeric values). A candidate whose SQL fails to execute
    is retried with repair feedback before it counts as a non-agreeing sample.
    Returns (score, note)."""
    t0 = perf_counter()
    candidates = []
    dropped = 0
    while len(candidates) < settings.self_consistency_samples - 1:
        dropped += 1
        if dropped > settings.self_consistency_samples * 3:
            break
        feedback = ""
        for c_attempt in range(settings.max_repair_attempts):
            try:
                sql = generate_sql(question, schema_context, error_feedback=feedback, temperature=0.6)
            except LLMTimeoutError:
                feedback = "SQL generation timed out; please try again."
                continue
            result = execute_sql(sql, engine)
            if result.get("success"):
                candidates.append(result)
                break
            feedback = result["error"]
    dropped -= len(candidates)

    all_columns = [set(original_result["columns"])] + [set(c["columns"]) for c in candidates]
    id_key = _identity_key(all_columns)

    original_sig = _row_signatures(original_result["rows"], id_key)
    agreement_count = 1 + sum(
        1
        for c in candidates
        if _row_signatures(c["rows"], id_key) == original_sig
    )

    score = agreement_count / settings.self_consistency_samples
    note = (
        f"Cross-checked against {settings.self_consistency_samples} candidates; "
        f"{agreement_count} agreed."
    )
    actual_rows = len(original_result.get("rows", []))
    susp = _breakdown_suspicion(question, actual_rows, engine)
    if susp:
        col, expected = susp
        if score >= 0.8:
            score = 2 / settings.self_consistency_samples
        note += (
            f" Answer may be incomplete: roughly {expected} groups are expected by "
            f"{col}, yet only {actual_rows} row(s) returned, so confidence is capped "
            f"at medium even though {agreement_count} candidate(s) agreed."
        )
    if dropped:
        note += f" {dropped} candidate sample(s) could not be produced after retries."
    trace_add(trace, "consistency", note[:200], perf_counter() - t0)
    return score, note


def _score_to_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def run_pipeline(question: str, schema_context: str, engine, trace: list | None = None) -> dict:
    """Run the generate -> execute -> verify loop with repair attempts."""
    error_feedback = ""
    sql = ""
    result = {}
    verification = {}

    for attempt in range(settings.max_repair_attempts):
        t0 = perf_counter()
        try:
            sql = generate_sql(
                question, schema_context, error_feedback, temperature=0.2
            )
        except LLMTimeoutError as exc:
            trace_add(
                trace,
                "generation",
                f"Attempt {attempt + 1}: generation timed out — retrying",
                perf_counter() - t0,
            )
            error_feedback = str(exc)
            continue
        gen_dur = perf_counter() - t0
        if error_feedback:
            trace_add(
                trace,
                "generation",
                f"Attempt {attempt + 1} (repaired): generated SQL — previous "
                f"failure: {error_feedback[:100]}",
                gen_dur,
            )
        else:
            trace_add(
                trace, "generation", f"Attempt {attempt + 1}: generated SQL", gen_dur
            )

        t0 = perf_counter()
        result = execute_sql(sql, engine)
        exec_dur = perf_counter() - t0
        if result["success"]:
            trace_add(
                trace,
                "execution",
                f"Executed — {len(result.get('rows') or [])} row(s)",
                exec_dur,
            )
        else:
            trace_add(
                trace,
                "execution",
                f"Failed — {(result.get('error') or '')[:120]}",
                exec_dur,
            )

        if not result["success"]:
            error_feedback = result["error"]
            continue

        susp = _breakdown_suspicion(question, len(result.get("rows", [])), engine)
        if susp:
            col, expected = susp
            error_feedback = (
                f"Expected roughly {expected} groups for the '{col}' breakdown, but "
                f"the query returned only {len(result.get('rows', []))} row(s). Did "
                f"you accidentally add a LIMIT? Remove it and return ALL groups."
            )
            continue

        t0 = perf_counter()
        verification = verify_result(question, sql, result)
        ver_dur = perf_counter() - t0
        if verification["matches"]:
            trace_add(
                trace,
                "verification",
                f"Accepted — {(verification['reason'] or '')[:160]}",
                ver_dur,
            )
            confidence_score, consistency_note = _check_consistency(
                question, schema_context, engine, result, trace
            )
            explanation = verification["reason"]
            if consistency_note:
                explanation += f" {consistency_note}"

            return {
                "success": True,
                "sql": sql,
                "result": result,
                "attempts": attempt + 1,
                "confidence": _score_to_label(confidence_score),
                "confidence_score": confidence_score,
                "explanation": explanation,
            }

        trace_add(
            trace,
            "verification",
            f"Rejected — {(verification['reason'] or '')[:160]}",
            ver_dur,
        )
        error_feedback = verification["reason"]
        if result.get("success"):
            rows_preview = str(result.get("rows", [])[:3])
            error_feedback += (
                f" Your last query returned these rows: {rows_preview}. "
                "If the rows look wrong or empty, fix the WHERE/GROUP BY/aggregation."
            )

    return {
        "success": False,
        "sql": sql,
        "result": result,
        "attempts": settings.max_repair_attempts,
        "confidence": "low",
        "confidence_score": 0.0,
        "explanation": "Could not verify a confident answer after multiple attempts: "
        + (verification.get("reason") or error_feedback or "unknown"),
    }
