"""
Schema-aware starter question generator for QueryVerify.

Analyzes the active database schema (sample database or user-uploaded dataset)
and generates up to 4 natural, unambiguous starter questions suitable for
display as clickable cards on the empty chat screen.
"""

import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger("queryverify.starter_questions")

_ID_SUFFIXES = ("_id", "_key", "id", "key", "number", "code", "guid", "uuid")
_ID_EXACT = {"id", "key", "number", "passengerid", "order_number"}


def classify_columns(columns_with_types: List[tuple[str, str]]) -> Dict[str, List[str]]:
    """Classify column names into numeric, text/categorical, date, and id categories."""
    numeric: List[str] = []
    text_cols: List[str] = []
    date_cols: List[str] = []
    all_cols: List[str] = []

    for col, ctype in columns_with_types:
        all_cols.append(col)
        col_lower = col.lower()
        type_upper = str(ctype).upper()

        is_id = (
            col_lower in _ID_EXACT
            or any(col_lower.endswith(sfx) for sfx in _ID_SUFFIXES)
        )

        is_date = (
            any(t in type_upper for t in ("DATE", "TIME", "TIMESTAMP"))
            or "date" in col_lower
            or "time" in col_lower
        )

        is_categorical_name = (
            col_lower in ("pclass", "class", "tier", "grade", "rank", "level", "status", "type", "category", "code")
            or "class" in col_lower
            or "status" in col_lower
        )

        is_binary = (
            col_lower in ("survived", "active", "deleted", "enabled", "status_flag", "flag")
            or col_lower.startswith(("is_", "has_"))
        )

        is_num = any(
            t in type_upper
            for t in ("INT", "FLOAT", "REAL", "DOUBLE", "NUMERIC", "DECIMAL", "BIGINT", "SMALLINT")
        )

        if is_date:
            date_cols.append(col)
        elif is_num and not is_id and not is_binary and not is_categorical_name:
            numeric.append(col)
        else:
            # Text / categorical candidate
            if not is_id and not is_binary:
                text_cols.append(col)
            elif is_categorical_name:
                text_cols.append(col)

    return {
        "all": all_cols,
        "numeric": numeric,
        "text": text_cols,
        "date": date_cols,
    }


def introspect_sqlite_db(db_path: Path) -> List[Dict[str, Any]]:
    """Introspect an SQLite database file and return structured table metadata."""
    if not db_path.exists():
        return []
    tables: List[Dict[str, Any]] = []
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for (tbl_name,) in rows:
            col_info = cur.execute(f"PRAGMA table_info('{tbl_name}')").fetchall()
            cols_with_types = [(r[1], r[2]) for r in col_info]
            classified = classify_columns(cols_with_types)
            tables.append({
                "table": tbl_name,
                "numeric": classified["numeric"],
                "text": classified["text"],
                "date": classified["date"],
                "all": classified["all"],
            })
    except Exception as exc:
        logger.warning("Could not introspect SQLite db %s: %s", db_path, exc)
    finally:
        conn.close()
    return tables


def get_active_schema_tables(
    db_mode: Optional[str] = None,
    upload_db_path: Optional[str] = None,
    upload_schema: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Return structured table and column classifications for the active database."""
    if db_mode == "My Uploaded Data" and upload_db_path:
        path = Path(upload_db_path)
        tables = introspect_sqlite_db(path)
        if tables:
            return tables
        # Fallback if SQLite introspection failed but upload_schema exists
        if upload_schema:
            result = []
            for item in upload_schema:
                t_name, cols = item[0], item[1]
                classified = classify_columns([(c, "TEXT") for c in cols])
                result.append({
                    "table": t_name,
                    "numeric": classified["numeric"],
                    "text": classified["text"],
                    "date": classified["date"],
                    "all": cols,
                })
            return result

    # Default / Sample database: try SQLAlchemy default_engine, then fallback to data/sample.db
    try:
        from app.db.connection import default_engine
        from sqlalchemy import inspect
        engine = default_engine()
        inspector = inspect(engine)
        tables = []
        for tbl in inspector.get_table_names():
            columns = inspector.get_columns(tbl)
            cols_with_types = [(c["name"], str(c["type"])) for c in columns]
            classified = classify_columns(cols_with_types)
            tables.append({
                "table": tbl,
                "numeric": classified["numeric"],
                "text": classified["text"],
                "date": classified["date"],
                "all": classified["all"],
            })
        if tables:
            return tables
    except Exception as exc:
        logger.info("SQLAlchemy default_engine inspection not available: %s", exc)

    # Fallback: check data/sample.db
    sample_db = Path(__file__).resolve().parent.parent / "data" / "sample.db"
    if sample_db.exists():
        return introspect_sqlite_db(sample_db)

    # Hard static fallback for offline sample schema
    return [
        {
            "table": "fact_sales",
            "numeric": ["sales_amount", "quantity", "price"],
            "text": [],
            "date": ["order_date"],
            "all": ["order_number", "product_key", "customer_key", "order_date", "sales_amount", "quantity", "price"],
        },
        {
            "table": "dim_customers",
            "numeric": [],
            "text": ["country", "first_name", "last_name", "marital_status", "gender"],
            "date": [],
            "all": ["customer_key", "customer_id", "first_name", "last_name", "country"],
        },
        {
            "table": "dim_products",
            "numeric": ["cost"],
            "text": ["category", "subcategory", "product_name"],
            "date": [],
            "all": ["product_key", "product_id", "product_name", "category", "subcategory", "cost"],
        },
    ]


def generate_starter_questions(tables: List[Dict[str, Any]]) -> List[str]:
    """Generate up to 4 schema-aware starter questions.

    Guaranteed to return 0-4 valid string questions without None, without
    crashing on edge-case schemas (missing text, missing numeric, single table).
    """
    if not tables:
        return []

    questions: List[str] = []

    # Choose primary table (prefer fact / main data table if available)
    primary_table = tables[0]["table"]
    for kw in ("fact", "sales", "titanic", "orders", "main", "data", "customers"):
        match = next((t["table"] for t in tables if kw in t["table"].lower()), None)
        if match:
            primary_table = match
            break

    # 1. Row count / volume question
    questions.append(f"How many rows are in {primary_table}?")

    # 2. Metric aggregation (total / average)
    metric_q: Optional[str] = None
    # Check fact / primary table first
    ordered_tables = sorted(
        tables,
        key=lambda t: 0 if any(k in t["table"].lower() for k in ("fact", "sales", "titanic")) else 1,
    )
    for t in ordered_tables:
        tbl = t["table"]
        for col in t.get("numeric", []):
            cl = col.lower()
            if any(kw in cl for kw in ("amount", "sales", "total", "revenue", "fare", "price")):
                metric_q = f"What is the total {col} in {tbl}?"
                break
            elif any(kw in cl for kw in ("cost",)):
                metric_q = f"What is the total {col} in {tbl}?"
                break
            elif any(kw in cl for kw in ("age", "rate", "score", "percent", "avg")):
                metric_q = f"What is the average {col} in {tbl}?"
                break
        if metric_q:
            break

    if not metric_q:
        # Fallback to first available numeric column
        for t in ordered_tables:
            tbl = t["table"]
            if t.get("numeric"):
                col = t["numeric"][0]
                metric_q = f"What is the average {col} in {tbl}?"
                break

    if metric_q and metric_q not in questions:
        questions.append(metric_q)

    # 3. Grouping / Breakdown (numeric metric by text category)
    breakdown_q: Optional[str] = None
    # Check intra-table first (e.g. titanic Fare by Pclass)
    for t in tables:
        tbl = t["table"]
        num_cols = sorted(
            t.get("numeric", []),
            key=lambda c: 0 if any(kw in c.lower() for kw in ("fare", "amount", "sales", "price", "cost", "revenue", "age", "quantity")) else 1
        )
        txt_cols = sorted(
            [c for c in t.get("text", []) if c.lower() not in ("name", "id", "ticket", "first_name", "last_name", "product_name")],
            key=lambda c: 0 if any(kw in c.lower() for kw in ("pclass", "category", "country", "sex", "gender", "embarked", "status", "type")) else 1
        )
        used_cols = [c for c in num_cols if metric_q and c.lower() in metric_q.lower()]
        unused_cols = [c for c in num_cols if c not in used_cols]
        active_num_cols = unused_cols if unused_cols else num_cols
        if active_num_cols and txt_cols:
            n_col = active_num_cols[0]
            c_col = txt_cols[0]
            agg = "total" if any(kw in n_col.lower() for kw in ("amount", "sales", "cost", "revenue")) else "average"
            breakdown_q = f"What is the {agg} {n_col} by {c_col} in {tbl}?"
            break

    # If no intra-table breakdown, check cross-table (e.g. fact_sales + dim_products/dim_customers)
    if not breakdown_q and len(tables) > 1:
        num_table = next((t for t in tables if t.get("numeric") and any(k in t["table"].lower() for k in ("fact", "sales", "data"))), None) or next((t for t in tables if t.get("numeric")), None)
        txt_table = next((t for t in tables if t != num_table and t.get("text")), None)
        if num_table and txt_table:
            num_cols = sorted(
                num_table["numeric"],
                key=lambda c: 0 if any(kw in c.lower() for kw in ("fare", "amount", "sales", "price", "cost", "revenue", "age", "quantity")) else 1
            )
            txt_candidates = sorted(
                [c for c in txt_table["text"] if any(kw in c.lower() for kw in ("category", "country", "status", "type", "gender", "department"))],
                key=lambda c: 0 if any(kw in c.lower() for kw in ("category", "country", "status")) else 1
            )
            n_col = num_cols[0]
            c_col = txt_candidates[0] if txt_candidates else txt_table["text"][0]
            agg = "total" if any(kw in n_col.lower() for kw in ("amount", "sales", "cost", "revenue")) else "average"
            breakdown_q = f"What is the {agg} {n_col} by {c_col} in {txt_table['table']}?"

    if breakdown_q and breakdown_q not in questions:
        questions.append(breakdown_q)

    # 4. Distinct count or category question
    distinct_q: Optional[str] = None
    for t in tables:
        tbl = t["table"]
        for col in t.get("text", []):
            cl = col.lower()
            if any(kw in cl for kw in ("country", "category", "embarked", "pclass", "gender", "sex", "status", "city", "department", "tag")):
                candidate = f"How many distinct {col} are in {tbl}?"
                if candidate not in questions:
                    distinct_q = candidate
                    break
        if distinct_q:
            break

    if distinct_q and distinct_q not in questions:
        questions.append(distinct_q)

    # Fill remaining slots up to 4 with any valid row counts or distinct queries
    for t in tables:
        tbl = t["table"]
        q_row = f"How many rows are in {tbl}?"
        if q_row not in questions and len(questions) < 4:
            questions.append(q_row)
        for col in t.get("text", []):
            q_dist = f"How many distinct {col} are in {tbl}?"
            if q_dist not in questions and len(questions) < 4:
                questions.append(q_dist)

    # Ensure uniqueness and limit to 4
    seen = set()
    result: List[str] = []
    for q in questions:
        if q and q not in seen:
            seen.add(q)
            result.append(q)
        if len(result) == 4:
            break

    return result
