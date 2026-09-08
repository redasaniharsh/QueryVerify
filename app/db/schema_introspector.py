"""
Reads real table/column names + types from the connected database, so the
SQL-generation prompt is grounded in the actual schema instead of the model
guessing column names. Also includes a few sample values for text columns so
the model can write correct filters (e.g. know that 'road bike' products are
named like 'Road-450 Red- 44', or that countries start with uppercase).
"""

from pathlib import Path

from sqlalchemy import inspect, text

from app.config import settings

SAMPLE_LIMIT = 3

# The sample CSV export does not declare FOREIGN KEY clauses in SQLite even
# though the joins are real. Those two join rules are curated domain knowledge
# that MUST keep appearing for the default database (dropping them would change
# what the generated-SQL model already trusts). Any other database — including
# user uploads — reports real declared foreign keys, so uploaded schemas never
# inherit sample-only facts.
SAMPLE_FK_LINES = (
    "  fact_sales.customer_key -> dim_customers.customer_key",
    "  fact_sales.product_key -> dim_products.product_key",
)


def _is_default_database(engine) -> bool:
    """True when the engine targets the configured default sample database."""
    url = engine.url
    if url.get_backend_name() != "sqlite":
        return False
    db = url.database or ""

    def _norm(p: str) -> str:
        return str(Path(p).resolve()).lower() if p else ""

    configured = settings.database_url.split(":///", 1)[-1]
    return _norm(db) == _norm(configured)


def _sample_values(conn, table: str, column: str) -> list[str]:
    q = text(
        f'SELECT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL AND CAST("{column}" AS TEXT) <> \'\' '
        f'GROUP BY "{column}" ORDER BY COUNT(*) DESC, MIN("{column}") LIMIT :n'
    )
    return [str(row[0]) for row in conn.execute(q, {"n": SAMPLE_LIMIT}).fetchall()]


def get_schema_context(engine) -> str:
    """Return a compact text block of the database schema for LLM prompts."""
    inspector = inspect(engine)
    lines = []

    fk_lines: list[str] = []

    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            col_defs = ", ".join(
                f"{col['name']} {col['type']}" for col in columns
            )
            lines.append(f"Table: {table_name} ({col_defs})")
            text_cols = [
                col["name"]
                for col in columns
                if str(col["type"]).upper().startswith(("TEXT", "CHAR", "VARCHAR"))
            ]
            for col in text_cols:
                samples = _sample_values(conn, table_name, col)
                if samples:
                    lines.append(
                        f"  sample values for {col}: {samples}"
                    )

            for fk in inspector.get_foreign_keys(table_name):
                for i, cc in enumerate(fk.get("constrained_columns", [])):
                    rc = (fk.get("referred_columns") or [None])[i] or cc
                    fk_lines.append(
                        f"  {table_name}.{cc} -> "
                        f"{fk['referred_table']}.{rc}"
                    )

    lines.append("")
    lines.append("Foreign keys:")
    if _is_default_database(engine):
        lines.extend(SAMPLE_FK_LINES)
    elif fk_lines:
        lines.extend(fk_lines)
    else:
        lines.append("  (none declared)")

    return "\n".join(lines)