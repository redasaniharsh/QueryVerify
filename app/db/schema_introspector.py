"""
Reads real table/column names + types from the connected database, so the
SQL-generation prompt is grounded in the actual schema instead of the model
guessing column names. Also includes a few sample values for text columns so
the model can write correct filters (e.g. know that 'road bike' products are
named like 'Road-450 Red- 44', or that countries start with uppercase).
"""

from sqlalchemy import inspect, text

SAMPLE_LIMIT = 3


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

    lines.append("")
    lines.append("Foreign keys:")
    lines.append("  fact_sales.customer_key -> dim_customers.customer_key")
    lines.append("  fact_sales.product_key -> dim_products.product_key")

    return "\n".join(lines)