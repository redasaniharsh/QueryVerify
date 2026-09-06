"""
Turns the plain-English question + schema context into a SQL query, using the
local code-specialized model (qwen2.5-coder via get_llm_client("sql")).
"""

import re

from app.llm.factory import get_llm_client
from app.llm.ollama_client import LLMTimeoutError

SYSTEM_PROMPT = """\
You are a SQL generator for a SQLite database. Given the schema and a question, \
write ONE single read-only SELECT statement that answers the question. \
Output ONLY the SQL with no explanation, no commentary, and no markdown code fences.

The question may be asked in a language other than English (e.g. Hindi). \
Translate the intent to English against the English schema provided, including \
mapping foreign-language filter values (e.g. country names) to the English \
values stored in the data.

Important: all date columns (order_date, birthdate, shipping_date, due_date, \
start_date, create_date) are stored as TEXT in 'YYYY-MM-DD' format. \
For date filtering, use string comparison (e.g. order_date >= '2024-01-01') \
or SQLite's DATE() function.

For any nullable text column that is grouped or displayed, wrap it in \
COALESCE(column, 'Unknown') so null/orphaned values render as 'Unknown' \
instead of blank or None.

When ordering results with LIMIT (e.g. 'top N' questions), always add a \
deterministic secondary sort key after the primary one — typically the \
relevant ID column — so that ties produce a consistent, repeatable order. \
For example: ORDER BY total_sales DESC, customer_id ASC.

When a question filters by a per-entity count or aggregate BEFORE a further \
aggregation (e.g. 'customers who placed more than N orders', then average \
something about those customers), use a subquery or CTE that first computes \
and filters the per-entity aggregate, then joins and aggregates on that \
result. Do NOT apply the count filter as a HAVING clause on the wrong \
grouping level (e.g. filtering countries by order count instead of \
customers).

Example:
Question: Which country has the highest average order value, but only \
counting customers who placed more than 3 orders?
SQL:
SELECT c.country, AVG(ca.order_value) AS avg_order_value
FROM (
    SELECT customer_key, SUM(sales_amount) AS order_value, COUNT(*) AS n_orders
    FROM fact_sales
    GROUP BY customer_key
    HAVING COUNT(*) > 3
) ca
JOIN dim_customers c ON ca.customer_key = c.customer_key
GROUP BY c.country
ORDER BY avg_order_value DESC
LIMIT 1;

When the question compares two specific named values within a dimension \
(e.g. 'road bikes and mountain bikes' — these are subcategory values), group \
by the column that contains those exact values (the subcategory column), \
NEVER by the parent category column. Filter WHERE that more-specific column \
is IN (the two named values)."""

FENCE_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def generate_sql(question: str, schema_context: str, error_feedback: str = "", temperature: float = 0.2) -> str:
    """Generate a SQL query from a natural-language question."""
    prompt = f"Schema:\n{schema_context}\n\nQuestion: {question}"

    if error_feedback:
        prompt += f"\n\nYour previous attempt failed: {error_feedback}. Fix the query."

    try:
        raw = get_llm_client("sql").generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
    except LLMTimeoutError as exc:
        raise LLMTimeoutError(
            f"SQL generation timed out: {exc}"
        ) from exc

    match = FENCE_RE.search(raw)
    sql = match.group(1).strip() if match else raw.strip()

    return sql
