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

Text patterns ('starts with', 'ends with', 'contains X' / 'has the word X'): \
filter with LIKE and place the wildcard on the correct side — column LIKE \
'X%' (starts with X), column LIKE '%X' (ends with X), column LIKE '%X%' \
(contains X). When the pattern qualifies the VALUES of a dimension ("how many \
country NAMES start with G"), count DISTINCT values of that column: \
COUNT(DISTINCT country) WHERE country LIKE 'G%' — NOT COUNT(*), which counts \
rows (here: customers), not country names. When the pattern qualifies \
individual rows ("how many passengers have a name containing John"), \
COUNT(*) WHERE <col> LIKE '...' is the correct form.

When you GROUP BY, the GROUP BY clause must contain EVERY non-aggregate \
column you reference in the SELECT list or the ORDER BY clause, otherwise \
SQL Server raises 'invalid in the ORDER BY/SELECT ... not contained in ... \
GROUP BY'. Either group by the key column itself (surrogate key or the \
natural key) and order by it, or order by an aggregate alias — never by a \
column that is neither grouped nor aggregated.

When a question ranks or counts per member of a dimension (e.g. 'highest X \
by / per <dim>', 'which <dim> had the most Y'), NULL members of that \
dimension are NOT real groups: exclude them with <dim> IS NOT NULL so a \
COALESCE 'Unknown' placeholder never outranks the real members. Only use \
COALESCE to LABEL a real group for display, never to turn a NULL into a \
competing rank.

When ordering results with LIMIT (e.g. 'top N' questions), always add a \
deterministic secondary sort key after the primary one — typically the \
relevant ID column — so that ties produce a consistent, repeatable order. \
For example: ORDER BY total_sales DESC, customer_id ASC.

Do NOT add a LIMIT clause unless the question explicitly asks for a specific \
number of results (e.g., "top 5", "the highest", "first 3"). A "by X" \
breakdown question wants ALL groups, not just one.

A fact/transaction table may store one row per LINE ITEM, so an order_number / \
order_id column repeats across many rows. A question about 'orders' means \
DISTINCT ORDERS, not line items: count them with COUNT(DISTINCT order_number) \
(or count inside a per-order subquery/CTE). A per-order average (e.g. 'average \
order value') must first aggregate to ONE row per order (SUM(...) GROUP BY \
order_number) in a subquery/CTE and then average over that — never apply AVG() \
to the line-level value directly, since multi-line orders would then be \
over-weighted.

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
is IN (the two named values).

When the question compares two subpopulations of the same 0/1 flag (e.g. \
'average fare paid by survivors vs non-survivors', survived=1 vs survived=0), \
return ONE row with a separate NAMED aggregate column per group, e.g. \
SELECT AVG(CASE WHEN survived = 1 THEN fare END) AS avg_fare_survived, \
AVG(CASE WHEN survived = 0 THEN fare END) AS avg_fare_not_survived \
FROM titanic; — do NOT GROUP BY the 0/1 flag: two rows labeled 0 and 1 are \
easy to misread, while named columns answer the comparison unambiguously."""

SYSTEM_PROMPT_MSSQL = """\
You are a SQL generator for a Microsoft SQL Server database. Given the schema \
and a question, write ONE single read-only SELECT statement that answers the \
question. Output ONLY the SQL with no explanation, no commentary, and no \
markdown code fences.

The question may be asked in a language other than English (e.g. Hindi). \
Translate the intent to English against the English schema provided, including \
mapping foreign-language filter values (e.g. country names) to the English \
values stored in the data.

Important: all date columns (order_date, birthdate, shipping_date, due_date, \
start_date, create_date) are stored as NVARCHAR in 'YYYY-MM-DD' format. \
For date filtering, use string comparison (e.g. order_date >= '2024-01-01') \
or the SUBSTRING() function. Do NOT use SQLite's DATE() function — it does \
not exist in SQL Server.

For any nullable text column that is grouped or displayed, wrap it in \
COALESCE(column, 'Unknown') so null/orphaned values render as 'Unknown' \
instead of blank or None.

Text patterns ('starts with', 'ends with', 'contains X' / 'has the word X'): \
filter with LIKE and place the wildcard on the correct side — column LIKE \
'X%' (starts with X), column LIKE '%X' (ends with X), column LIKE '%X%' \
(contains X). When the pattern qualifies the VALUES of a dimension ("how many \
country NAMES start with G"), count DISTINCT values of that column: \
COUNT(DISTINCT country) WHERE country LIKE 'G%' — NOT COUNT(*), which counts \
rows (here: customers), not country names. When the pattern qualifies \
individual rows ("how many passengers have a name containing John"), \
COUNT(*) WHERE <col> LIKE '...' is the correct form. LIKE on nvarchar is \
case-insensitive already — ends-with/contains filters do NOT need \
LEFT()/RIGHT()/CHARINDEX().

When a question ranks or counts per member of a dimension (e.g. 'highest X \
by / per <dim>', 'which <dim> had the most Y'), NULL members of that \
dimension are NOT real groups: exclude them with <dim> IS NOT NULL so a \
COALESCE 'Unknown' placeholder never outranks the real members. Only use \
COALESCE to LABEL a real group for display, never to turn a NULL into a \
competing rank.

When you GROUP BY, the GROUP BY clause must contain EVERY non-aggregate \
column you reference in the SELECT list or the ORDER BY clause, otherwise \
SQL Server raises 'invalid in the ORDER BY/SELECT ... not contained in ... \
GROUP BY'. Either group by the key column itself (surrogate key or the \
natural key) and order by it, or order by an aggregate alias — never by a \
column that is neither grouped nor aggregated.

When ordering results, use SELECT TOP (n) instead of LIMIT (LIMIT is not \
valid in SQL Server). Always add a deterministic secondary sort key after the \
primary one — typically the relevant ID column — so that ties produce a \
consistent, repeatable order. For example: SELECT TOP (5) ... ORDER BY \
total_sales DESC, customer_id ASC.

For "which X have more than N" questions, SELECT the grouping column AND the \
measure itself (e.g. SELECT Pclass, COUNT(*), then GROUP BY Pclass \
HAVING COUNT(*) > 200) so every qualifying group shows its value.

When selecting a ranked list (e.g. 'who are the 3 oldest passengers'), also \
include the table's id/key column in the SELECT list (e.g. PassengerId, Age) \
so every result row is uniquely identifiable and deterministically tie-broken \
with a secondary ORDER BY on that id column.

Do NOT add a TOP clause unless the question explicitly asks for a specific \
number of results (e.g., "top 5", "the highest", "first 3"). A "by X" \
breakdown question wants ALL groups, not just one.

When a fact table has several rows sharing the same order_number (line items), \
COUNT(*) or COUNT(order_number) counts LINE ITEMS, not orders. If the question \
asks 'how many orders' (e.g. filtered by date or customer), use \
COUNT(DISTINCT order_number), or count inside a per-order CTE. If the question \
asks to AVERAGE per order (e.g. 'average order value'), first build a CTE of \
per-order totals (SUM(sales_amount) GROUP BY order_number; per customer too if \
the breakdown is by customer), then AVG() over that CTE - never average the \
line-level sales_amount directly, since multi-line orders would be \
over-weighted and the result would not be an order average. Apply any 'orders \
with more than N' condition with HAVING inside that CTE, then aggregate and \
filter ON TOP of it.

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
SELECT TOP (1) c.country, AVG(ca.order_value) AS avg_order_value
FROM (
    SELECT customer_key, SUM(sales_amount) AS order_value, COUNT(*) AS n_orders
    FROM fact_sales
    GROUP BY customer_key
    HAVING COUNT(*) > 3
) ca
JOIN dim_customers c ON ca.customer_key = c.customer_key
GROUP BY c.country
ORDER BY avg_order_value DESC, c.country ASC;

When the question compares two specific named values within a dimension \
(e.g. 'road bikes and mountain bikes' — these are subcategory values), group \
by the column that contains those exact values (the subcategory column), \
NEVER by the parent category column. Filter WHERE that more-specific column \
is IN (the two named values).

When the question compares two subpopulations of the same 0/1 flag (e.g. \
'average fare paid by survivors vs non-survivors', survived=1 vs survived=0), \
return ONE row with a separate NAMED aggregate column per group, e.g. \
SELECT AVG(CASE WHEN survived = 1 THEN fare END) AS avg_fare_survived, \
AVG(CASE WHEN survived = 0 THEN fare END) AS avg_fare_not_survived \
FROM titanic; — do NOT GROUP BY the 0/1 flag: two rows labeled 0 and 1 are \
easy to misread, while named columns answer the comparison unambiguously."""

FENCE_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------- semantic guard
# Deterministic, schema-agnostic checks run against the GENERATED SQL before it
# executes. These catch the recurring aggregation-shape mistakes (counting line
# items when the question says "orders", averaging line amounts instead of
# per-order totals, dropping the measure from a HAVING query) and feed them back
# into the repair loop, so the model gets corrected before running, not after.


def _first_clause(sql: str) -> str:
    """The SELECT clause only (from the first SELECT up to the FROM)."""
    m = re.search(r"\bselect\b", sql, re.IGNORECASE)
    if not m:
        return sql
    rest = sql[m.end():]
    fm = re.search(r"\bfrom\b", rest, re.IGNORECASE)
    return rest[: fm.start()] if fm else rest


def semantic_feedback(question: str, sql: str) -> str | None:
    """Return repair feedback when the SQL shape contradicts the question's
    aggregation semantics, else None. Purely deterministic — no LLM call."""
    ql = question.lower()
    sql_l = sql.lower()

    # 1. 'how many orders' must not count line items.
    if re.search(r"\bhow many orders\b", ql) and not re.search(
        r"\bselect\s+distinct\b|distinct\s*\(|count\s*\(\s*distinct", sql_l
    ):
        if re.search(r"\bcount\s*\(", sql_l) and re.search(
            r"\bfact_sales\b|order_number", sql_l
        ):
            return (
                "This schema stores several rows per order (line items): "
                "COUNT(*) / COUNT(order_number) counts LINE ITEMS, not orders. "
                "Count orders with COUNT(DISTINCT order_number)."
            )

    # 2. 'average order value' must average per-ORDER totals, not line amounts.
    if re.search(r"\b(?:average|avg|mean)\s+order\s+value\b", ql):
        # A subquery / CTE is present but must group by order_number per country
        # and drop orphaned null countries so only real countries appear.
        per_order_ok = (
            re.search(r"\bwith\s+['\"`]?\w+", sql_l)
            and re.search(r"\bgroup\s+by\b[^;]*order_number", sql_l)
            and re.search(r"\bcountry\b", sql_l)
        )
        if not per_order_ok:
            return (
                "Average order value must come from per-ORDER totals. Build this "
                "exact CTE shape: WITH order_totals AS (SELECT c.country, "
                "f.order_number, SUM(f.sales_amount) AS order_value FROM fact_sales f "
                "JOIN dim_customers c ON f.customer_key = c.customer_key "
                "WHERE c.country IS NOT NULL GROUP BY c.country, f.order_number) "
                "SELECT country, AVG(order_value) AS avg_order_value FROM order_totals "
                "GROUP BY country HAVING COUNT(order_number) > 100 ORDER BY "
                "AVG(order_value) DESC. Only include known countries - exclude NULL "
                "country rows so every returned group is a real country."
            )

    # 3. 'more than N' (+GROUP BY/HAVING) must include the measure in the SELECT.
    if re.search(r"more\s+than\s+\d+", ql) and re.search(r"\bgroup\s+by\b", sql_l) and re.search(r"\bhaving\b", sql_l):
        if not re.search(r"\b(count|sum|avg|min|max)\s*\(", _first_clause(sql_l)):
            return (
                "The SELECT list only contains the grouping column(s); include the "
                "measure too (e.g. SELECT Pclass, COUNT(*) FROM titanic GROUP BY "
                "Pclass HAVING COUNT(*) > 200) so each qualifying group shows its value."
            )

    # 4. "no X recorded" / "missing X" / "without X" must filter on NULL, never
    # on the display placeholder 'Unknown' (that string is a COALESCE label that
    # is not actually stored in the data, so filtering on it yields zero rows).
    if re.search(r"\b(?:no|without|missing)\b.*\b(?:recorded|listed|filled|entered|value|cabin|ticket|date|name)\b", ql) or \
       re.search(r"\b(?:missing|not provided|unknown|blank|empty)\b", ql):
        is_null_ok = re.search(r"\bis\s+null\b|\bis\s+not\s+null\b|<>|!=|not\s+in", sql_l)
        placeholder_filter = re.search(r"=\s*['\"`]unknown['\"`]|=\s*['\"`]\s*['\"`]|is\s+(not\s+)?(distinct\s+from\s+)?['\"`]unknown['\"`]", sql_l)
        if not is_null_ok and placeholder_filter:
            return (
                "A missing/absent value must be filtered with IS NULL (or "
                "IS NULL OR <col> = '' AFTER COALESCE), never by comparing the "
                "column to the string 'Unknown'. 'Unknown' is only a display "
                "placeholder produced by COALESCE for the SELECT list — it is "
                "not stored in the data, so <col> = 'Unknown' matches zero "
                "rows. Rewrite the WHERE clause to use IS NULL."
            )

    # 5. rate / proportion questions must not use a bare integer AVG (SQL Server
    # truncates AVG of an int flag column to 0).
    if re.search(r"\b(?:rate|ratio|proportion|percentage|percent|share|fraction)\b", ql):
        m = re.search(r"\bavg\s*\(([^)]*)\)", sql_l)
        if m and not re.search(r"float|\bdouble\b|numeric|decimal|1\.0", m.group(1)):
            return (
                "A rate must come from a FLOAT average. AVG(<int flag column>) "
                "truncates to 0 on SQL Server (e.g. AVG(Survived) = 0). Write "
                "AVG(CAST(Survived AS float)) or AVG(Survived * 1.0) instead."
            )

    # 6. Ranked per-dimension groups (highest/lowest/top X by <dim>) must exclude
    # NULL members — a COALESCE 'Unknown' placeholder must never outrank real
    # groups (e.g. 'highest survival rate by port' must not report 'Unknown').
    # Only fire on true ranking-of-a-dimension phrasings ("which <dim> ... the
    # highest <measure>", or "<superlative> <measure> by <dim>"). NOT on
    # "sorted from highest to lowest", which just orders a pre-filtered set.
    _ranking_dim = re.search(
        r"\bwhich\s+\w+\b.{0,45}?\b(?:highest|lowest|best|worst|most|largest|smallest)\b"
        r"|\b(?:highest|lowest|best|worst|most|largest|smallest)\b(?:\s+\w+){0,3}\s+by\b",
        ql,
    )
    if _ranking_dim:
        gm = re.search(r"\bgroup\s+by\s+([\w.]+(?:\s*,\s*[\w.]+)*)", sql_l)
        if gm:
            cols = [c.strip().split(".")[-1] for c in gm.group(1).split(",")]
            missing_null_guard = [
                c for c in cols if not re.search(rf"\b{re.escape(c)}\s+is\s+not\s+null\b", sql_l)
            ]
            if missing_null_guard:
                c = missing_null_guard[0]
                return (
                    f"Ranking per member of '{c}': NULL is not a real member of "
                    f"that dimension. Exclude it with WHERE {c} IS NOT NULL (or "
                    f"{c} IS NOT NULL in the JOIN) so a COALESCE 'Unknown' "
                    f"placeholder never outranks the real groups."
                )

    # 7. 'average of flag X' must not zero-pad: AVG(CASE WHEN <cond> THEN <col>
    # ELSE 0 END) pulls the mean toward zero. Use ELSE NULL (or better, GROUP BY
    # the flag) so the average reflects only the subgroup.
    if re.search(r"\bavg\s*\(\s*case\b", sql_l) and re.search(r"\belse\s+0\b", sql_l):
        return (
            "AVG(CASE WHEN <cond> THEN <col> ELSE 0 END) is biased: the ELSE 0 "
            "adds a zero for every non-matching row and drags the average down. "
            "Either drop the ELSE entirely (AVG(CASE WHEN <cond> THEN <col> END), "
            "which treats non-matching rows as NULL) or GROUP BY the flag column "
            "and take AVG(<col>) per group."
        )

    return None


def _system_prompt_for(dialect: str) -> str:
    """System prompt tuned for the dialect the query will run against."""
    if dialect != "sqlite":
        return SYSTEM_PROMPT_MSSQL
    return SYSTEM_PROMPT


def generate_sql(question: str, schema_context: str, error_feedback: str = "", temperature: float = 0.2, dialect: str = "sqlite") -> str:
    """Generate a SQL query from a natural-language question."""
    prompt = f"Schema:\n{schema_context}\n\nQuestion: {question}"

    if error_feedback:
        prompt += f"\n\nYour previous attempt failed: {error_feedback}. Fix the query."

    try:
        raw = get_llm_client("sql").generate(
            prompt, system=_system_prompt_for(dialect), temperature=temperature
        )
    except LLMTimeoutError as exc:
        raise LLMTimeoutError(
            f"SQL generation timed out: {exc}"
        ) from exc

    match = FENCE_RE.search(raw)
    sql = match.group(1).strip() if match else raw.strip()

    return sql
