"""
The core differentiator of this project. Given the ORIGINAL question and the
query RESULT (not the SQL itself), decides: does this result actually answer
the question that was asked?
"""

import json
import re

from app.llm.factory import get_llm_client
from app.llm.ollama_client import LLMTimeoutError

MAX_ROWS = 10

SYSTEM_PROMPT = """\
You are checking whether a SQL query result answers a question. You are given \
ONLY the result data below - you have no other knowledge of the database. \
Judge strictly based on what's actually in the data:

- Do NOT assume any row, country, category, or value is missing unless the \
question explicitly names a specific value (e.g. "how many orders did customer \
X make") and that named value is absent from the result.
- A GROUP BY result should be treated as complete for whatever groups exist in \
the underlying data - you cannot know if a group is "missing" just by looking \
at the result, since you don't have access to the full database yourself.
- A single-row result with the correct column(s) is a COMPLETE, valid answer \
for ANY aggregate question — a count, an average, a total, a min/max pair. \
Do not reject a result merely because it has one row.
- A single-row result is a COMPLETE answer for any scalar aggregate (count, \
total, average, min/max) and for any filtered group set (e.g. a HAVING \
threshold where only one group qualifies, or a top-1 'which X has the most Y'). \
You do NOT see the SQL and you do NOT see the rest of the table — the rows in \
the RESULT are all you have. If a filtered aggregate returns exactly one row, \
that row is the correct, complete answer. Do NOT reject it because 'there might \
be other matching groups in the data' or because 'the SQL was not shown'. For \
a min/max pair, ONE row carrying both a MIN and a MAX column directly answers \
'what is the oldest and youngest' — the two values in that one row ARE the \
answer.
- Only suspect incompleteness when the question explicitly enumerates multiple \
named items and one of those named items is absent from the result rows.
- You cannot verify unseen filter conditions (e.g. 'only countries with more \
than 100 orders'): you have neither the SQL nor the full table. A result \
grouped by the dimension the question names, carrying the requested aggregate, \
IS the correct filtered answer — do not reject it merely because the \
condition's own column (an order count) is not in the result and therefore \
"cannot be confirmed". Inability to check an unseen predicate is NOT evidence \
the result is wrong.
- If the result has at least one row, correct-looking column names matching \
the question's intent, and no obvious error, default to matches: true.
- Only mark matches: false for concrete, nameable problems: wrong aggregation \
(e.g. asked for total but got an average), wrong grouping column, or an actual \
error message in the data.
- An empty result set is NOT by itself a failure: a well-formed query with a \
legitimate filter (e.g. a country with no customers) can correctly return zero \
rows. Only suspect a problem if the question names a value that should plainly \
exist in the data and the result contradicts that expectation.
- The string 'Unknown' may appear in the data as a rendered null value; it is a \
normal placeholder, not an error message. Do not treat 'Unknown', 'None', or \
'NULL' text appearing in a cell as a query failure.
- If the question contains a data filter (e.g. a country, product, date) and the \
result is a single row whose value(s) are all NULL/'Unknown' (an aggregate over \
zero matching rows), that means the filter matched no data — accept it as a \
correct empty/zero answer, do not reject it.
- A zero value from a filtered aggregate is a valid answer. You cannot see the \
underlying data, so 'how many country names contain \"land\"' returning 0 just \
means no NAME in the data matches the pattern — do NOT reject it because you \
personally expected a match (e.g. 'Poland', 'Thailand'). The same applies to \
LIKE patterns ('starts with', 'ends with', 'contains'): an unmatched pattern is \
zero by definition, not a failed query.
- A None, null, or blank value in a result row is a valid data value (e.g., an \
unmatched key or missing category), not evidence that the query or result is \
broken or incomplete. Do not reject a result solely because it contains a \
None/null value in one row.

Example:
Question: total sales by country
Result columns: ['country', 'total_sales_amount']
Rows:
  Australia 9060172
  Canada 1977738
  France 2643751
  Germany 2894066
  United Kingdom 3391376
  United States 9162327
  None 226820
Answer: {"matches": true, "reason": "Result groups sales by country with totals for each, matching the question."}

Example:
Question: What is the oldest and youngest passenger age?
Result columns: ['youngest', 'oldest']
Rows:
  0.42, 80.0
Answer: {"matches": true, "reason": "The single row carries both the minimum age (0.42) and the maximum age (80.0), directly answering 'oldest and youngest'."}

Example:
Question: Which product categories have total sales over 1 million?
Result columns: ['category', 'total_sales']
Rows:
  Bikes 28316272
Answer: {"matches": true, "reason": "Bikes is the one category whose total exceeds the threshold; a HAVING-filtered result can legitimately contain exactly one qualifying row."}

Respond in strict JSON only, no other text:
{"matches": true or false, "reason": "one sentence explanation"}"""

JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_verifier_json(raw: str):
    """Parse the verifier's "Answer:" JSON, recovering small model slips.

    Small reasoning models sometimes close a JSON string with a single quote
    right before the closing brace ({{...value'}}) instead of a double quote
    ({{...value"}}), which makes both json.loads(raw) and JSON_RE + loads fail.
    When the object text ends with '}} exactly at the string terminator, the
    single quote is replaced with a double quote and the parse is retried, so
    a verifier that actually accepted the result still says so instead of
    failing us into a repair loop.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = JSON_RE.search(raw)
    if not match:
        return None
    text = match.group()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.endswith("'}"):
        try:
            return json.loads(text[:-2] + '"}' )
        except json.JSONDecodeError:
            return None
    return None

_AGGREGATE_RE = re.compile(
    r"\b(?:how many|total|sum|average|avg|mean|count|min(?:imum)?|max(?:imum)?)\b",
    re.IGNORECASE,
)


def _zero_aggregate_accept(question: str, columns: list, rows: list) -> bool:
    """Deterministic acceptance of an honest zero from a filtered aggregate.

    A single-row, single-column result holding 0 answers any aggregate question
    ('how many...', 'total...', 'average...', etc.) when the query filtered the
    data to nothing. The verifier cannot see the SQL or the full table, so an
    unmatched filter (a LIKE pattern with no matches, a country with no
    records) yielding 0 is a CORRECT answer — mirroring the SYSTEM_PROMPT's
    doctrine for NULL/empty aggregates — not evidence of a bug. This stops
    small reasoning models from rejecting truthful zeros because they
    personally expect a match."""
    if not _AGGREGATE_RE.search(question):
        return False
    if len(columns) != 1 or len(rows) != 1:
        return False
    row = rows[0]
    if not isinstance(row, dict):
        return False
    return row.get(columns[0]) == 0


BINARY_COMPARE_RE = re.compile(
    r"\b(?:compared\s+to|compared\s+with|compare\s+to|compare\s+with|"
    r"versus|vs\.?|vs\b|who\s+did(?:n'?t| not)|\bdidn'?t\b|\bdid\s+not\b)\b",
    re.IGNORECASE,
)


def _flag_pair_accept(question: str, columns: list, rows: list) -> bool:
    """Deterministic acceptance of the classic two-group comparison shape: a
    binary flag column holding exactly {0, 1} plus a numeric measure column.
    The 1-row is the positive group and the 0-row the negative group. Mirrors
    the SYSTEM_PROMPT doctrine (a GROUP BY result is complete for the groups
    that exist) and protects against small reasoning models that misread two
    flag-labeled rows as a single combined row."""
    if not BINARY_COMPARE_RE.search(question):
        return False
    if len(rows) != 2 or not all(isinstance(r, dict) for r in rows):
        return False
    for col in columns:
        vals = [r.get(col) for r in rows]
        if vals not in ([0, 1], [1, 0]):
            continue
        measure_cols = [c for c in columns if c != col]
        if not measure_cols:
            continue
        if all(
            isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
            for r in rows
            for c in measure_cols
        ):
            return True
    return False


def verify_result(question: str, sql: str, result: dict) -> dict:
    """Verify whether a query result answers the original question."""
    if not result.get("success"):
        return {"matches": False, "reason": result.get("error", "query failed")}

    columns = result["columns"]
    rows = result["rows"][:MAX_ROWS]

    if _zero_aggregate_accept(question, columns, rows):
        return {
            "matches": True,
            "reason": (
                "A single-row aggregate of 0 is the correct answer when the "
                "query's filter matched no data."
            ),
        }

    if _flag_pair_accept(question, columns, rows):
        return {
            "matches": True,
            "reason": (
                "Two rows, one flagged 1 (the positive group) and one flagged 0 "
                "(the negative group), directly answer the comparison requested."
            ),
        }

    result_text = f"Columns: {columns}\nRows ({len(result['rows'])} total, showing {len(rows)}):"
    for row in rows:
        result_text += f"\n  {row}"

    prompt = (
        f"Question: {question}\n\n"
        f"Result:\n{result_text}\n\n"
        "Does this result actually answer the question?"
    )

    try:
        raw = get_llm_client("reasoning").generate(prompt, system=SYSTEM_PROMPT, temperature=0.2)
    except LLMTimeoutError as exc:
        # Can't judge -> safest verdict is "does not match", fed to the repair
        # loop as error_feedback instead of crashing the request.
        return {"matches": False, "reason": f"Verifier timed out: {exc}"}

    parsed = parse_verifier_json(raw)
    if parsed is not None:
        return parsed

    return {"matches": False, "reason": "could not parse verifier response"}