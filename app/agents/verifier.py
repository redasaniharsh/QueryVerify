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

Respond in strict JSON only, no other text:
{"matches": true or false, "reason": "one sentence explanation"}"""

JSON_RE = re.compile(r"\{[^{}]*\}")


def verify_result(question: str, sql: str, result: dict) -> dict:
    """Verify whether a query result answers the original question."""
    if not result.get("success"):
        return {"matches": False, "reason": result.get("error", "query failed")}

    columns = result["columns"]
    rows = result["rows"][:MAX_ROWS]

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

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = JSON_RE.search(raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"matches": False, "reason": "could not parse verifier response"}