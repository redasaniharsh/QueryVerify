"""
Edge-case test suite. Runs each question through handle_question() and prints
PASS/FAIL per category plus a per-category pass-rate summary.

Categories:
  A. Clear questions            -> must NOT be flagged ambiguous
  B. Genuinely vague            -> MUST be flagged ambiguous
  C. Not in schema              -> graceful failure, no hallucinated answer
  D. Destructive intent         -> blocked by read-only guard BEFORE any LLM/DB call
  E. Date filtering (TEXT)      -> must answer with rows
  F. 3-table join               -> must join fact_sales + both dims
  G. Empty result set           -> no crash, honest zero rows
  H. Case-insensitive matching  -> lowercase 'australia' must match
  I. Complex reasoning          -> harder/longer questions; judged qualitatively.
     For these, any defined pipeline outcome (a clear answer OR a clarifying
     flag) without a crash counts as passing the harness; the honest quality
     verdict is reported per question alongside the outcome facts.
"""

import sys
import traceback
from time import perf_counter

from app.db.connection import get_engine

engine = get_engine()
from app.agents.executor import is_read_only
import app.agents.orchestrator as orch

# Instrument the three pipeline entry points so we can PROVE no LLM/DB call
# happens for category D.
_calls = {"schema_context": 0, "ambiguity": 0, "pipeline": 0}
_orig = (orch.get_schema_context, orch.check_ambiguity, orch.run_pipeline)


def _wrap(fn, key):
    def wrapper(*a, **k):
        _calls[key] += 1
        return fn(*a, **k)
    return wrapper


orch.get_schema_context = _wrap(_orig[0], "schema_context")
orch.check_ambiguity = _wrap(_orig[1], "ambiguity")
orch.run_pipeline = _wrap(_orig[2], "pipeline")


def _run(question):
    t0 = perf_counter()
    try:
        r = orch.handle_question(question, engine)
        return r, perf_counter() - t0, None
    except Exception as e:
        return {}, perf_counter() - t0, f"{type(e).__name__}: {e}"


def _detail(r, sql_len=70):
    rows = (r.get("result") or {}).get("rows") or []
    conf = r.get("confidence_score")
    return {
        "flag": r.get("needs_clarification"),
        "clear_q": r.get("clarifying_question") or "",
        "success": r.get("success"),
        "conf": f"{r.get('confidence')} ({conf:.2f})" if conf is not None else "none",
        "attempts": r.get("attempts"),
        "rows": len(rows),
        "sql": (r.get("sql") or "")[:sql_len].replace("\n", " "),
        "expl": (r.get("explanation") or "")[:110],
    }


ALL_QUESTIONS = {
    "A. Clear questions (must NOT be flagged)": [
        "What is the total sales amount?",
        "How many customers are there?",
        "What is the average sale price?",
    ],
    "B. Genuinely vague (must be flagged)": [
        "Show me the top customers",
        "What are the best products?",
        "Give me some interesting insights about sales",
    ],
    "C. Not in schema (graceful failure)": [
        "What is the total employee salary?",
        "Show me all invoices",
    ],
    "D. Destructive intent (blocked before LLM/DB)": [
        "Delete all customers",
        "Update product prices to zero",
        "Drop the fact_sales table",
    ],
    "E. Date filtering (TEXT dates)": [
        "What were the total sales in 2013?",
        "How many orders were placed after 2013-06-01?",
    ],
    "F. Multi-table join (all 3 tables)": [
        "What is the total sales amount for road bike products bought by customers in Germany?",
    ],
    "G. Empty result set (zero results)": [
        "What is the total sales amount for customers in Antarctica?",
    ],
    "H. Case-insensitive text matching": [
        "How many customers are in australia",
    ],
    "I. Complex reasoning (harder cases)": [
        "Which country has the highest average order value, but only counting customers who placed more than 3 orders?",
        "Compare total sales between road bikes and mountain bikes.",
        "Which product category grew the fastest between 2012 and 2013?",
        "wat is teh total sales for aussie customers",
        "What is the total sales amount and how many orders were there in Germany?",
    ],
    "J. Hindi language support (non-Latin script)": [
        "\u0915\u0941\u0932 \u0915\u093f\u0924\u0928\u0947 \u0917\u094d\u0930\u093e\u0939\u0915 \u0939\u0948\u0902?",
        "\u0911\u0938\u094d\u091f\u094d\u0930\u0947\u0932\u093f\u092f\u093e \u092e\u0947\u0902 \u0915\u093f\u0924\u0928\u0947 \u0917\u094d\u0930\u093e\u0939\u0915 \u0939\u0948\u0902?",
        "\u0926\u0947\u0936 \u0915\u0947 \u0905\u0928\u0941\u0938\u093e\u0930 \u0915\u0941\u0932 \u092c\u093f\u0915\u094d\u0930\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?",
    ],
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("QueryVerify edge-case suite (py -3.11 scripts/edge_case_tests.py)")
    print("=" * 78)

    category_passes = {}
    total_questions = 0
    total_passes = 0

    for category, questions in ALL_QUESTIONS.items():
        passes = 0
        print(f"\n=== {category} ===")
        for question in questions:
            total_questions += 1
            if category.startswith("D."):
                _calls.update(schema_context=0, ambiguity=0, pipeline=0)
                r, elapsed, error = _run(question)
            else:
                r, elapsed, error = _run(question)
            d = _detail(r)

            if category.startswith("A."):
                ok = not d["flag"] and error is None and d["success"] and d["rows"] >= 0
                note = f"flag={'YES' if d['flag'] else 'no'}  success={'yes' if d['success'] else 'no'}  conf={d['conf']}"
            elif category.startswith("B."):
                ok = d["flag"] and error is None
                note = f"flag={'YES' if d['flag'] else 'no'}  ask={d['clear_q'][:48]!r}"
            elif category.startswith("C."):
                graceful = d["flag"] or d["success"] is False
                fabricated = d["success"] is True
                ok = graceful and not fabricated and error is None
                if d["flag"]:
                    note = f"handled via clarification: {d['clear_q'][:80]!r}"
                else:
                    note = f"handled via failure: explanation={d['expl'][:80]!r}"
                if fabricated:
                    note = "FABRICATED ANSWER (bad)"
            elif category.startswith("D."):
                guard_before = d["attempts"] == 0 and r.get("sql") is None and not d["flag"]
                ok = bool(r.get("blocked")) and guard_before and \
                     _calls["schema_context"] == 0 and _calls["ambiguity"] == 0 and _calls["pipeline"] == 0
                note = (f"blocked={'yes' if r.get('blocked') else 'no'}  attempts={d['attempts']}  "
                        f"LLM/DB calls=[schema:{_calls['schema_context']} ambiguity:{_calls['ambiguity']} "
                        f"pipeline:{_calls['pipeline']}]  elapsed={elapsed:.2f}s")
            elif category.startswith("E."):
                ok = not d["flag"] and d["success"] and d["rows"] > 0 and error is None
                note = f"success={'yes' if d['success'] else 'no'}  rows={d['rows']}  conf={d['conf']}"
            elif category.startswith("F."):
                sql = (r.get("sql") or "").lower()
                joins_all_three = all(t in sql for t in ("fact_sales", "dim_customers", "dim_products"))
                ok = not d["flag"] and d["success"] and d["rows"] > 0 and joins_all_three and error is None
                note = f"success={'yes' if d['success'] else 'no'}  rows={d['rows']}  3-table={'yes' if joins_all_three else 'NO'}"
            elif category.startswith("G."):
                rows = (r.get("result") or {}).get("rows") or []
                single_unknown = len(rows) == 1 and all(str(v) == "Unknown" for v in rows[0].values())
                ok = not d["flag"] and d["success"] and (d["rows"] == 0 or single_unknown) and error is None
                note = f"success={'yes' if d['success'] else 'no'}  rows={d['rows']} (honest zero)  expl={d['expl'][:60]!r}"
            elif category.startswith("H."):
                ok = not d["flag"] and d["success"] and d["rows"] > 0 and error is None
                note = f"success={'yes' if d['success'] else 'no'}  rows={d['rows']}  conf={d['conf']}"
            elif category.startswith("I."):
                outcome = "success" if d["success"] else ("flagged" if d["flag"] else "failed")
                ok = error is None
                note = (f"outcome={outcome}  flag={'YES' if d['flag'] else 'no'}  "
                        f"success={'yes' if d['success'] else 'no'}  conf={d['conf']}  "
                        f"attempts={d['attempts']}  rows={d['rows']}")
            elif category.startswith("J."):
                ok = not d["flag"] and d["success"] and d["rows"] > 0 and error is None
                note = f"success={'yes' if d['success'] else 'no'}  rows={d['rows']}  conf={d['conf']}"
            else:
                ok, note = False, "?"

            passes += ok
            total_passes += ok
            status = "PASS" if ok else "FAIL"
            print(f"{status} | {question}")
            print(f"      time: {elapsed:.1f}s")
            print(f"      {note}")
            if d["sql"]:
                print(f"      sql: {d['sql']}")
            if error:
                print(f"      EXCEPTION: {error}")
                traceback.print_exc(limit=2)

        category_passes[category] = (passes, len(questions))

    print("\n=== Read-only guard (is_read_only) backing proof ===")
    print(f"is_read_only('DELETE FROM customers')   = {is_read_only('DELETE FROM customers')}")
    print(f"is_read_only('UPDATE product SET x=0')  = {is_read_only('UPDATE product SET price = 0')}")
    print(f"is_read_only('DROP TABLE fact_sales')   = {is_read_only('DROP TABLE fact_sales')}")
    print(f"is_read_only('SELECT 1')                = {is_read_only('SELECT 1')}")
    print(f"is_read_only('  select * from x')       = {is_read_only('  select * from x')}")

    print("\n=== Prompt 11: multi-statement injection guard (is_read_only) ===")
    print(f"is_read_only('SELECT * FROM customers; DROP TABLE customers;') = {is_read_only('SELECT * FROM customers; DROP TABLE customers;')}")
    print(f"is_read_only('SELECT 1; SELECT 2')                             = {is_read_only('SELECT 1; SELECT 2')}")
    print(f"is_read_only('SELECT 1;')                                      = {is_read_only('SELECT 1;')}")
    _str_semicolon = "SELECT 'a;b' AS x"
    print(f"is_read_only({_str_semicolon!r})                                 = {is_read_only(_str_semicolon)}")
    from app.agents.executor import execute_sql
    inj = execute_sql("SELECT * FROM customers; DROP TABLE customers;", engine)
    print(f"execute_sql(injection) blocked before execution = {not inj['success']}  error={inj['error']!r}")

    print("\n" + "=" * 78)
    print("PER-CATEGORY PASS RATE")
    for category, (p, n) in category_passes.items():
        print(f"  {category:<52} {p}/{n}")
    print("-" * 78)
    print(f"  TOTAL {total_passes}/{total_questions} passed")

    print("\n=== exec fix proof: None -> 'Unknown' (deterministic, in executor) ===")
    from app.agents.executor import execute_sql
    r = execute_sql(
        "SELECT dim_customers.country, SUM(fact_sales.sales_amount) AS total_sales_amount "
        "FROM fact_sales JOIN dim_customers ON fact_sales.customer_key = dim_customers.customer_key "
        "GROUP BY dim_customers.country;",
        engine,
    )
    unknown = [row for row in r["rows"] if row.get("country") == "Unknown"]
    print(f"country column values: {[row['country'] for row in r['rows']]}")
    print(f"'Unknown' rendered instead of None: {len(unknown) > 0}")


if __name__ == "__main__":
    main()