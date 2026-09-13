"""
Titanic 10-question suite — SQL Server (mssql) against the uploaded "titanic"
table in the QueryVerifyTitanic database.

Every answer is checked against the ground truth computed independently with
pandas straight off the source CSV. PASS means the NUMBER (or the expected
flag/decline behavior) is right — not merely that the query ran.

Run:
    $env:QV_ENV_FILE=".env.mssql"
    py -3.11 scripts/titanic_case_tests.py

The script refuses to run unless the active dialect is mssql.

Question list (in order):
   1. How many passengers are there?                              -> 891
   2. How many passengers survived?                               -> 342
   3. What is the average age of passengers?                      -> ~29.70
   4. How many male passengers survived?                          -> 109
   5. What is the survival rate by passenger class?               -> ~0.63/0.47/0.24
   6. What is the average fare paid by passengers who survived
      compared to those who didn't?                               -> ~48.40 vs ~22.12
   7. Which embarkation port had the highest survival rate?       -> C (~0.5536)
   8. Show me the top passengers        (MUST still be flagged)
   9. What is the total sales amount in this dataset?   (declined — no sales data)
  10. How many customers are there in Germany?          (declined — no customer data)
"""

import sys
import traceback
from decimal import Decimal
from time import perf_counter

import pandas as pd
from sqlalchemy import create_engine, text

from app.config import settings
import app.agents.orchestrator as orch

TITANIC_DB = "QueryVerifyTitanic"
TITANIC_CSV = r"C:\Users\Harsh\Downloads\Titanic-Dataset.csv"


def _confirm_dialect() -> object:
    if not settings.database_url.startswith("mssql"):
        print(
            f"ABORT: active DATABASE_URL is not mssql "
            f"({settings.database_url!r}). Set $env:QV_ENV_FILE='.env.mssql'."
        )
        sys.exit(2)

    titanic_url = settings.database_url.replace("/QueryVerifyTest?", f"/{TITANIC_DB}?")
    engine = create_engine(titanic_url)
    if engine.dialect.name != "mssql":
        print(f"ABORT: engine dialect is {engine.dialect.name!r}, expected mssql.")
        sys.exit(3)

    with engine.connect() as c:
        n = c.execute(text("SELECT COUNT(*) FROM titanic")).scalar()
    print("Dialect confirmed: mssql (queryverify titanic)")
    print(f"  titanic (QueryVerifyTitanic) rows : {n}")
    return engine


# --------------------------------------------------------------------------
# Result extraction helpers
# --------------------------------------------------------------------------
def _num(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    return None


def _col(columns, *keywords):
    for kw in keywords:
        for c in columns:
            if kw.lower() in c.lower():
                return c
    return None


def _scalar_row(rows, keywords):
    if not rows:
        return None
    row = rows[0]
    vals = [v for v in (row[c] for c in row if _num(row[c]) is not None) if v is not None]
    target = _col(list(row), *keywords)
    if target is not None and _num(row[target]) is not None:
        return float(_num(row[target]))
    if len(vals) == 1:
        return float(_num(vals[0]))
    return None


def _rows_to_map(rows, label_kw, value_kw):
    if not rows:
        return {}
    label_col = _col(list(rows[0]), *label_kw)
    value_col = _col(list(rows[0]), *value_kw)
    out = {}
    for r in rows:
        label = r[label_col] if label_col else None
        val = _num(r.get(value_col)) if value_col else None
        if val is None:
            continue
        out[str(label)] = float(val)
    return out


def _close(a, b, tol=0.5):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Ground truth (pandas, straight off the CSV)
# --------------------------------------------------------------------------
def load_ground_truth():
    t = pd.read_csv(TITANIC_CSV)
    emb = t.dropna(subset=["Embarked"])
    return {
        "t": t,
        "n_passengers": int(len(t)),
        "n_survived": int(t["Survived"].sum()),
        "avg_age": float(t["Age"].mean()),
        "male_survived": int(((t["Sex"] == "male") & (t["Survived"] == 1)).sum()),
        "rate_by_pclass": {
            str(k): float(v) for k, v in t.groupby("Pclass")["Survived"].mean().items()
        },
        "fare_by_survived": {
            str(k): float(v) for k, v in t.groupby("Survived")["Fare"].mean().items()
        },
        "rate_by_embarked": {
            str(k): float(v) for k, v in emb.groupby("Embarked")["Survived"].mean().items()
        },
        "top_port": emb.groupby("Embarked")["Survived"].mean().idxmax(),
    }


# --------------------------------------------------------------------------
# Questions: (category, question, expect_flag, checker)
# checker(rows, gt) -> (pass_bool, pipeline_value, ground_truth_value, note)
# For expect_flag questions the checker is unused; behavior is asserted.
# --------------------------------------------------------------------------
def define_questions(gt):
    checks = []

    def q1(rows, gt):
        val = _scalar_row(rows, ["passengers", "count", "total", "people"])
        want = float(gt["n_passengers"])
        return _close(val, want), val, want, ""

    checks.append(("count passengers", "How many passengers are there?", False, q1))

    def q2(rows, gt):
        val = _scalar_row(rows, ["surviv", "sum", "count", "total"])
        want = float(gt["n_survived"])
        return _close(val, want), val, want, ""

    checks.append(("count survived", "How many passengers survived?", False, q2))

    def q3(rows, gt):
        val = _scalar_row(rows, ["age", "avg", "mean", "average"])
        want = round(gt["avg_age"], 4)
        return _close(val, want, 0.05), val, want, ""

    checks.append(("average age", "What is the average age of passengers?", False, q3))

    def q4(rows, gt):
        val = _scalar_row(rows, ["male", "surviv", "count"])
        want = float(gt["male_survived"])
        return _close(val, want), val, want, ""

    checks.append(("male survivors", "How many male passengers survived?", False, q4))

    def q5(rows, gt):
        got = _rows_to_map(rows, ["class", "pclass"], ["rate", "survival", "survive", "avg", "mean"])
        want = gt["rate_by_pclass"]
        ok = got.keys() == want.keys() and all(
            _close(got[k], want[k], 0.01) for k in want
        )
        return ok, got, want, "rates ~0.63 / 0.47 / 0.24 for classes 1/2/3"

    checks.append(
        ("survival rate by class", "What is the survival rate by passenger class?", False, q5)
    )

    def q6(rows, gt):
        want = gt["fare_by_survived"]  # {"0": ~22.12, "1": ~48.40}
        got_map = _rows_to_map(rows, ["survived", "survivor", "group", "outcome"], ["fare", "avg", "average", "mean"])
        got = None
        if set(got_map) == set(want) and got_map and all(
            _close(got_map[k], want[k], 2.0) for k in want
        ):
            return True, got_map, want, "answered directly (not flagged)"
        vals = set()
        for r in rows:
            for v in r.values():
                nv = _num(v)
                if nv is not None:
                    vals.add(float(nv))
        ok = len(vals) >= 2 and all(any(abs(round(v, 2) - round(w, 2)) <= 2.0 for v in vals) for w in want.values())
        note = "answered directly (not flagged)" if ok else "could not verify both group fares"
        return ok, got_map or sorted(vals), want, note

    checks.append(
        ("avg fare survived vs not",
         "What is the average fare paid by passengers who survived compared to those who didn't?",
         False, q6)
    )

    def q7(rows, gt):
        got = _rows_to_map(rows, ["embarked", "port"], ["rate", "survival", "survive", "avg", "mean"])
        want_top = gt["top_port"]
        want_rate = gt["rate_by_embarked"][want_top]
        if not got:
            return False, got, want_top, "no rows"
        best = max(got, key=lambda k: got[k])
        ok = best == want_top and _close(got[best], want_rate, 0.02)
        note = f"answered directly (not flagged); top port={best} rate={got[best]:.4f}"
        return ok, got, {"top": want_top, "rate": want_rate}, note

    checks.append(
        ("highest survival rate by port",
         "Which embarkation port had the highest survival rate?",
         False, q7)
    )

    # Behavior checks (expect_flag=True): must be flagged/declined, with NO
    # hallucinated SQL result.
    checks.append(
        ("top passengers must STILL be flagged",
         "Show me the top passengers", True, None)
    )
    checks.append(
        ("no-sales question must be declined",
         "What is the total sales amount in this dataset?", True, None)
    )
    checks.append(
        ("no-customers question must be declined",
         "How many customers are there in Germany?", True, None)
    )

    return checks


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    engine = _confirm_dialect()
    gt = load_ground_truth()

    print("\n" + "=" * 84)
    print("TITANIC 10-QUESTION SUITE — SQL SERVER (mssql, uploaded 'titanic' table)")
    print("Ground truth: pandas on the Titanic CSV. PASS means the NUMBER matches.")
    print("=" * 84)

    rows_log = []
    n_pass = n_total = 0

    for category, question, expect_flag, checker in define_questions(gt):
        rows_log.append(f"\n### {category}")
        t0 = perf_counter()
        try:
            r = orch.handle_question(question, engine)
            elapsed = perf_counter() - t0
            err = None
        except Exception as exc:
            r, elapsed, err = {}, perf_counter() - t0, f"{type(exc).__name__}: {exc}"

        sql = (r.get("sql") or "").strip()
        result = r.get("result") or {}
        rows = result.get("rows") or []
        flag = bool(r.get("needs_clarification"))
        success = bool(r.get("success"))

        if err:
            ok, pipe_val, want_val = False, None, None
            note = f"exception {err}"
        elif expect_flag:
            ok = flag and not rows and not sql
            reason = (r.get("clarifying_question") or r.get("explanation") or "")[:150]
            pipe_val = reason
            want_val = "flagged/declined without any hallucinated SQL result"
            note = (
                f"correctly flagged/declined: {reason}"
                if ok
                else "NOT flagged/declined (regression)"
            )
        elif flag:
            ok = False
            pipe_val = (r.get("clarifying_question") or "")[:150]
            want_val = "should have answered directly"
            note = f"STILL FLAGGED (should answer now): {pipe_val}"
        else:
            sub_ok, sub_val, sub_want, sub_note = checker(rows, gt)
            ok = bool(sub_ok) and success
            pipe_val, want_val, note = sub_val, sub_want, sub_note or ""

        status = "PASS" if ok else "FAIL"
        print(f"\n{status} | {category}")
        print(f"Q: {question}")
        if sql:
            cur = 0
            while cur < len(sql):
                print(f"   SQL: {sql[cur:cur + 120]}")
                cur += 120
        print(f"   pipeline answer: {pipe_val}")
        print(f"   ground truth    : {want_val}")
        print(f"   confidence={r.get('confidence')} attempts={r.get('attempts')} rows={len(rows)} elapsed={elapsed:.1f}s flag={flag}")
        if note:
            print(f"   note: {note}")
        if not ok:
            print(f"   rows: {rows[:5]}")

        rows_log.append(f"STATUS: {status} | {question}")
        rows_log.append(f"SQL: {sql}")
        rows_log.append(f"PIPELINE: {pipe_val}")
        rows_log.append(f"GROUND TRUTH: {want_val}")
        rows_log.append(f"NOTE: {note}")
        rows_log.append(f"conf={r.get('confidence')} attempts={r.get('attempts')} rows={len(rows)} err={err}")

        n_total += 1
        n_pass += 1 if ok else 0

    print("\n" + "=" * 84)
    print(f"  TOTAL {n_pass}/{n_total}")
    print("=" * 84)

    with open("titanic_case_tests.log", "w", encoding="utf-8") as f:
        f.write("\n".join(rows_log))
    print("\nfull trace written to titanic_case_tests.log")


if __name__ == "__main__":
    main()