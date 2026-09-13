"""
Comprehensive core SQL property testing against SQL Server (mssql dialect).

Proves the system handles fundamental SQL building blocks on SQL Server, not
just business questions. Every answer is compared against the ground truth
computed independently with pandas directly on the source CSVs — a PASS means
the NUMBER is right, not merely that the query ran without error.

Run through BOTH datasets on SQL Server:
  * Sample sales data  -> database "QueryVerifyTest"   (fact_sales + dims)
  * Uploaded Titanic   -> database "QueryVerifyTitanic" (single 'titanic' table)

Usage:
    $env:QV_ENV_FILE=".env.mssql"
    py -3.11 scripts/test_sql_properties_mssql.py

The script refuses to run unless the active dialect is mssql.
"""

import re
import sys
import traceback
from decimal import Decimal
from time import perf_counter

import pandas as pd
from sqlalchemy import create_engine, text

from app.config import settings
from app.db.connection import DATA_DIR
import app.agents.orchestrator as orch

TITANIC_DB = "QueryVerifyTitanic"
TITANIC_CSV = r"C:\Users\Harsh\Downloads\Titanic-Dataset.csv"


# --------------------------------------------------------------------------
# Dialect confirmation — do NOT run any part of this on SQLite.
# --------------------------------------------------------------------------
def _confirm_dialect() -> tuple:
    if not settings.database_url.startswith("mssql"):
        print(
            f"ABORT: active DATABASE_URL is not mssql "
            f"({settings.database_url!r}). Set $env:QV_ENV_FILE='.env.mssql'."
        )
        sys.exit(2)

    sample_engine = create_engine(settings.database_url)
    titanic_url = settings.database_url.replace("/QueryVerifyTest?", f"/{TITANIC_DB}?")
    titanic_engine = create_engine(titanic_url)

    d1, d2 = sample_engine.dialect.name, titanic_engine.dialect.name
    if d1 != "mssql" or d2 != "mssql":
        print(f"ABORT: engine dialects are {d1!r} / {d2!r}, expected mssql.")
        sys.exit(3)

    with sample_engine.connect() as c:
        n = c.execute(text("SELECT COUNT(*) FROM fact_sales")).scalar()
    with titanic_engine.connect() as c:
        t = c.execute(text("SELECT COUNT(*) FROM titanic")).scalar()

    print("Dialect confirmed: mssql")
    print(f"  sample (QueryVerifyTest) fact_sales rows : {n}")
    print(f"  titanic (QueryVerifyTitanic) rows        : {t}")
    print(f"  config URL: {settings.database_url}")
    return sample_engine, titanic_engine


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
    """Take the one numeric value from the first result row that matches
    keyword column names."""
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
    """rows -> {label: float} using the label/value columns found by keyword."""
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
# Ground truth (pandas, straight off the CSVs)
# --------------------------------------------------------------------------
def load_ground_truth():
    sales = pd.read_csv(DATA_DIR / "fact_sales.csv")
    cust = pd.read_csv(DATA_DIR / "dim_customers.csv")
    prod = pd.read_csv(DATA_DIR / "dim_products.csv")
    t = pd.read_csv(TITANIC_CSV)

    s_cust = sales.merge(cust[["customer_key", "country"]], on="customer_key", how="left")
    s_prod = sales.merge(prod[["product_key", "category"]], on="product_key", how="left")

    s_idx = sales.merge(cust[["customer_key", "country"]], on="customer_key", how="left")
    order_tot = s_idx.groupby("order_number")["sales_amount"].sum()
    order_by_country = s_idx.assign(
        osum=s_idx.groupby("order_number")["sales_amount"].transform("sum")
    ).groupby(["country", "order_number"]).first()
    avg_by = (
        order_by_country.reset_index()
        .groupby("country")
        .agg(avg_order_value=("osum", "mean"), n_orders=("order_number", "count"))
        .reset_index()
    )
    avg_by = avg_by[avg_by["n_orders"] > 100].sort_values(
        "avg_order_value", ascending=False
    )

    dates = pd.to_datetime(sales["order_date"], errors="coerce")
    h1 = sales.loc[(dates >= "2013-01-01") & (dates <= "2013-06-30")]

    cust_totals = (
        s_idx.groupby("customer_key")["sales_amount"].sum().reset_index()
    )

    return {
        "sales": sales,
        "prod": prod,
        "t": t,
        "s_idx": s_idx,
        "s_prod": s_prod,
        "dates": dates,
        "h1_orders": int(h1["order_number"].nunique()),
        "h1_rows": int(len(h1)),
        "avg_by": avg_by.set_index("country")["avg_order_value"].round(4).to_dict(),
        "cust_totals_sum": float(sales["sales_amount"].sum()),
        "cust_totals_zero": int((cust_totals["sales_amount"] == 0).sum()),
        "n_customers": int(cust["customer_key"].nunique()),
    }


# --------------------------------------------------------------------------
# Question definitions: (category, dataset, question, checker)
# dataset: "sample" | "titanic"
# checker(rows, gt) -> (pass_bool, pipeline_value, ground_truth_value, note)
# --------------------------------------------------------------------------
def define_questions(gt):
    sales, prod, t = gt["sales"], gt["prod"], gt["t"]
    checks = []

    # ---------------- K. Aggregation -------------------------------
    def k_sample(rows, gt):
        vals = sorted(float(_num(v)) for r in rows for v in r.values() if _num(v) is not None)
        want = sorted([float(sales["sales_amount"].min()), float(sales["sales_amount"].max())])
        ok = len(vals) >= 2 and vals[0] == want[0] and vals[-1] == want[-1]
        return ok, (vals[0], vals[-1]) if vals else None, want, ""

    checks.append(("K. Aggregation (min/max)", "sample",
                   "What is the minimum and maximum sale amount?", k_sample))

    def k_titanic(rows, gt):
        vals = sorted(float(_num(v)) for r in rows for v in r.values() if _num(v) is not None)
        a = t["Age"].dropna()
        want = sorted([float(a.min()), float(a.max())])
        ok = len(vals) >= 2 and vals[0] == want[0] and vals[-1] == want[-1]
        return ok, (vals[0], vals[-1]) if vals else None, want, ""

    checks.append(("K. Aggregation (min/max)", "titanic",
                   "What is the oldest and youngest passenger age?", k_titanic))

    # ---------------- L. GROUP BY + HAVING -------------------------
    def l_sample(rows, gt):
        got = _rows_to_map(rows, ["category"], ["sales", "total", "amount"])
        cat = gt["s_prod"].groupby("category")["sales_amount"].sum()
        want = {str(k): float(v) for k, v in cat[cat > 1_000_000].items()}
        ok = got.keys() == want.keys() and all(_close(got[k], want[k]) for k in want)
        return ok, got, want, ""

    checks.append(("L. GROUP BY + HAVING", "sample",
                   "Which product categories have total sales over 1 million?", l_sample))

    def l_titanic(rows, gt):
        got = _rows_to_map(rows, ["class", "pclass"], ["count", "n", "passengers"])
        want = {str(k): int(v) for k, v in t.groupby("Pclass").size().items() if v > 200}
        ok = got.keys() == want.keys() and all(_close(got[k], want[k]) for k in want)
        return ok, got, want, ""

    checks.append(("L. GROUP BY + HAVING", "titanic",
                   "Which passenger classes have more than 200 passengers?", l_titanic))

    # ---------------- M. LEFT JOIN -----------------------------------
    def m_sample(rows, gt):
        n_rows = len(rows)
        n_cust = gt["n_customers"]
        val_col = _col(list(rows[0]), "total", "sales", "sum", "amount") if rows else None
        sums = [
            float(r[val_col]) for r in rows
            if val_col and val_col in r and _num(r[val_col]) is not None
        ]
        zeros = sum(1 for v in sums if v == 0.0)
        ok = (
            n_rows == n_cust
            and _close(sum(sums), gt["cust_totals_sum"], 0.01)
            and zeros == gt["cust_totals_zero"]
        )
        return ok, {"rows": n_rows, "total": round(sum(sums), 2), "zeros": zeros}, \
               {"rows": n_cust, "total": gt["cust_totals_sum"], "zeros": 0}, ""

    checks.append(("M. LEFT JOIN (incl. zero-sales)", "sample",
                   "List all customers and their total sales, including customers with zero sales.", m_sample))

    # ---------------- N. BETWEEN / IN / IS NULL ----------------------
    def n_sample(rows, gt):
        val = _scalar_row(rows, ["count", "number", "sales"])
        want = float(((sales["sales_amount"] >= 100) & (sales["sales_amount"] <= 500)).sum())
        return _close(val, want), val, want, ""

    checks.append(("N. Range filtering (BETWEEN)", "sample",
                   "How many sales were between 100 and 500?", n_sample))

    def n_titanic(rows, gt):
        val = _scalar_row(rows, ["count", "number", "passengers", "missing", "null"])
        want = float(t["Age"].isna().sum())
        return _close(val, want), val, want, ""

    checks.append(("N. NULL handling (IS NULL)", "titanic",
                   "How many passengers have a missing (null) age value?", n_titanic))

    # ---------------- O. Ranking ORDER BY + TOP ----------------------
    def o_titanic(rows, gt):
        top3 = t.sort_values(["Age", "PassengerId"], ascending=[False, True])[["PassengerId", "Age"]].head(3)
        want = [(int(r.PassengerId), float(r.Age)) for r in top3.itertuples()]
        got = []
        for r in rows:
            pid = None
            age = None
            for k, v in r.items():
                nv = _num(v)
                if nv is not None:
                    if _is_pid(k):
                        pid = int(nv)
                    elif age is None:
                        age = float(nv)
            if pid is not None and age is not None:
                got.append((pid, age))
        ok = sorted(got) == sorted(want)
        return ok, got, want, ""

    checks.append(("O. Ranking (ORDER BY + TOP, tie-break)", "titanic",
                   "Who are the 3 oldest passengers?", o_titanic))

    # ---------------- P. DISTINCT ------------------------------------
    def p_sample(rows, gt):
        val = _scalar_row(rows, ["category", "count", "distinct"])
        want = float(prod["category"].nunique())
        sold = float(gt["s_prod"]["category"].nunique())
        note = " (sensitivity: sold-only categories = %d)" % int(sold)
        return _close(val, want), val, want, note

    checks.append(("P. DISTINCT / de-dup", "sample",
                   "How many distinct product categories are there?", p_sample))

    def p_titanic(rows, gt):
        val = _scalar_row(rows, ["class", "count", "distinct", "ticket"])
        want = float(t["Pclass"].nunique())
        return _close(val, want), val, want, ""

    checks.append(("P. DISTINCT / de-dup", "titanic",
                   "How many distinct ticket classes are there?", p_titanic))

    # ---------------- Q. case-insensitive / partial text --------------
    def q_titanic(rows, gt):
        val = _scalar_row(rows, ["count", "name", "passengers", "john"])
        want = float(t["Name"].str.lower().str.contains("john", na=False).sum())
        return _close(val, want), val, want, ""

    checks.append(("Q. Case-insensitive/partial text", "titanic",
                   'How many passengers have a name containing "John"?', q_titanic))

    # ---------------- R. NULL vs zero vs empty string ------------------
    def r_titanic(rows, gt):
        val = _scalar_row(rows, ["cabin", "count", "passengers", "no"])
        want = float(t["Cabin"].isna().sum())
        ok = _close(val, want)
        return ok, val, want, "NULL-only dataset: 687 null cabins, 0 empty strings — answer must be 687, not 0"

    checks.append(("R. NULL vs zero vs empty string", "titanic",
                   "How many passengers have no cabin recorded?", r_titanic))

    # ---------------- S. date range -------------------------------------
    def s_sample(rows, gt):
        val = _scalar_row(rows, ["order", "count", "date", "first"])
        note = f" (sensitivity: line-item rows in window = {gt['h1_rows']})"
        return _close(val, gt["h1_orders"]), val, gt["h1_orders"], note

    checks.append(("S. Date/time range filtering", "sample",
                   "How many orders were placed in the first half of 2013?", s_sample))

    # ---------------- T. combined stress ---------------------------------
    def t_sample(rows, gt):
        got = _rows_to_map(rows, ["country", "region"], ["avg", "order", "value", "average"])
        want = {str(k): float(v) for k, v in gt["avg_by"].items()}
        ok = got.keys() == want.keys() and all(_close(got[k], want[k], 1.0) for k in want)
        return ok, got, want, "6 countries expected (Australia, Germany, UK, France, US, Canada)"

    checks.append(("T. Combined stress (agg+having+sort)", "sample",
                   "What is the average order value, grouped by country, only for countries with more than 100 orders, sorted from highest to lowest?", t_sample))

    return checks


def _is_pid(k):
    k = k.lower()
    return "passenger" in k or "id" == k or k.endswith("_id")


# --------------------------------------------------------------------------
# SQLite-syntax regression flags
# --------------------------------------------------------------------------
def _sql_flags(sql: str) -> list[str]:
    flags = []
    flags.append("LIMIT (SQLite-style; must be TOP on SQL Server)") if re.search(
        r"\bLIMIT\b", sql, re.IGNORECASE
    ) else None
    flags.append("SQLite DATE() function (does not exist on SQL Server)") if re.search(
        r"\bDATE\s*\(", sql, re.IGNORECASE
    ) else None
    flags.append("backtick identifiers (SQLite-style)") if "`" in sql else None
    flags.append("SQLite || concatenation used") if re.search(r"\|\|", sql) else None
    return flags


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    sample_engine, titanic_engine = _confirm_dialect()
    engines = {"sample": sample_engine, "titanic": titanic_engine}
    gt = load_ground_truth()

    print("\n" + "=" * 84)
    print("CORE SQL PROPERTY TESTS — SQL SERVER (mssql)")
    print("Ground truth: pandas on source CSVs. PASS means the NUMBER matches.")
    print("=" * 84)

    rows_log = []
    category_stats = {}

    for category, dataset, question, checker in define_questions(gt):
        rows_log.append(f"\n### {category} [{dataset}]")
        t0 = perf_counter()
        try:
            r = orch.handle_question(question, engines[dataset])
            elapsed = perf_counter() - t0
            err = None
        except Exception as exc:
            r, elapsed, err = {}, perf_counter() - t0, f"{type(exc).__name__}: {exc}"

        sql = (r.get("sql") or "").strip()
        result = r.get("result") or {}
        rows = result.get("rows") or []
        success = bool(r.get("success"))
        flag = bool(r.get("needs_clarification"))

        flags = _sql_flags(sql)
        syntax_ok = not flags

        # value_correct: does the pipeline's returned data match pandas ground
        # truth (independent of whether the pipeline judged itself confident)?
        value_correct = False
        checker_note = ""
        try:
            sub_ok, sub_val, sub_want, sub_note = checker(rows, gt)
            value_correct = bool(sub_ok)
            checker_note = sub_note or ""
        except Exception as exc:
            sub_val, sub_want = None, None
            checker_note = f"checker exception: {exc}"

        delivered = success and not flag and err is None
        if err:
            ok, status_note = False, f"exception {err}"
        elif flag:
            ok = False
            status_note = f"flagged: {r.get('clarifying_question', '')[:140]}"
        elif not success:
            ok = False
            if value_correct and rows:
                status_note = (
                    "pipeline returned NO confirmed answer, but SQL + values match "
                    "ground truth (verifier rejected the valid result)"
                )
            else:
                status_note = f"pipeline failed: {(r.get('explanation') or '')[:160]}"
        else:
            ok = value_correct
            status_note = ""

        if delivered and not flags:
            status_note += " | mssql syntax OK"
        if delivered and flags:
            status_note += " | SYNTAX: " + "; ".join(flags)

        if flag:
            pipe_val, want_val = None, sub_want
        elif err:
            pipe_val, want_val = None, None
        elif value_correct:
            pipe_val, want_val = sub_val, sub_want
        else:
            pipe_val, want_val = sub_val, sub_want

        note = (status_note + (" | " + checker_note if checker_note else "")).strip(" |")

        status = "PASS" if ok else "FAIL"
        print(f"\n{status} | {category} [{dataset}]")
        print(f"Q: {question}")
        if sql:
            cur = 0
            while cur < len(sql):
                piece = sql[cur:cur + 120]
                print(f"   SQL: {piece}")
                cur += 120
        print(f"   pipeline answer: {pipe_val}")
        print(f"   ground truth   : {want_val}")
        print(f"   confidence={r.get('confidence')} attempts={r.get('attempts')} rows={len(rows)} elapsed={elapsed:.1f}s")
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

        key = category.split(" (")[0]
        s = category_stats.setdefault(key, [0, 0])
        s[1] += 1
        s[0] += 1 if ok else 0

    print("\n" + "=" * 84)
    print("CATEGORY-BY-CATEGORY PASS RATE (SQL Server)")
    total_p = total_n = 0
    for key, (p, n) in category_stats.items():
        print(f"  {key:<42} {p}/{n}")
        total_p += p
        total_n += n
    print("-" * 84)
    print(f"  TOTAL {total_p}/{total_n}")

    with open("sql_properties_mssql.log", "w", encoding="utf-8") as f:
        f.write("\n".join(rows_log))
    print("\nfull trace written to sql_properties_mssql.log")


if __name__ == "__main__":
    main()