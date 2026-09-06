"""
Deterministic safety-guard tests. No LLM, no Ollama, no database required.

These are the checks that can run on GitHub's runners (no GPU, no local models):
pure pattern matching and single-statement SQL parsing that decide what the
pipeline is ALLOWED to do before any model is ever consulted.

What's covered:
  * SQL read-only / multi-statement guard   -- app.agents.executor
      is_read_only() and the executor rejection path (injection like
      "SELECT ...; DROP TABLE ..." is caught before any DB access).
  * Destructive-intent guard                -- app.agents.orchestrator
      has_destructive_intent(): "delete/update/drop/..." blocked pre-LLM.
  * Absent-entity guard                     -- app.agents.orchestrator
      absent_entity(): names a topic that is not in this schema, so no SQL is
      silently fabricated around it.
  * 'by X' breakdown row-count guard        -- app.agents.repair_loop
      _breakdown_suspicion(): a full-breakdown question (by country/category/...)
      whose result is suspiciously short of the dimension's group count is the
      accidental-LIMIT blind spot; _check_consistency() caps confidence at medium
      when such a result reaches the self-consistency step.
  * Syntax / import blow-up check           -- py_compile of app/ and frontend/,
      plus importing app.main and the guard modules (catches syntax errors and
      broken imports immediately on every push).

These tests were extracted from scripts/edge_case_tests.py (the deterministic
sections). The LLM-dependent end-to-end suite remains there and is run manually
with a local Ollama instance.
"""

import importlib
import pathlib

import pytest

from app.agents.executor import is_read_only, execute_sql
from app.agents.orchestrator import has_destructive_intent, absent_entity
from app.agents import repair_loop

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# SQL read-only / multi-statement guard (app.agents.executor)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE product SET price = 0",
        "DROP TABLE fact_sales",
        "INSERT INTO customers VALUES (1, 'x')",
        "TRUNCATE TABLE fact_sales",
        "ALTER TABLE fact_sales ADD COLUMN x INTEGER",
        "WITH cte AS (SELECT 1) SELECT * FROM cte",
        "SELECT * FROM customers; DROP TABLE customers;",
        "SELECT 1; SELECT 2",
    ],
)
def test_is_read_only_rejects_non_single_select(sql):
    assert is_read_only(sql) is False


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "  select * from x",
        "SELECT COUNT(customer_id) FROM dim_customers",
        "SELECT 'a;b' AS x",
        "SELECT 1;",
    ],
)
def test_is_read_only_accepts_single_select(sql):
    assert is_read_only(sql) is True


def test_is_read_only_rejects_empty_query():
    assert is_read_only("") is False


def test_execute_sql_rejects_injection_before_touching_db():
    # The rejection short-circuits before any engine work, so this is DB-free.
    # engine=None proves the guard fires without a database being reachable.
    res = execute_sql("SELECT * FROM customers; DROP TABLE customers;", engine=None)
    assert res["success"] is False
    assert "single SELECT" in res["error"]


def test_execute_sql_rejects_non_select_statement():
    res = execute_sql("DELETE FROM customers", engine=None)
    assert res["success"] is False
    assert "SELECT" in res["error"]


# --------------------------------------------------------------------------
# Destructive-intent guard (app.agents.orchestrator)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "Delete all customers",
        "Update product prices to zero",
        "Drop the fact_sales table",
        "Please remove all sales orders",
        "Insert a new customer row",
        "Wipe the sales data",
        "Alter the customers table",
        "Truncate the fact_sales table",
        "erase every record",
    ],
)
def test_has_destructive_intent_detects_attacks(question):
    assert has_destructive_intent(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is the total sales amount?",
        "How many customers are there?",
        "What is the average sale price?",
        "How many orders were placed after 2013-06-01?",
        "Which country has the highest average order value?",
        "",
        None,
    ],
)
def test_has_destructive_intent_ignores_benign_questions(question):
    assert has_destructive_intent(question) is False


# --------------------------------------------------------------------------
# Absent-entity guard (app.agents.orchestrator)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the total employee salary?", "employee"),
        ("Show me all invoices", "invoices"),
        ("Do we have any suppliers?", "suppliers"),
        ("Show me all refunds", "refunds"),
        ("Tell me about campaigns", "campaigns"),
        ("Give me the inventory levels", "inventory"),
        ("What is the payroll total?", "payroll"),
    ],
)
def test_absent_entity_detects_non_schema_topics(question, expected):
    assert absent_entity(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "How many customers are there?",
        "What is the total sales amount?",
        "How many orders were placed after 2013-06-01?",
        "What are the best road bike products in Germany?",
        "",
        None,
    ],
)
def test_absent_entity_ignores_schema_topics(question):
    assert absent_entity(question) is None


# --------------------------------------------------------------------------
# 'by X' breakdown row-count guard (app.agents.repair_loop)
# --------------------------------------------------------------------------
# DB-free: _expected_group_count is monkeypatched so no engine is touched.

_TRUE_COUNTS = {"country": 7, "category": 5, "subcategory": 37, "product_line": 5}


def _patch_counts(monkeypatch):
    monkeypatch.setattr(repair_loop, "_expected_group_count", lambda engine, col: _TRUE_COUNTS.get(col))


def test_breakdown_suspicion_flags_single_row_by_country(monkeypatch):
    _patch_counts(monkeypatch)
    assert repair_loop._breakdown_suspicion(
        "What is the total sales amount by country?", 1, engine=None
    ) == ("country", 7)


def test_breakdown_suspicion_silent_when_all_groups_present(monkeypatch):
    _patch_counts(monkeypatch)
    assert repair_loop._breakdown_suspicion(
        "What is the total sales amount by country?", 7, engine=None
    ) is None


@pytest.mark.parametrize(
    "question",
    [
        "What is the total sales amount by category?",
        "Show sales per product subcategory",
        "Total revenue for each country",
        "What is the average order value grouped by product line?",
    ],
)
def test_breakdown_suspicion_detects_breakdown_phrasing(monkeypatch, question):
    _patch_counts(monkeypatch)
    assert repair_loop._breakdown_suspicion(question, 1, engine=None) is not None


@pytest.mark.parametrize(
    "question",
    [
        # 'by customers in Germany' is a join filter, not a country breakdown.
        "What is the total sales amount for road bike products bought by customers in Germany?",
        "Which product category grew the fastest between 2012 and 2013?",
        "What were the total sales in 2013?",
        "Show me the top customers",
        "Compare total sales between road bikes and mountain bikes.",
        "What is the total employee salary?",
        "How many customers are in australia",
    ],
)
def test_breakdown_suspicion_silent_without_breakdown_phrasing(monkeypatch, question):
    _patch_counts(monkeypatch)
    assert repair_loop._breakdown_suspicion(question, 1, engine=None) is None


def test_consistency_does_not_execute_sql_when_nothing_to_compare(monkeypatch):
    # A question with no breakdown phrasing at all must not reach the guard.
    _patch_counts(monkeypatch)
    assert repair_loop._breakdown_suspicion("How many customers are there?", 1, engine=None) is None


# --------------------------------------------------------------------------
# Consistency cap when a suspiciously short breakdown agrees anyway
# --------------------------------------------------------------------------

_LIMIT1_SQL = (
    "SELECT c.country, SUM(s.sales_amount) AS total_sales FROM fact_sales s "
    "JOIN dim_customers c ON s.customer_key = c.customer_key "
    "GROUP BY c.country ORDER BY total_sales DESC LIMIT 1"
)
_ONE_ROW = {
    "success": True,
    "columns": ["country", "total_sales"],
    "rows": [{"country": "United States", "total_sales": 9162327}],
}


def test_consistency_caps_high_score_when_short_breakdown_agrees(monkeypatch):
    # All 3 samples agree on the WRONG (LIMIT 1) answer; the new cap must keep
    # this at medium instead of awarding a 1.0 "high" from consistent errors.
    monkeypatch.setattr(repair_loop, "_expected_group_count", lambda engine, col: 7)
    monkeypatch.setattr(repair_loop, "generate_sql", lambda *a, **k: _LIMIT1_SQL)
    monkeypatch.setattr(repair_loop, "execute_sql", lambda sql, engine: dict(_ONE_ROW))
    score, note = repair_loop._check_consistency(
        "What is the total sales amount by country?",
        schema_context="fake schema",
        engine=None,
        original_result=dict(_ONE_ROW),
    )
    assert score == 2 / 3
    assert "capped" in note
    assert "incomplete" in note
    assert "7 groups" in note


def test_consistency_no_cap_when_breakdown_not_suspicious(monkeypatch):
    # A full 7-row breakdown that agrees everywhere stays at full score.
    monkeypatch.setattr(repair_loop, "_expected_group_count", lambda engine, col: 7)
    monkeypatch.setattr(repair_loop, "generate_sql", lambda *a, **k: _LIMIT1_SQL)
    rows7 = {
        "success": True,
        "columns": ["country", "total_sales"],
        "rows": [{"country": f"C{i}", "total_sales": 1000 + i} for i in range(7)],
    }
    monkeypatch.setattr(repair_loop, "execute_sql", lambda sql, engine: dict(rows7))
    score, note = repair_loop._check_consistency(
        "What is the total sales amount by country?",
        schema_context="fake schema",
        engine=None,
        original_result=dict(rows7),
    )
    assert score == 1.0
    assert "capped" not in note


# --------------------------------------------------------------------------
# Syntax / import check across app/ and frontend/
# --------------------------------------------------------------------------

_PYTHON_DIRS = [PROJECT_ROOT / "app", PROJECT_ROOT / "frontend"]


def _iter_py_files():
    files = []
    for d in _PYTHON_DIRS:
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            files.append(p)
    return files


@pytest.mark.parametrize("path", _iter_py_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_py_file_is_syntactically_valid(path):
    compile(path.read_bytes(), str(path), "exec")


def test_core_modules_import():
    for name in [
        "app.main",
        "app.agents.executor",
        "app.agents.orchestrator",
        "app.agents.sql_generator",
        "app.agents.verifier",
        "app.agents.repair_loop",
    ]:
        importlib.import_module(name)