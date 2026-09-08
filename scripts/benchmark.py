"""
Benchmark harness: baseline (one-shot generate + execute, no verification)
vs. the full QueryVerify pipeline (handle_question) across 10 test questions.
"""

from app.db.connection import get_engine

engine = get_engine()
from app.db.schema_introspector import get_schema_context
from app.agents.sql_generator import generate_sql
from app.agents.executor import execute_sql
from app.agents.orchestrator import handle_question

QUESTIONS = [
    "What is the total sales amount?",
    "How many customers are there?",
    "What is the total sales amount by country?",
    "What are the top 5 products by quantity sold?",
    "What is the average order value by product category?",
    "What was the total quantity of products sold in 2024?",
    "Which country has the most customers?",
    "How many different products were sold?",
    "What are the top 5 customers by total sales?",
    "Show me the top customers",
]


def baseline(question: str, schema_context: str) -> bool:
    """One-shot generation + execution. No repair, no verification."""
    try:
        sql = generate_sql(question, schema_context)
        result = execute_sql(sql, engine)
        return result.get("success", False)
    except Exception:
        return False


def run() -> None:
    schema_context = get_schema_context(engine)

    print(f"{'Question':<55} | base | QV  | conf             | att")
    print("-" * 100)

    baseline_ok = 0
    qv_ok = 0
    repairs = 0
    flagged_ambiguous = 0
    attempts_total = 0
    n_answered = 0

    for question in QUESTIONS:
        base_ok = baseline(question, schema_context)
        baseline_ok += base_ok

        try:
            resp = handle_question(question, engine)
        except Exception as e:
            print(f"{question:<55} | {'Y' if base_ok else 'N':<4} | ERROR | {str(e)[:40]:<18} | -")
            continue

        if resp.get("needs_clarification"):
            flagged_ambiguous += 1
            qv_ok_this = False
            confidence = f"flagged: {resp['clarifying_question'][:22]}"
            attempts = "-"
        else:
            attempts = resp.get("attempts")
            attempts_total += attempts or 0
            n_answered += 1
            if attempts and attempts > 1:
                repairs += 1
            success = resp.get("success", False) and resp.get("result", {}).get("success", False)
            qv_ok_this = bool(success)
            qv_ok += qv_ok_this
            confidence = f"{resp.get('confidence')} ({resp.get('confidence_score'):.2f})" if resp.get("confidence_score") is not None else "none"

        print(
            f"{question[:55]:<55} | {'Y' if base_ok else 'N':<4} | "
            f"{'Y' if qv_ok_this else 'N':<3} | {confidence:<18} | {attempts}"
        )

    print("-" * 100)
    print(f"baseline success rate:            {baseline_ok}/{len(QUESTIONS)}")
    print(f"QueryVerify success rate:         {qv_ok}/{len(QUESTIONS)}")
    avg_attempts = attempts_total / n_answered if n_answered else 0
    print(f"avg attempts (answered):          {avg_attempts:.2f}")
    print(f"questions needing repair (att>1): {repairs}/{len(QUESTIONS)}")
    print(f"correctly flagged ambiguous:      {flagged_ambiguous}/{len(QUESTIONS)}")


if __name__ == "__main__":
    run()