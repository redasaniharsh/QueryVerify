"""
Systematic real-world phrasing sweep.

Tests natural, everyday phrasings of the SAME 3 underlying questions (18
variations) plus 3 genuinely vague questions that MUST still ask for
clarification. Catches resolver false positives: cases where the deterministic
metric resolver flags a perfectly clear question as ambiguous.

Usage (default SQLite sample):
    py scripts/phrasing_sweep.py
"""

import sys

from app.db.connection import get_engine
from app.db.schema_introspector import get_schema_context
from app.agents.ambiguity_checker import check_ambiguity

GROUPS = {
    "G1 countries": [
        "How many countries are there?",
        "What is the total number of countries?",
        "How many total countries are there?",
        "Count the number of countries.",
        "How many distinct countries do we have?",
        "Give me the country count.",
    ],
    "G2 customers": [
        "How many customers are there?",
        "What is the total number of customers?",
        "Give me the customer count.",
        "How many customers do we have in total?",
        "Count all customers.",
        "What's the number of customers?",
    ],
    "G3 sales": [
        "What is the total sales amount?",
        "How much have we sold in total?",
        "What's the total revenue?",
        "Give me the sum of all sales.",
        "What is our total sales figure?",
        "How much total sales do we have?",
    ],
}

MUST_FLAG = [
    "Show me the top customers",
    "What are the best products?",
    "Give me some sales insights",
]


def main() -> int:
    engine = get_engine()
    schema_context = get_schema_context(engine)
    bad = 0

    print(f"schema: {engine.url}\n")

    for group, questions in GROUPS.items():
        print(f"== {group} ==")
        verdicts = []
        for q in questions:
            res = check_ambiguity(q, schema_context)
            amb = bool(res.get("ambiguous"))
            verdicts.append(amb)
            if amb:
                bad += 1
                print(f"  FLAGGED  | {q}")
                print(f"             {res.get('clarifying_question', '')!r}")
            else:
                cols = res.get("columns") or []
                colstr = ", ".join(f"{t}.{c}" for t, c in cols) if cols else "(judge: clear)"
                print(f"  clear    | {q}  -> {colstr}")
        if all(v == verdicts[0] for v in verdicts):
            print(f"  -> consistent: all {'FLAGGED' if verdicts[0] else 'clear'}\n")
        else:
            bad += 1
            print(f"  -> INCONSISTENT within group ({verdicts})\n")

    print("== Group 4 (genuinely ambiguous — MUST still ask) ==")
    g4_ok = 0
    for q in MUST_FLAG:
        res = check_ambiguity(q, schema_context)
        amb = bool(res.get("ambiguous"))
        status = "OK  (still flagged)" if amb else "REGRESSION (now clear!)"
        print(f"  {status} | {q}")
        if not amb:
            bad += 1
            g4_ok += 1
        else:
            print(f"              {res.get('clarifying_question', '')!r}")

    print(f"\nTOTAL PROBLEMS: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())