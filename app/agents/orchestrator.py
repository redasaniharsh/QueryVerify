"""
Entry point the API calls. Order of operations:

  0. has_destructive_intent(question)
       -> if the question asks for a destructive action, return a blocked
          response BEFORE any LLM or database call
  1. get_schema_context(engine)
  2. check_ambiguity(question, schema_context)
       -> if ambiguous: return the clarifying question, STOP here
  3. run_pipeline(question, schema_context, engine)
  4. return the confidence report
"""

import re
from difflib import SequenceMatcher
from time import perf_counter

from app.db.schema_introspector import get_schema_context
from app.agents.ambiguity_checker import (
    check_ambiguity,
    translate_question,
    _parse_schema_tables,
)
from app.agents.repair_loop import run_pipeline
from app.utils.trace import add as trace_add

DESTRUCTIVE_RE = re.compile(
    r"\b(delete|drop|truncate|update|insert|alter|remove|wipe|erase|destroy|modify)\b",
    re.IGNORECASE,
)

# Business-domain entities that are NOT in this schema. If a user names one,
# it is not a data filter (it is a thing being asked about), and no SQL should
# silently substitute a similar table/column. Deterministic, so this does not
# depend on LLM judgement.
ABSENT_ENTITIES = re.compile(
    r"\b(invoices?|employees?|salary|salaries|payroll|suppliers?|refunds?|"
    r"campaigns?|inventory|stock|hr|marketing)\b",
    re.IGNORECASE,
)


def has_destructive_intent(question: str) -> bool:
    """True if the question targets a destructive operation."""
    return bool(question and DESTRUCTIVE_RE.search(question))


def absent_entity(question: str) -> str | None:
    """Return the matched not-in-schema entity term, or None."""
    if question:
        m = ABSENT_ENTITIES.search(question)
        if m:
            return m.group(0)
    return None


# --------------------------------------------------------------------------
# Schema-aware concept guard (deterministic, zero AI cost).
#
# The static ABSENT_ENTITIES list above handles a fixed set of business terms
# fast regardless of schema. This layer goes further: it checks every business
# concept the question names against the ACTUAL schema that is active right now
# (the sample DB *or* a user upload). A concept like "sales" or "customers" is
# declined deterministically when no real table or column in the active schema
# relates to it — instead of relying solely on the LLM judge to notice.
#
# Each concept maps to the schema-token aliases that would count as "covering"
# it (substring / direction-aware): "revenue" is covered by a "sales_amount"
# column, "orders" by "order_number", etc. Words in this map are business
# descriptors; neutral words (how, many, total, average, dataset...) are never
# checked, and non-concept words are never the trigger for a decline.
# --------------------------------------------------------------------------

CONCEPT_ALIASES: dict[str, list[str]] = {
    "sales": ["sales", "sale", "sell"],
    "revenue": ["sales", "revenue", "income", "amount"],
    "income": ["sales", "revenue", "income", "amount"],
    "profit": ["sales", "revenue", "profit", "margin"],
    "expenses": ["expense", "cost", "spend"],
    "spending": ["spending", "expense", "cost"],
    "purchases": ["purchase", "order", "sales"],
    "customers": ["customer", "clients", "members"],
    "clients": ["customer", "client"],
    "orders": ["order"],
    "invoices": ["invoice", "bill"],
    "products": ["product", "sku", "items", "stock"],
    "items": ["item", "product", "sku"],
    "stock": ["stock", "inventory", "quantity", "units"],
    "inventory": ["inventory", "stock", "quantity", "units"],
    "quantity": ["quantity", "qty", "units", "count"],
    "price": ["price", "cost", "amount"],
    "cost": ["cost", "price", "expense"],
    "salary": ["salary", "pay", "wage", "compensation"],
    "payroll": ["payroll", "pay", "salary"],
    "employees": ["employee", "staff"],
    "suppliers": ["supplier", "vendor"],
    "refunds": ["refund", "return"],
    "campaigns": ["campaign", "promo", "marketing"],
    "hr": ["hr", "employee"],
    "marketing": ["marketing", "campaign"],
    "category": ["category", "subcategory", "class"],
    "country": ["country", "region", "nation"],
}


def _schema_covers(concept: str, aliases: list[str], vocab: set[str]) -> bool:
    """True if the active schema has a table/column that plausibly covers the
    concept: an alias (or the concept itself) appears inside a schema token, a
    schema token appears inside an alias, or they are fuzzy-equal."""
    candidates = set(aliases)
    candidates.add(concept.lower())
    for cand in candidates:
        cand = cand.lower()
        if not cand:
            continue
        for token in vocab:
            if cand in token or (len(cand) >= 3 and token in cand):
                return True
            if SequenceMatcher(None, cand, token).ratio() >= 0.8:
                return True
    return False


def schema_concept_mismatch(question: str, schema_context: str) -> list[str]:
    """Concepts the question names that have NO real column/table in this
    schema. Returns [] when every named concept is covered. Deterministic."""
    if not question:
        return []
    ql = question.lower()
    vocab: set[str] = set()
    for table, cols in _parse_schema_tables(schema_context).items():
        vocab.add(table.lower())
        vocab.update(c.lower() for c in cols)
    if not vocab:
        return []

    missing: list[str] = []
    for concept, aliases in CONCEPT_ALIASES.items():
        forms = {concept, concept[:-1]} if concept.endswith("s") else {concept}
        if not any(re.search(rf"\b{re.escape(f)}\b", ql) for f in forms):
            continue
        if not _schema_covers(concept, aliases, vocab):
            missing.append(concept)
    return missing


def handle_question(question: str, engine) -> dict:
    """Full pipeline: guard -> schema -> ambiguity check -> generate/execute/verify.

    Collects a step-by-step trace (schema, ambiguity, generation, execution,
    verification, consistency, final confidence) that every response carries so
    the UI can show how the answer was produced."""
    lang, gloss = translate_question(question)
    lang_meta = {"detected_language": lang, "translated_question": gloss}

    if has_destructive_intent(question):
        return {
            "needs_clarification": False,
            "blocked": True,
            "success": False,
            "sql": None,
            "result": None,
            "confidence": "low",
            "confidence_score": 0.0,
            "attempts": 0,
            "explanation": (
                "Blocked by the read-only guard: destructive action detected in "
                "the question, rejected before any LLM or database call."
            ),
            "trace": [
                {
                    "step": "guard",
                    "detail": "Blocked — destructive intent detected before any "
                    "LLM or database call",
                    "duration_s": 0.0,
                }
            ],
            **lang_meta,
        }

    missing = absent_entity(question)
    if missing:
        return {
            "needs_clarification": True,
            "clarifying_question": (
                f"This database has no '{missing}' data — it only contains "
                "the tables shown in the sidebar schema. Ask about those, "
                "or upload a CSV with the data you want to query."
            ),
            "trace": [
                {
                    "step": "absent_entity",
                    "detail": f"Schema check — '{missing}' is not a table or column "
                    "in this database",
                    "duration_s": 0.0,
                }
            ],
            **lang_meta,
        }

    trace: list = []

    t0 = perf_counter()
    schema_context = get_schema_context(engine)
    trace_add(
        trace,
        "schema_read",
        f"Read {schema_context.count('Table: ')} tables from the database",
        perf_counter() - t0,
    )

    # Deterministic, schema-aware layer: any business concept the question
    # names that maps to nothing in the ACTIVE schema is refused here, without
    # any LLM call. Neutral and value words never trigger this.
    missing_concepts = schema_concept_mismatch(
        gloss if gloss is not None else question, schema_context
    )
    if missing_concepts:
        display = ", ".join(f"'{m}'" for m in missing_concepts)
        return {
            "needs_clarification": True,
            "clarifying_question": (
                f"This dataset has no column or table related to {display} — "
                "it only contains the tables shown in the sidebar schema. Ask "
                "about those, or upload a file that includes that data."
            ),
            "trace": [
                {
                    "step": "absent_entity",
                    "detail": f"Schema check — no column or table in this dataset "
                    f"relates to {display}",
                    "duration_s": 0.0,
                }
            ],
            **lang_meta,
        }

    t0 = perf_counter()
    ambiguity = check_ambiguity(question, schema_context)
    if ambiguity.get("ambiguous"):
        trace_add(
            trace,
            "ambiguity",
            f"Ambiguous — {(ambiguity.get('clarifying_question') or '')[:120]}",
            perf_counter() - t0,
        )
        return {
            "needs_clarification": True,
            "clarifying_question": ambiguity["clarifying_question"],
            "trace": trace,
            **lang_meta,
        }

    cols = ambiguity.get("columns")
    if cols:
        col_str = ", ".join(f"{tbl}.{col}" for tbl, col in cols)
    else:
        col_str = "resolved by classification rules"
    trace_add(trace, "ambiguity", f"Clear — {col_str}", perf_counter() - t0)

    hint = ambiguity.get("schema_hint")
    if hint:
        schema_context = schema_context + "\n\n" + hint

    result = run_pipeline(question, schema_context, engine, trace)
    score = result.get("confidence_score")
    label = result.get("confidence")
    if score is not None:
        trace_add(trace, "final", f"{label} confidence ({score:.2f})", 0.0)
    else:
        trace_add(trace, "final", "No confidence assigned", 0.0)

    result["needs_clarification"] = False
    result["trace"] = trace
    result.update(lang_meta)
    return result
