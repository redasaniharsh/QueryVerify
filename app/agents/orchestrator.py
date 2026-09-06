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

from app.db.schema_introspector import get_schema_context
from app.agents.ambiguity_checker import check_ambiguity, translate_question
from app.agents.repair_loop import run_pipeline

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


def handle_question(question: str, engine) -> dict:
    """Full pipeline: guard -> schema -> ambiguity check -> generate/execute/verify."""
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
            **lang_meta,
        }

    missing = absent_entity(question)
    if missing:
        return {
            "needs_clarification": True,
            "clarifying_question": (
                f"This database has no '{missing}' data — it only contains "
                "customers, products, and sales. Did you mean sales orders?"
            ),
            **lang_meta,
        }

    schema_context = get_schema_context(engine)
    ambiguity = check_ambiguity(question, schema_context)

    if ambiguity.get("ambiguous"):
        return {
            "needs_clarification": True,
            "clarifying_question": ambiguity["clarifying_question"],
            **lang_meta,
        }

    hint = ambiguity.get("schema_hint")
    if hint:
        schema_context = schema_context + "\n\n" + hint

    result = run_pipeline(question, schema_context, engine)
    result["needs_clarification"] = False
    result.update(lang_meta)
    return result
