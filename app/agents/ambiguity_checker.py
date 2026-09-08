"""
Looks at the user's question + schema context and decides: is this question
answerable as-is, or too vague to generate a reliable SQL query for?

Layered design:
  1. A deterministic, schema-aware metric resolver runs first. If the question
     names a metric whose column(s) it can resolve exactly, it rules WITHOUT an
     LLM call: one candidate column -> "clear"; several genuinely different
     columns -> deterministic clarification; none -> falls through to the LLM.
  2. The LLM judge only handles heuristic cases (ranking words, vague wording…).
This removes judge flakiness on exact-metric questions (e.g. price vs cost).
"""

import json
import re
from typing import Optional

from app.llm.factory import get_llm_client
from app.llm.ollama_client import LLMTimeoutError

# entity word -> table
_ENTITY_TABLES = {
    "customer": "dim_customers",
    "customers": "dim_customers",
    "product": "dim_products",
    "products": "dim_products",
    "sale": "fact_sales",
    "sales": "fact_sales",
    "order": "fact_sales",
    "orders": "fact_sales",
    "transaction": "fact_sales",
    "transactions": "fact_sales",
}

# metric word -> candidate columns per table (for THIS fixed demo schema)
_TABLE_METRICS = {
    "dim_customers": {
        "count": ["customer_id"],
        "number": ["customer_id"],
        "customer": ["customer_id"],
    },
    "dim_products": {
        "cost": ["cost"],
        "price": ["cost"],
    },
    "fact_sales": {
        "amount": ["sales_amount"],
        "value": ["sales_amount"],
        "sales": ["sales_amount"],
        "revenue": ["sales_amount"],
        "price": ["price"],
        "quantity": ["quantity"],
        "order": ["order_number"],
        "transactions": ["order_number"],
        "number": ["order_number"],
    },
}

# metric token -> entity table it anchors to (used when the question names both
# the metric word and an entity, so we disambiguate on the metric's column(s)
# inside the entity the metric belongs to rather than on unrelated id columns).
_METRIC_ANCHOR = {
    "amount": "fact_sales",
    "value": "fact_sales",
    "sales": "fact_sales",
    "revenue": "fact_sales",
    "price": "fact_sales",  # "sale price" resolves to sales; bare "price"
                            # (no entity words) still flags via dim_products + fact_sales
    "quantity": "fact_sales",
    "count": "dim_customers",
    "number": "fact_sales",
    "cost": "dim_products",
}

_METRIC_RE = re.compile(
    r"\b(how many|total|sum|average|avg|mean|count|number|minimum|min|maximum|max)\b",
    re.IGNORECASE,
)

_TABLE_RE = re.compile(r"Table:\s*(\w+)\s*\((.*?)\)\s*$", re.MULTILINE)

# Devanagari (Hindi) -> English gloss for the core vocabulary the deterministic
# resolver + LLM judge rely on. Applied WORD-LEVEL: only Devanagari words (and
# romanized-Hindi words, see _ROMAN_GLOSS) are translated; Latin-script English
# words, country names and proper nouns pass through untouched in place. Not
# exhaustive — it maps the metric/entity/country words needed to route the
# question, and the leftover Hindi filler words are harmless to the resolver
# (only the gloss is shown to the LLM).
_DEVA_RE = re.compile(r"[\u0900-\u097F]")

# Romanized-Hindi (Hinglish) word -> English. Values are the English word(s);
# None means a stop word to DROP (grammar words like hai/hain/kya), not a
# translation. Applied per-word only, so English words are never touched.
_ROMAN_GLOSS = {
    "mein": "in",
    "kitne": "how many",
    "kitni": "how much",
    "kitna": "how much",
    "kitnay": "how many",
    "kya": None,
    "hai": None,
    "hain": None,
    "ha": None,
    "ka": None,
    "ki": None,
    "ke": None,
    "ko": None,
    "kyun": None,
    "karein": None,
    "karen": None,
    "karo": None,
    "karna": None,
    "kijiye": None,
    "karke": None,
    "batao": None,
    "bataiye": None,
    "bata": None,
    "aur": "and",
    "kuch": None,
    "meri": None,
    "mere": None,
    "mujhe": None,
    "hum": None,
}
_DEVA_GLOSS = {
    # multi-word first (longest-match replaces first)
    "\u0938\u0902\u092f\u0941\u0915\u094d\u0924 \u0930\u093e\u091c\u094d\u092f \u0905\u092e\u0947\u0930\u093f\u0915\u093e": "United States",
    "\u092f\u0942\u0928\u093e\u0907\u091f\u0947\u0921 \u0915\u093f\u0902\u0917\u0921\u092e": "United Kingdom",
    "\u0915\u0947 \u0905\u0928\u0941\u0938\u093e\u0930": "by",
    # entities
    "\u0917\u094d\u0930\u093e\u0939\u0915\u094b\u0902": "customers",
    "\u0917\u094d\u0930\u093e\u0939\u0915": "customer",
    "\u0909\u0924\u094d\u092a\u093e\u0926\u094b\u0902": "products",
    "\u0909\u0924\u094d\u092a\u093e\u0926": "product",
    "\u0911\u0930\u094d\u0921\u0930": "order",
    "\u0906\u0926\u0947\u0936": "order",
    # metrics / aggregates
    "\u0915\u0941\u0932": "total",
    "\u0915\u093f\u0924\u0928\u0947": "how many",
    "\u0915\u093f\u0924\u0928\u093e": "how many",
    "\u0915\u093f\u0924\u0928\u0940": "how many",
    "\u092c\u093f\u0915\u094d\u0930\u0940": "sales",
    "\u092c\u0947\u091a\u0940": "sales",
    "\u0906\u092e\u0926\u0928\u0940": "revenue",
    "\u0930\u093e\u0936\u093f": "amount",
    "\u092e\u093e\u0924\u094d\u0930\u093e": "quantity",
    "\u0915\u0940\u092e\u0924": "price",
    "\u092e\u0942\u0932\u094d\u092f": "value",
    "\u0932\u093e\u0917\u0924": "cost",
    "\u0914\u0938\u0924": "average",
    "\u0909\u091a\u094d\u091a\u0924\u092e": "maximum",
    "\u0928\u094d\u092f\u0942\u0928\u0924\u092e": "minimum",
    # dimensions / filters
    "\u0926\u0947\u0936": "country",
    "\u0905\u0928\u0941\u0938\u093e\u0930": "by",
    "\u092e\u0947\u0902": " in ",
    # country names that appear in the demo data
    "\u0911\u0938\u094d\u091f\u094d\u0930\u0947\u0932\u093f\u092f\u093e": "Australia",
    "\u0905\u0938\u094d\u091f\u094d\u0930\u0947\u0932\u093f\u092f\u093e": "Australia",  # ASR variant (Whisper: अस्ट्रेलिया)
    "\u091c\u0930\u094d\u092e\u0928\u0940": "Germany",
    "\u0915\u0928\u093e\u0921\u093e": "Canada",
    "\u092b\u094d\u0930\u093e\u0902\u0938": "France",
    "\u0935\u093f\u0926\u0947\u0936": "abroad",
    "\u0905\u092e\u0947\u0930\u093f\u0915\u093e": "United States",
    "\u092c\u094d\u0930\u093f\u091f\u0947\u0928": "United Kingdom",
    # ASR/TTS variants of ग्राहक and अनुसार (from whisper test clips)
    "\u0917\u094d\u0930\u0939\u093e\u0917": "customers",
    "\u092e\u0941\u0938\u093e\u0930": "by",
    # fused words Whisper produces when it drops the space between tokens
    "\u0915\u0941\u0932\u094d\u092c\u093f\u0915\u094d\u0930\u0940": "total sales",
}

# Pure-Devanagari case markers / particles that add no routing value; dropped
# when they survive phase 1 unmapped (e.g. "के" left over after "के अनुसार").
_DEVA_STOP = frozenset(
    [
        "\u0915\u0947", "\u0915\u0940", "\u0915\u093e", "\u0915\u094b",
        "\u0938\u0947", "\u092a\u0930", "\u092e\u0947", "\u0915\u0930",
    ]
)


def _split_deva_token(token: str, keys: list[str]) -> list[str] | None:
    """Greedily split a fused Devanagari token (space dropped by Whisper, e.g.
    'कुल्बिक्री') into known _DEVA_GLOSS keys. Returns the list of gloss values,
    or None when the token cannot be fully resolved."""
    parts: list[str] = []
    rest = token
    while rest:
        match_key = None
        for key in keys:  # longest-first
            if rest.startswith(key):
                match_key = key
                break
        if match_key is None:
            return None
        parts.append(match_key)
        rest = rest[len(match_key):]
    return parts


def _hindi_gloss(question: str) -> str | None:
    """Word-level English gloss for Hindi / Hinglish questions.

    Returns None when the question is plain English (no Devanagari script and no
    romanized-Hindi words). Devanagari words are translated with _DEVA_GLOSS
    (multi-word keys first), romanized-Hindi (Hinglish) words with _ROMAN_GLOSS,
    and every other Latin-script word is left untouched in place. Partial
    translation is fine: the goal is to give the resolver its English
    metric/entity words and the LLM a readable question, not to produce a
    grammatical translation."""
    has_deva = bool(_DEVA_RE.search(question))
    has_roman = any(
        w.lower() in _ROMAN_GLOSS
        for w in re.findall(r"[A-Za-z]+", question)
    )
    if not has_deva and not has_roman:
        return None

    # Phase 1: Devanagari, whole-phrase longest-match first (handles keys with
    # spaces like 'संयुक्त राज्य अमेरिका' and 'के अनुसार'). Substring replace is
    # safe here because English words cannot contain Devanagari script.
    out = question
    for src, dst in sorted(_DEVA_GLOSS.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(src, dst)

    # Phase 2: word level — translate/drop romanized-Hindi tokens, split fused
    # Devanagari tokens, leave every other Latin-script word untouched in place.
    deva_keys = sorted(_DEVA_GLOSS, key=len, reverse=True)
    rebuilt = []
    for w in out.split():
        low = w.lower()
        if low in _ROMAN_GLOSS:
            repl = _ROMAN_GLOSS[low]
            if repl is not None:
                rebuilt.extend(repl.split())
            continue  # None = stop word, dropped
        if _DEVA_RE.search(w):
            if w in _DEVA_GLOSS:
                rebuilt.extend(_DEVA_GLOSS[w].split())
            else:
                split = _split_deva_token(w, deva_keys)
                if split is not None:
                    for sk in split:
                        rebuilt.extend(_DEVA_GLOSS[sk].split())
                elif w in _DEVA_STOP:
                    continue  # dangling case particle, drop
                else:
                    rebuilt.append(w)  # unknown Devanagari: keep verbatim
            continue
        rebuilt.append(w)
    out = " ".join(rebuilt)

    # Phase 3: leftover Devanagari filler (हैं/है/क्या/करें) + whitespace cleanup.
    # Phase 2 can linger 'में' -style Devanagari tokens, so strip filler again.
    out = re.sub(r"(?:\u0915\u094d\u092f\u093e|\u0939\u0948\u0902|\u0939\u0948|\u0915\u0930\u0947\u0902|\u0939\u094b)\s*", " ", out)
    out = re.sub(r"\s+", " ", out).strip()

    # Phase 4: 'के अनुसार' -> 'by' keeps the dimension first ("country by total
    # sales"), which reads like 'top country by sales' and prompts the generator
    # to add LIMIT 1. Metric-first phrasing ("total sales by country") asks the
    # LLM for the per-dimension breakdown instead.
    m = re.match(r"^(.+?)\s+by\s+(.+)$", out, re.IGNORECASE)
    if m and not re.search(
        r"\b(how many|total|sum|average|avg|mean|count|number|sales|amount|revenue|price|quantity)\b",
        m.group(1), re.IGNORECASE,
    ):
        return f"{m.group(2).strip()} by {m.group(1).strip()}"
    return out


def translate_question(question: str) -> tuple[Optional[str], Optional[str]]:
    """Detect the input language and return (language_code, English_gloss).

    Returns (None, None) for plain English. Detects Devanagari Hindi and
    romanized Hinglish; both report code "hi". The gloss is what the rest of the
    pipeline reasons over, and is also surfaced to the UI for the user to verify.
    """
    gloss = _hindi_gloss(question)
    if gloss is None:
        return None, None
    return "hi", gloss

SYSTEM_PROMPT = """\
You are an ambiguity checker. Given a user question and a database schema, \
determine whether the question is clear enough to write a SQL query for.

Rules:
- A question with a specific, single metric and no undefined ranking criteria \
(e.g. total, count, average of a named column) is NOT ambiguous, even if short.
- A named filter value (a country, region, date, year, price, product, etc.) is \
a data filter, NOT an ambiguity. Never ask for clarification because of a filter \
value, even an unusual one like a country with no data.
- Referencing an entity (table/column/term) that might not exist in the schema \
is NOT ambiguity: whether the question is answerable is decided at execution \
time, not here.
- Only mark ambiguous when a ranking word like "top", "best", "worst", "most \
popular", "best-selling" has no defined criterion (by revenue? by count? by \
recency?), or when a required filter value is missing (which year? which product?).
- A question naming a specific filter value (a place, date, category, or ID) is \
NOT ambiguous, even if that value might not exist in the data or might return \
zero results — that's a valid, answerable question with the answer 'zero' or \
'none found', not a vague question needing clarification. Ambiguity means the \
CRITERIA are unclear (e.g. 'top' without saying by what), not that the ANSWER \
might turn out to be empty.
- Ambiguity also covers a single metric word that maps to genuinely different \
columns. Example: 'price' could mean a product's cost in the products table or \
the per-unit price in the sales table. If the user's metric name matches \
several columns with different meanings, ask a short disambiguating question \
naming the columns (e.g. 'Did you mean the product's cost or the sale price?').
- A metric that maps to exactly one column for the entity named in the question \
is NOT ambiguous (e.g. 'average sale price' -> one column). Only flag when a \
real multi-column choice exists.

Respond in strict JSON only, no other text:
{"ambiguous": true or false, "clarifying_question": "if ambiguous, ask a short \
clarifying question; otherwise empty string"}

Examples (schema contains sales, customers and products tables):
User: What is the total sales amount?
Assistant: {"ambiguous": false, "clarifying_question": ""}

User: How many customers are there?
Assistant: {"ambiguous": false, "clarifying_question": ""}

User: What is the total sales amount for customers in Antarctica?
Assistant: {"ambiguous": false, "clarifying_question": ""}

User: Show me the top customers
Assistant: {"ambiguous": true, "clarifying_question": "What do you mean by 'top' \
customers? Total sales, number of orders, or average spending?"}"""

JSON_RE = re.compile(r"\{[^{}]*\}")


def _parse_schema_tables(schema_context: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for m in _TABLE_RE.finditer(schema_context):
        tbl = m.group(1)
        cols_raw = m.group(2)
        cols = [c.split()[0] for c in cols_raw.split(",") if c.strip()]
        out[tbl] = cols
    return out


def _resolve_metric(question: str, schema_tables: dict[str, list[str]]) -> Optional[dict]:
    """
    Deterministic metric→column resolver.

    Only treats a question as ambiguous when ONE metric concept (e.g. 'price')
    genuinely maps to multiple target columns. Distinct nouns that each map to a
    single column (a requested metric plus filter/grouping/HAVING words like
    'customers' or 'orders') are NOT ambiguity.

    Returns one of:
      {"ambiguous": True,  "clarifying_question": "..."}  – one concept, >1 column
      {"ambiguous": False, "clarifying_question": "",
       "columns": [(tbl, col), ...],
       "metric_terms": {token: [(tbl, col), ...]}}         – clean resolution
      None                                                – ambiguous; let LLM judge
    """
    ql = question.lower()
    if not _METRIC_RE.search(ql):
        return None

    # Only consider tables that actually exist in THIS schema. The resolver is
    # anchored on the demo vocabulary, but every candidate must be a real table
    # in the connected database (a user-uploaded DB never inherits demo facts).
    present = set(schema_tables.keys())

    # entity
    entity_tables: list[str] = []
    seen = set()
    for tok, tbl in _ENTITY_TABLES.items():
        if tbl in present and tok in ql and tbl not in seen:
            entity_tables.append(tbl)
            seen.add(tbl)

    # The metric word is the anchor. Look up columns ONLY in the entity whose
    # name the metric shares, e.g. "amount" belongs to fact_sales. Use the
    # metric's anchor table when one is defined.
    metric_tok = None
    for met_tok, _cols in _TABLE_METRICS.get("fact_sales", {}).items():
        if met_tok in ql:
            metric_tok = met_tok
            break
    anchor = _METRIC_ANCHOR.get(metric_tok)
    if anchor is not None and anchor in entity_tables:
        metric_tables = [anchor]
    else:
        metric_tables = entity_tables

    search_tables = [
        t for t in (metric_tables or list(_TABLE_METRICS.keys())) if t in present
    ]
    by_concept: dict[str, set] = {}
    for tbl in search_tables:
        metrics = _TABLE_METRICS.get(tbl, {})
        for met_tok, cols in metrics.items():
            if met_tok in ql:
                by_concept.setdefault(met_tok, set()).update((tbl, c) for c in cols)

    # Generic fallback for any schema: when a metric word literally names a
    # column that exists in the actual tables (e.g. "price" column in a user
    # upload), resolve to it directly. Demo columns never collide with this
    # (sales_amount != "amount", order_number != "order", ...), so the sample
    # behavior stays byte-identical: "price" still resolves to two columns.
    metric_vocab = {tok for tbl_metrics in _TABLE_METRICS.values() for tok in tbl_metrics}
    for met_tok in metric_vocab:
        if met_tok in ql:
            for tbl, cols in schema_tables.items():
                for col in cols:
                    if col.lower() == met_tok:
                        by_concept.setdefault(met_tok, set()).add((tbl, col))

    if not by_concept:
        return None

    # Same-concept competition only: flag when a SINGLE concept maps to >1 column.
    for met_tok, cols in by_concept.items():
        if len(cols) > 1:
            col_descs = [f"{t}.{c}" for t, c in sorted(cols)]
            return {
                "ambiguous": True,
                "clarifying_question": "Did you mean " + " or ".join(col_descs) + "?",
            }

    columns: list[tuple[str, str]] = []
    for cols in by_concept.values():
        for t, c in cols:
            if (t, c) not in columns:
                columns.append((t, c))
    return {
        "ambiguous": False,
        "clarifying_question": "",
        "columns": columns,
        "metric_terms": {tok: sorted(cols) for tok, cols in by_concept.items()},
    }


def _metric_role(sub_phrase: str) -> str:
    """Pick the aggregate driven by the metric keyword in a sub-phrase."""
    pl = sub_phrase.lower()
    if re.search(r"\b(how many|count|number)\b", pl):
        return "COUNT"
    if re.search(r"\b(average|avg|mean)\b", pl):
        return "AVG"
    return "SUM"


def _resolve_compound(question: str, schema_tables: dict[str, list[str]]) -> Optional[dict]:
    """
    Compound multi-metric questions (e.g. 'total sales amount and how many
    orders...'). Splits on ' and ', resolves each half through _resolve_metric,
    and returns a NOT-ambiguous verdict plus a schema hint that makes the SQL
    generator produce BOTH aggregates. Falls back to flagging only when one of
    the halves is itself genuinely ambiguous. Returns None if not a compound
    metric question.
    """
    splits = list(re.finditer(r"\s+and\s+", question))
    if not splits:
        return None

    for split in splits:
        left = question[: split.start()].strip()
        right = question[split.end():].strip()
        if not _METRIC_RE.search(left) or not _METRIC_RE.search(right):
            continue  # 'and' joins non-metric phrases here; not a compound metric ask

        left_res = _resolve_metric(left, schema_tables)
        right_res = _resolve_metric(right, schema_tables)
        if left_res is None or right_res is None:
            continue  # one side is not a resolvable metric phrase

        if left_res.get("ambiguous") or right_res.get("ambiguous"):
            bad = left_res if left_res.get("ambiguous") else right_res
            return {"ambiguous": True, "clarifying_question": bad["clarifying_question"]}

        hint_lines = [
            "REQUIRED METRICS NOTE: the user asked for TWO separate metrics in one "
            "question. Write a single SELECT that includes ALL of the following aggregates:"
        ]
        columns: list[tuple[str, str]] = []
        for phrase, res in ((left, left_res), (right, right_res)):
            fn = _metric_role(phrase)
            for tbl, col in res["columns"]:
                hint_lines.append(
                    f"- {fn}({tbl}.{col}) AS {col}  (from the phrase: {phrase.strip()})"
                )
                if (tbl, col) not in columns:
                    columns.append((tbl, col))
        return {
            "ambiguous": False,
            "clarifying_question": "",
            "columns": columns,
            "schema_hint": "\n".join(hint_lines),
        }

    return None


def check_ambiguity(question: str, schema_context: str) -> dict:
    """Check if a question is ambiguous and needs clarification."""
    schema_tables = _parse_schema_tables(schema_context)

    # Hindi / Hinglish / non-Latin questions: work from an English gloss so the
    # deterministic resolver and the LLM judge both see recognizable vocabulary.
    lang, gloss = translate_question(question)
    q = gloss if gloss is not None else question
    meta = {"detected_language": lang, "translated_question": gloss}
    hint = None
    if gloss is not None:
        hint = (
            "LANGUAGE NOTE: the user asked in Hindi. English translation: "
            + gloss
            + "\nGenerate the SQL against the English schema, translating any "
            "values (e.g. country names) to the English forms stored in the data."
        )
        # 'total sales by country' is a grouped-breakdown ask: the metric is
        # before 'by'. Tell the generator to return one row per group instead of
        # trimming to the top group with LIMIT.
        if re.search(r"\bby\s+\w+\??\s*$", gloss, re.IGNORECASE) and re.search(
            r"\b(how many|total|sum|average|avg|mean|count|number|sales|amount|revenue|price|quantity)\b",
            re.split(r"\s+by\s+", gloss)[0], re.IGNORECASE,
        ):
            hint += (
                " 'by X' means GROUP BY X: return one row per group, do NOT "
                "add LIMIT to narrow to a single group."
            )

    compound = _resolve_compound(q, schema_tables)
    if compound is not None:
        if hint:
            compound = dict(compound)
            compound["schema_hint"] = (compound.get("schema_hint") or "") + "\n\n" + hint
        compound.update(meta)
        return compound

    resolved = _resolve_metric(q, schema_tables)
    if resolved is not None:
        if hint:
            resolved = dict(resolved)
            resolved["schema_hint"] = (resolved.get("schema_hint") or "") + "\n\n" + hint
        resolved.update(meta)
        return resolved

    prompt = f"Schema:\n{schema_context}\n\nQuestion: {q}"

    try:
        raw = get_llm_client("reasoning").generate(
            prompt, system=SYSTEM_PROMPT, temperature=0.2
        )
    except LLMTimeoutError:
        # Judge unavailable: default to "not ambiguous" so the pipeline still
        # gets a chance to answer; a later stage can fail gracefully.
        return {"ambiguous": False, "clarifying_question": "", **meta}

    try:
        verdict = json.loads(raw)
        verdict.update(meta)
        return verdict
    except json.JSONDecodeError:
        pass

    match = JSON_RE.search(raw)
    if match:
        try:
            verdict = json.loads(match.group())
            verdict.update(meta)
            return verdict
        except json.JSONDecodeError:
            pass

    return {"ambiguous": False, "clarifying_question": "", **meta}
