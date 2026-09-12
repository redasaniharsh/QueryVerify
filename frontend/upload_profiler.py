"""
Data-driven primary-key / foreign-key detection for uploaded CSVs.

Decisions are made from actual cell values only — never from column names:
a column called ``id`` that contains blanks or duplicates is NOT a key, and
a column called ``ref_num`` with unique, complete values IS a key.

Rules
-----
* A *primary key* is a column where every row has a value and no value is
  repeated (both properties are checked on the data itself).
* If no clean key exists, a column that is *nearly* unique (>= 95%) but has
  a few repeated values is surfaced as a warning: it looks like an intended
  identifier, so results that identify rows by it may be inaccurate.
* A *foreign key* is a column in one table whose actual values are (95%+)
  all present in another table's genuine primary key. The few values that
  are not there are "orphan references" and earn a soft, informational note
  only — repeated values are expected in a foreign key and are never
  flagged as duplicates.

Nothing here touches Streamlit, the database, or the network, so it can be
unit-tested fully off-line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Share of non-blank values that must exist in the target table's primary
#: key for a column to be treated as a foreign-key reference.
FK_SUBSET_RATIO = 0.95
#: Unique-value share at/below which a noisy column stops looking key-like.
ALMOST_UNIQUE_RATIO = 0.95
#: Ignore columns with fewer than this many non-blank values while hunting
#: for foreign-key matches (tiny columns produce meaningless matches).
MIN_FK_VALUES = 3

_PRIMARY_KIND_FLOAT = "float64"


def _is_blank(value) -> bool:
    """True for None, NaN/NA, or a whitespace-only string."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _column_stats(series: pd.Series) -> dict:
    """Value-level facts used by every rule below. No names involved."""
    blank = series.map(_is_blank)
    present = series[~blank]
    n = len(present)
    counts = present.value_counts(dropna=False)
    unique_vals = len(counts)
    dup_rows = int(counts[counts > 1].sum())  # rows sharing a value with another row
    ratio = (unique_vals / n) if n else 0.0
    return {
        "n": n,
        "missing": int(blank.sum()),
        "unique_vals": unique_vals,
        "dup_rows": dup_rows,
        "ratio": ratio,
    }


def _pick_primary(candidates: list[str], df: pd.DataFrame) -> str:
    """Prefer the first non-float candidate; floats only if nothing else."""
    non_float = [c for c in candidates if str(df[c].dtype) != _PRIMARY_KIND_FLOAT]
    return (non_float or candidates)[0]


@dataclass
class PrimaryKeyResult:
    """Per-table detection outcome."""

    table: str
    primary_key: str | None
    candidates: list[str] = field(default_factory=list)
    warning: str | None = None
    warning_column: str | None = None


def detect_primary_keys(dfs: dict[str, pd.DataFrame]) -> dict[str, PrimaryKeyResult]:
    """One PrimaryKeyResult per uploaded table (order preserved)."""
    results: dict[str, PrimaryKeyResult] = {}
    for table, df in dfs.items():
        stats = {col: _column_stats(df[col]) for col in df.columns}
        candidates = [
            col
            for col, s in stats.items()
            if s["missing"] == 0 and s["ratio"] == 1.0
        ]
        primary_key = _pick_primary(candidates, df) if candidates else None
        res = PrimaryKeyResult(
            table=table,
            primary_key=primary_key,
            candidates=candidates,
        )
        if primary_key is None:
            noisy = None
            for col, s in stats.items():
                if s["missing"] == 0 and s["ratio"] >= ALMOST_UNIQUE_RATIO and s["ratio"] < 1.0:
                    if noisy is None or s["ratio"] > stats[noisy]["ratio"]:
                        noisy = col
            if noisy is not None:
                res.warning_column = noisy
                res.warning = (
                    f"Column '{noisy}' looks like it should uniquely identify "
                    f"each row, but {stats[noisy]['dup_rows']} rows share the "
                    f"same value - results involving this column may be "
                    "inaccurate."
                )
        results[table] = res
    return results


@dataclass
class ForeignKeyInfo:
    """A data-derived reference from one uploaded table to another."""

    table: str
    column: str
    references_table: str
    references_column: str
    matched_rows: int
    total_rows: int
    orphan_rows: int
    ratio: float


def detect_foreign_keys(
    dfs: dict[str, pd.DataFrame],
    pk_results: dict[str, PrimaryKeyResult],
) -> list[ForeignKeyInfo]:
    """Compare every column against every *genuine* PK of the other tables."""
    pks = {t: r.primary_key for t, r in pk_results.items() if r.primary_key}
    infos: list[ForeignKeyInfo] = []
    for a, df_a in dfs.items():
        if df_a.empty:
            continue
        own_pk = pks.get(a)
        for col in df_a.columns:
            if col == own_pk:
                continue
            stats = _column_stats(df_a[col])
            if stats["n"] < MIN_FK_VALUES:
                continue
            vals = df_a[col][~df_a[col].map(_is_blank)]
            best: ForeignKeyInfo | None = None
            for b, pk_col in pks.items():
                if b == a:
                    continue
                pk_vals = set(
                    dfs[b][pk_col][~dfs[b][pk_col].map(_is_blank)].tolist()
                )
                if not pk_vals:
                    continue
                matched = sum(1 for v in vals if _exists_in_set(v, pk_vals))
                total = len(vals)
                ratio = (matched / total) if total else 0.0
                if ratio >= FK_SUBSET_RATIO and (
                    best is None
                    or (ratio, matched) > (best.ratio, best.matched_rows)
                ):
                    best = ForeignKeyInfo(
                        table=a,
                        column=col,
                        references_table=b,
                        references_column=pk_col,
                        matched_rows=matched,
                        total_rows=total,
                        orphan_rows=total - matched,
                        ratio=ratio,
                    )
            if best is not None:
                infos.append(best)
    return infos


def _exists_in_set(value, value_set: set) -> bool:
    if _is_blank(value):
        return False
    try:
        return value in value_set
    except TypeError:
        # Mixed/hash-conflicting types (rare in CSV output from pandas);
        # fall back to equality matching so detection never crashes.
        return any(value == other for other in value_set if not _is_blank(other))


def profile_uploads(dfs: dict[str, pd.DataFrame]) -> tuple:
    """Run the full detection in one call.

    Returns ``(primary_keys, foreign_keys)``. Foreign-key analysis only runs
    when at least two tables are uploaded together.
    """
    pk_results = detect_primary_keys(dfs)
    fk_infos = (
        detect_foreign_keys(dfs, pk_results) if len(dfs) >= 2 else []
    )
    return pk_results, fk_infos