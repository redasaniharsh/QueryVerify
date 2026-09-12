"""
Data-driven PK/FK detection tests (no LLM, no DB, no Ollama).

The three required scenarios, run on raw DataFrames — decisions must come
from actual cell values, never column names:

  1. Clean genuine primary key  -> detected, no warning.
  2. Intended key with 2 accidental duplicates (named 'ref_num', not 'id')
     -> warned about, purely from the data.
  3. Two related tables, one FK-style column with a few orphan references
     -> soft informational note (not a warning); repeats in the FK column
     are expected and NOT flagged.
"""

import pandas as pd

from frontend.upload_profiler import (
    profile_uploads,
    detect_primary_keys,
    detect_foreign_keys,
)


def test_case1_clean_primary_key_detected_no_warning():
    df = pd.DataFrame(
        {
            "record_key": [f"K{i:03d}" for i in range(1, 101)],
            "name": [f"cust {i % 7}" for i in range(100)],
            "city": ["Delhi"] * 50 + ["Mumbai"] * 50,
        }
    )
    pks, fks = profile_uploads({"customers": df})
    res = pks["customers"]
    assert res.primary_key == "record_key"
    assert res.warning is None
    assert res.warning_column is None
    assert fks == []


def test_case2_duplicate_id_warned_from_data_only():
    # 'ref_num' is deliberately NOT an id-like name: detection must still
    # notice that it almost uniquely identifies rows but has 2 collisions.
    vals = [f"X{i:03d}" for i in range(1, 51)]
    vals[0] = "X025"  # X025 already exists -> exactly one pair now repeats
    df = pd.DataFrame(
        {
            "ref_num": vals,
            "fill": [i % 3 for i in range(50)],
        }
    )
    pks, _ = profile_uploads({"t": df})
    res = pks["t"]
    assert res.primary_key is None, "no column is fully unique -> no PK"
    assert res.warning_column == "ref_num"
    assert "2 rows share the same value" in res.warning
    assert "ref_num" in res.warning


def test_no_uid_by_naming_only():
    # A column *named* id but with blanks or duplicates is not a key.
    df = pd.DataFrame(
        {
            "id": [1, 2, None, 4],          # has a blank
            "other": ["a", "b", "c", "a"],  # has a duplicate
        }
    )
    pks, _ = profile_uploads({"t": df})
    res = pks["t"]
    assert res.primary_key is None
    assert res.warning is None  # nothing is >=95% unique, so no false alarm
    assert res.candidates == []


def test_case3_foreign_key_orphans_informational_not_warning():
    products = pd.DataFrame(
        {
            "product_id": list(range(1, 11)),
            "product_name": [f"prod{i}" for i in range(1, 11)],
        }
    )
    # 42 orders. product_id repeats (normal for an FK) AND 2 of its values
    # (99, 100) do not exist in products.product_id -> orphan references.
    orders = pd.DataFrame(
        {
            "order_id": [f"O{i:03d}" for i in range(1, 43)],
            "product_id": [1, 2, 1, 3, 4, 2, 5, 6, 7, 8] * 4 + [99, 100],
        }
    )
    pks, fks = profile_uploads({"products": products, "orders": orders})

    assert pks["products"].primary_key == "product_id"
    assert pks["orders"].primary_key == "order_id"
    # Repeats in the FK column are expected: no warning on orders at all.
    assert pks["orders"].warning is None

    fk = [f for f in fks if f.table == "orders" and f.column == "product_id"]
    assert len(fk) == 1
    info = fk[0]
    assert info.references_table == "products"
    assert info.references_column == "product_id"
    assert info.orphan_rows == 2
    assert info.ratio >= 0.95


def test_fk_requires_genuine_primary_key_in_target():
    # Table B's column is unique-looking but has a blank -> NOT a genuine PK,
    # and its only other column repeats, so table A's values overlapping it
    # must NOT be called a foreign key (no genuine key to reference).
    b = pd.DataFrame({"code": [1, 2, 3, None], "grp": ["a", "a", "b", "b"]})
    a = pd.DataFrame({"ref": [1, 2, 3, 1], "row": [11, 12, 13, 14]})
    pks = detect_primary_keys({"a": a, "b": b})
    assert pks["a"].primary_key == "row"
    assert pks["b"].primary_key is None
    fks = detect_foreign_keys({"a": a, "b": b}, pks)
    assert not any(f.column == "ref" for f in fks)


def test_single_table_produces_no_foreign_keys():
    df = pd.DataFrame({"pk": list(range(10)), "ref": [1] * 10})
    pks, fks = profile_uploads({"solo": df})
    assert pks["solo"].primary_key == "pk"
    assert fks == []


def test_blank_rows_do_not_count_as_orphans():
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "product_id": [1, 2, 1, None, ""],  # blanks, not orphans
        }
    )
    products = pd.DataFrame({"product_id": [1, 2, 3], "name": ["a", "b", "c"]})
    pks, fks = profile_uploads({"products": products, "orders": orders})
    fk = [f for f in fks if f.column == "product_id"]
    assert len(fk) == 1
    assert fk[0].orphan_rows == 0
    assert fk[0].matched_rows == 3