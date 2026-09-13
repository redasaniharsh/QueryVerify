"""
Upload loader tests: CSV and Excel (.xlsx) must load identically and feed the
same downstream pipeline. No LLM, no DB, no Ollama.

The app writes each loaded DataFrame to one per-file SQLite table; the tests
verify CSV and XLSX are indistinguishable at both the parse layer and the
SQLite round-trip layer.
"""

import io
import sqlite3

import pandas as pd
import pytest

from frontend.upload_loader import load_uploaded_file


def _sample_frame():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["alice", "bob", "carol", "dave"],
            "score": [9.5, 8.0, 7.25, 6.0],
            "active": [True, False, True, False],
        }
    )


def _csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def _xlsx_bytes(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def test_csv_and_xlsx_load_identically():
    df = _sample_frame()
    from_csv = load_uploaded_file("scores.csv", _csv_bytes(df))
    from_xlsx = load_uploaded_file("scores.xlsx", _xlsx_bytes(df))
    pd.testing.assert_frame_equal(from_csv, from_xlsx)


def test_csv_and_xlsx_round_trip_through_sqlite_identically():
    df = _sample_frame()
    conn_csv = sqlite3.connect(":memory:")
    conn_xlsx = sqlite3.connect(":memory:")
    try:
        df.to_sql("scores", conn_csv, index=False, if_exists="replace")
        load_uploaded_file("scores.xlsx", _xlsx_bytes(df)).to_sql(
            "scores", conn_xlsx, index=False, if_exists="replace"
        )
        pd.testing.assert_frame_equal(
            pd.read_sql_query("SELECT * FROM scores", conn_csv),
            pd.read_sql_query("SELECT * FROM scores", conn_xlsx),
        )
    finally:
        conn_csv.close()
        conn_xlsx.close()


def test_extension_matching_is_case_insensitive():
    df = _sample_frame()
    pd.testing.assert_frame_equal(
        load_uploaded_file("SCORES.XLSX", _xlsx_bytes(df)),
        load_uploaded_file("scores.CSV", _csv_bytes(df)),
    )


@pytest.mark.parametrize(
    "name", ["data.xls", "data.txt", "data.xlsx.exe", "votes", ""]
)
def test_unsupported_extensions_rejected(name):
    with pytest.raises(ValueError, match="CSV and Excel"):
        load_uploaded_file(name, b"whatever")


def test_empty_uploaded_file_raises_rather_than_silent():
    # An empty CSV has nothing to parse; the app's catch-all turns this into a
    # clean "Could not parse ..." message, so the loader must raise, not return
    # a silently-empty table.
    with pytest.raises(Exception):
        load_uploaded_file("empty.csv", b"")