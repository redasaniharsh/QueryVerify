"""Single loader for user-uploaded spreadsheet files (CSV + Excel .xlsx).

The Streamlit app and the test suite share one implementation so both file
formats are guaranteed to feed the same downstream path: DataFrame -> one
SQLite table per file in the session-scoped upload database.
"""

import io
import os

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def load_uploaded_file(name: str, data: bytes) -> pd.DataFrame:
    """Parse one uploaded file into a DataFrame.

    CSV files use pandas.read_csv; Excel .xlsx files use pandas.read_excel with
    the openpyxl engine (installed as a hard dependency). Raises ValueError with
    a user-facing message for unsupported extensions.
    """
    ext = os.path.splitext(name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(data))
    if ext == ".xlsx":
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    raise ValueError(f"{name}: only CSV and Excel (.xlsx) files are supported.")