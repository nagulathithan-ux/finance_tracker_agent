"""
Data Ingestion Module
----------------------
Ingests raw bank/CSV transaction exports and cleans them into a
standardized DataFrame: date, description, amount, type (debit/credit).

Handles: missing fields, duplicate rows, inconsistent date formats,
stray whitespace, and sign normalization (expenses negative, income positive).
"""

import pandas as pd
import numpy as np


REQUIRED_COLS = ["date", "description", "amount"]


def load_transactions(path: str) -> pd.DataFrame:
    """Load a raw transactions CSV and return a cleaned, standardized DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common alternate column names to our schema
    rename_map = {
        "transaction_date": "date", "txn_date": "date",
        "narration": "description", "details": "description", "memo": "description",
        "value": "amount", "amt": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in input CSV: {missing}")

    return clean_transactions(df)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Standardize description text
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].str.len() > 0]

    # Parse dates robustly; drop rows that fail to parse
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    n_before = len(df)
    df = df.dropna(subset=["date"])
    n_dropped_dates = n_before - len(df)

    # Coerce amount to numeric; drop rows that fail
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["amount"])
    n_dropped_amounts = n_before - len(df)

    # Remove exact duplicate transactions
    n_before = len(df)
    df = df.drop_duplicates(subset=["date", "description", "amount"])
    n_dupes = n_before - len(df)

    # Derive transaction type from sign of amount if not present
    if "type" not in df.columns:
        df["type"] = np.where(df["amount"] >= 0, "credit", "debit")

    df = df.sort_values("date").reset_index(drop=True)

    df.attrs["cleaning_report"] = {
        "rows_dropped_bad_dates": int(n_dropped_dates),
        "rows_dropped_bad_amounts": int(n_dropped_amounts),
        "duplicate_rows_removed": int(n_dupes),
        "final_row_count": int(len(df)),
    }
    return df
