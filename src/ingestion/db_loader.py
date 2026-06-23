"""
db_loader.py — Loads the REAL Excel structured datasets into DuckDB.

Differences from the original (broken) version:
  - Reads actual files from STRUCT_DIR, does NOT create fake one-row mocks.
  - Normalises column names (lowercase, underscores) for consistent SQL.
  - Exposes get_schema_context() so the SQL agent always uses the REAL schema.
  - Safe to call multiple times (CREATE OR REPLACE).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd
from loguru import logger

from src.core.config import DUCKDB_PATH, STRUCT_DIR, EMAIL_DIR, EXCEL_TABLE_MAP

# Date columns in these datasets are stored as text like "14 Jun 2023" (DD Mon YYYY).
# DuckDB cannot cast that, which is why date math kept failing. We parse such
# columns to real DATE at load time so the SQL agent can do (date_a - date_b),
# date_diff('year', a, b), etc., without dialect-specific functions.
_DATE_FORMATS = ["%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]


def _looks_like_date_col(col_name: str) -> bool:
    return ("date" in col_name) or ("deadline" in col_name)


def _coerce_dates(df: "pd.DataFrame") -> list:
    """
    Convert any column whose name implies a date into a real DATE
    (via pandas → DuckDB). Returns the list of columns that were converted.
    Only converts when the whole non-null column parses cleanly, so we never
    silently corrupt a column that merely contains the word 'date'.

    Uses .dt.date (not datetime64) so DuckDB stores a DATE — that makes
    (date_a - date_b) yield an integer number of days, which is what the SQL
    agent naturally reaches for.
    """
    converted = []
    for col in df.columns:
        if not _looks_like_date_col(col):
            continue
        s = df[col]
        non_null = s.notna().sum()
        if non_null == 0:
            continue
        if pd.api.types.is_datetime64_any_dtype(s):       # already datetime
            df[col] = s.dt.date
            converted.append(col)
            continue
        for fmt in _DATE_FORMATS:                          # try text formats
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            if parsed.notna().sum() == non_null:           # every value parsed
                df[col] = parsed.dt.date
                converted.append(col)
                break
    return converted


def _normalize_col(name: str) -> str:
    """Convert column name to snake_case lowercase."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _find_excel_file(directory: Path, stem_key: str) -> Path | None:
    """Case-insensitive file finder for Excel files."""
    for f in directory.iterdir():
        if f.suffix.lower() in (".xlsx", ".xls", ".csv"):
            if f.stem.lower().replace(" ", "_") == stem_key.replace(" ", "_"):
                return f
            if f.stem.lower() == stem_key.lower():
                return f
    return None


def load_all_tables(db_path: str = DUCKDB_PATH, data_dir: Path = STRUCT_DIR) -> Dict[str, list]:
    """
    Load all recognised Excel files from data_dir into DuckDB.
    Returns a dict mapping table_name → list of column names.
    """
    con = duckdb.connect(db_path)
    schema_info: Dict[str, list] = {}

    # Build a deduplicated map of (stem_normalised → table_name)
    seen_tables: set = set()
    for file_stem, table_name in EXCEL_TABLE_MAP.items():
        if table_name in seen_tables:
            continue

        excel_path = _find_excel_file(data_dir, file_stem)
        if excel_path is None:
            logger.warning(f"[DB] No file found for table '{table_name}' in {data_dir} (looked for '{file_stem}')")
            continue

        try:
            df = pd.read_excel(excel_path) if excel_path.suffix.lower() != ".csv" else pd.read_csv(excel_path)
            df.columns = [_normalize_col(c) for c in df.columns]
            date_cols = _coerce_dates(df)          # "14 Jun 2023" text → real DATE
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
            col_names = df.columns.tolist()
            schema_info[table_name] = col_names
            seen_tables.add(table_name)
            logger.info(
                f"[DB] Loaded '{table_name}' ({len(df)} rows, cols: {col_names}; "
                f"date cols parsed: {date_cols}) from {excel_path.name}"
            )
        except Exception as exc:
            logger.error(f"[DB] Failed to load '{table_name}' from {excel_path}: {exc}")

    con.close()
    return schema_info


def load_emails_table(db_path: str = DUCKDB_PATH, email_dir: Path = EMAIL_DIR) -> int:
    """
    Parse every email PDF into an `emails` table so the SQL agent can COUNT and
    GROUP BY correspondence — "how many contractors wrote in", "unique emails per
    person per topic", etc. Vector search cannot count across a corpus; SQL can.

    One row per individual email (the synthetic thread-summary chunk is skipped).
    Columns: source_file, thread_id, email_index, sender, recipients, cc,
             date_str, email_date (DATE, nullable), email_year (INT, nullable),
             subject, body.
    """
    try:
        from src.ingestion.email_parser import parse_email_directory
    except Exception as exc:
        logger.error(f"[DB] Cannot import email parser: {exc}")
        return 0

    chunks = parse_email_directory(email_dir)
    rows = []
    for c in chunks:
        if getattr(c, "email_index", 1) == 0:      # skip thread-summary chunk
            continue
        rows.append({
            "source_file": getattr(c, "source", ""),
            "thread_id":   getattr(c, "thread_id", ""),
            "email_index": getattr(c, "email_index", 1),
            "sender":      getattr(c, "sender", ""),
            "recipients":  getattr(c, "recipients", ""),
            "cc":          getattr(c, "cc", ""),
            "date_str":    getattr(c, "date_str", ""),
            "subject":     getattr(c, "subject", ""),
            "body":        getattr(c, "text", ""),
        })

    if not rows:
        logger.warning(f"[DB] No emails found in {email_dir}; 'emails' table not created.")
        return 0

    df = pd.DataFrame(rows)
    # Best-effort real date; email_year is a robust fallback for year filters.
    df["email_date"] = pd.to_datetime(df["date_str"], errors="coerce").dt.date
    yr = df["date_str"].astype(str).str.extract(r"(20\d{2})")[0]
    df["email_year"] = pd.to_numeric(yr, errors="coerce")

    con = duckdb.connect(db_path)
    con.execute("CREATE OR REPLACE TABLE emails AS SELECT * FROM df")
    n = con.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    con.close()
    logger.info(f"[DB] Loaded 'emails' ({n} rows) from {email_dir}")
    return int(n)


def get_schema_context(db_path: str = DUCKDB_PATH) -> str:
    """
    Returns a human-readable schema description for all tables in DuckDB.
    This is injected directly into the SQL agent's prompt so it always
    uses real column names.
    """
    con = duckdb.connect(db_path)
    try:
        tables = con.execute("SHOW TABLES").fetchdf()
        if tables.empty:
            return "No tables found. Run load_all_tables() first."

        lines = ["Available DuckDB Tables and their columns:\n"]
        for _, row in tables.iterrows():
            tbl = row["name"]
            cols_df = con.execute(f"DESCRIBE {tbl}").fetchdf()
            col_defs = ", ".join(
                f"{r['column_name']} ({r['column_type']})"
                for _, r in cols_df.iterrows()
            )
            # Sample 2 rows; truncate long cells (e.g. email body) to keep the prompt lean.
            sample = con.execute(f"SELECT * FROM {tbl} LIMIT 2").fetchdf()
            sample = sample.astype(str).apply(lambda c: c.str.slice(0, 80))
            lines.append(f"Table: {tbl}\n  Columns: {col_defs}")
            lines.append(f"  Sample rows:\n{sample.to_string(index=False)}\n")
        return "\n".join(lines)
    finally:
        con.close()


if __name__ == "__main__":
    load_all_tables()
    load_emails_table()
    print("\n=== Schema loaded ===")
    print(get_schema_context())