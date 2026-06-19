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

from src.core.config import DUCKDB_PATH, STRUCT_DIR, EXCEL_TABLE_MAP


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


# Matches a "date" or "deadline" token bounded by underscores/start/end of a
# normalised (snake_case) column name. Deliberately word-bounded so it does
# NOT match unrelated columns that merely contain the substring "date"
# (e.g. "validated", "mandate", "update").
_DATE_COL_RE = re.compile(r"(?:^|_)(date|deadline)s?(?:$|_)")


def _coerce_date_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Parse date-looking columns into real datetime64 values so DuckDB creates
    them as native DATE columns instead of VARCHAR.

    Why this matters: source Excel files have inconsistent date formats
    (e.g. "14 Jun 2023" alongside ISO dates). When pandas can't infer a single
    dtype for a column it falls back to plain strings, and DuckDB then stores
    that column as VARCHAR. Every downstream `col::DATE` cast or date_diff()
    call the SQL agent generates then fails with
    "invalid date field format ... expected format is (YYYY-MM-DD)" — even
    though the agent's SQL is correct. Fixing the type at ingestion time
    (here) is the right layer; no amount of prompt-tuning the SQL agent can
    fix a string-typed date column.
    """
    for col in df.columns:
        if not _DATE_COL_RE.search(col):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue  # pandas already parsed it cleanly — nothing to do

        # NOTE: pandas silently mis-parses mixed-format date columns unless you
        # pass format="mixed" explicitly — without it, pandas locks onto the
        # format of the FIRST non-null value and returns NaT for every row that
        # doesn't match that exact format, even though the column is clearly
        # full of valid dates. dayfirst=True resolves DD/MM vs MM/DD ambiguity
        # for purely numeric dates the Singapore-convention way.
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=True)
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        success_rate = parsed.notna().sum() / non_null
        if success_rate >= 0.9:
            df[col] = parsed.dt.date  # plain date objects -> DuckDB DATE type
            n_failed = non_null - parsed.notna().sum()
            if n_failed:
                logger.warning(
                    f"[DB] '{table_name}.{col}': {n_failed} value(s) could not be "
                    f"parsed as dates and were set to NULL."
                )
        else:
            logger.warning(
                f"[DB] '{table_name}.{col}' looks like a date column but only "
                f"{success_rate:.0%} of values parsed as dates — left as VARCHAR. "
                f"Inspect this column manually if SQL date arithmetic on it fails."
            )
    return df


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
            df = _coerce_date_columns(df, table_name)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
            col_names = df.columns.tolist()
            schema_info[table_name] = col_names
            seen_tables.add(table_name)
            logger.info(f"[DB] Loaded '{table_name}' ({len(df)} rows, cols: {col_names}) from {excel_path.name}")
        except Exception as exc:
            logger.error(f"[DB] Failed to load '{table_name}' from {excel_path}: {exc}")

    con.close()
    return schema_info


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
            # Sample 2 rows for context
            sample = con.execute(f"SELECT * FROM {tbl} LIMIT 2").fetchdf()
            lines.append(f"Table: {tbl}\n  Columns: {col_defs}")
            lines.append(f"  Sample rows:\n{sample.to_string(index=False)}\n")
        return "\n".join(lines)
    finally:
        con.close()


if __name__ == "__main__":
    schema = load_all_tables()
    print("\n=== Schema loaded ===")
    print(get_schema_context())