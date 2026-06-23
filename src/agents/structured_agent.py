"""
structured_agent.py — Schema-aware Text-to-SQL agent.

Key behaviours:
  1. Fetches REAL schema at runtime via get_schema_context() — no hardcoded columns.
  2. Robust SQL extraction: isolates the statement even if the LLM wraps it in prose.
  3. DuckDB-specific guidance for the two things that broke real queries:
       - date math (dates are real DATE; use (a-b) for days, date_diff for years)
       - inconsistent casing in categorical columns (always ILIKE, never =)
  4. Retries once on SQL error, feeding the error back to the LLM.
  5. Never leaks SQL into the user-facing answer (SQL stays in telemetry metadata).
"""
from __future__ import annotations

import re
from typing import Dict, Any

import duckdb
from loguru import logger

from src.core.state           import AgentState
from src.core.llm             import get_llm
from src.core.config          import DUCKDB_PATH
from src.ingestion.db_loader  import get_schema_context

SQL_GENERATION_PROMPT = """\
You are a DuckDB SQL expert for the HCSA knowledge system.

{schema}

USER QUERY: {query}

Write a single valid DuckDB SQL SELECT statement that answers the query.
Rules:
- Use only the tables and columns listed above.
- DuckDB SQL syntax only.
- One line. No markdown, no backticks, no semicolons.
- For counts use COUNT(*) or COUNT(DISTINCT ...). Alias aggregates (e.g. COUNT(*) AS count).
- Join with explicit JOIN ... ON syntax.

DATA QUIRKS — follow these exactly or the results will be wrong:
- DATES: every column named *_date or *_deadline is a real DATE.
    * Difference in DAYS: subtract directly  ->  (end_date - start_date)   [integer days]
    * Difference in YEARS / MONTHS: date_diff('year', start, end) or date_diff('month', start, end)
    * "today"/"now" = CURRENT_DATE.  An ONGOING / incomplete project has actual_completion_date IS NULL.
    * NEVER use julianday(), DATE_SUB(), AGE(), DATEDIFF(), or ::DATE casts on these columns —
      they are already DATE, and those functions do not exist or differ in DuckDB.
- CASING: categorical text values are inconsistently cased — e.g. inspection_result holds BOTH
  'FAIL' and 'Fail', and 'PASS' and 'Pass'; project_status holds 'Suspended' among uppercase values.
  ALWAYS compare text with ILIKE, NEVER with =. For pass/fail counts use a prefix, e.g.
  SUM(CASE WHEN inspection_result ILIKE 'fail%' THEN 1 ELSE 0 END).

OUTPUT THE SQL STATEMENT ONLY. Do not explain it. Do not write any words before or
after it. The first token of your reply must be SELECT or WITH.
"""

RETRY_PROMPT = """\
The previous SQL failed.
ERROR: {error}
PREVIOUS SQL: {prev_sql}

{schema}

USER QUERY: {query}

Write a corrected single-line DuckDB SQL statement. Reminders: dates are real DATE
(use (end_date - start_date) for days, date_diff('year', a, b) for years; never
julianday/DATE_SUB/AGE/DATEDIFF); compare text with ILIKE, never =.
OUTPUT THE SQL ONLY — no explanation, no markdown. First token must be SELECT or WITH.
"""

SUMMARY_PROMPT = """\
Question: "{query}"

Result data:
{result}

Write a concise plain-language answer (2-4 sentences) stating what the data shows.
Do not mention SQL, queries, or databases. Do not invent values beyond the data above.
"""


_SQL_CONT = re.compile(
    r"(?i)^\s*(from|where|group|order|having|join|left|right|inner|outer|full|"
    r"cross|natural|on|and|or|not|union|intersect|except|limit|offset|select|"
    r"with|as|using|when|then|else|end|case|in|like|ilike|between|is|distinct|"
    r"[(),*]|\w+\s*[=<>!])")


def _extract_sql(raw: str) -> str:
    """
    Isolate a single SQL statement from a possibly-chatty LLM reply.

    llama-3.3-70b often wraps SQL in prose ("To address the error...\\nSELECT ...
    \\nThis query first...") especially on retry. Only stripping code fences let
    that prose reach DuckDB and crash it with 'syntax error at or near "To"'.
    This strips preamble, trailing explanation, fences, and semicolons.
    """
    text = (raw or "").strip()
    # 1. A fenced block already isolates the SQL — use its body.
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        body = fenced.group(1).strip()
        kw = re.search(r"\b(WITH|SELECT)\b", body, re.IGNORECASE)
        if kw:
            body = body[kw.start():]
        return re.sub(r"\s+", " ", body.split(";")[0]).strip()
    # 2. Drop any preamble by anchoring at the first SQL keyword.
    kw = re.search(r"\b(WITH|SELECT)\b", text, re.IGNORECASE)
    if kw:
        text = text[kw.start():]
    text = text.split(";")[0]
    # 3. Keep the first line, then only lines that continue the SQL; prose stops it.
    nonempty = [l for l in text.splitlines() if l.strip()]
    if not nonempty:
        return ""
    out = [nonempty[0].strip()]
    for ln in nonempty[1:]:
        if _SQL_CONT.match(ln.strip()):
            out.append(ln.strip())
        else:
            break
    return re.sub(r"\s+", " ", " ".join(out)).strip()


class StructuredAgent:
    def __init__(self):
        self.llm = get_llm()

    def _generate_sql(self, query: str, schema: str) -> str:
        prompt = SQL_GENERATION_PROMPT.format(schema=schema, query=query)
        return _extract_sql(self.llm.invoke([("user", prompt)]).content)

    def _retry_sql(self, query: str, schema: str, prev_sql: str, error: str) -> str:
        prompt = RETRY_PROMPT.format(error=error, prev_sql=prev_sql, query=query, schema=schema)
        return _extract_sql(self.llm.invoke([("user", prompt)]).content)

    def execute_query(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        schema = get_schema_context(DUCKDB_PATH)
        con = duckdb.connect(DUCKDB_PATH)

        sql = self._generate_sql(query, schema)
        logger.info(f"[StructuredAgent] Generated SQL: {sql}")

        result_df = None
        error_msg = None
        try:
            result_df = con.execute(sql).fetchdf()
        except Exception as exc:
            error_msg = str(exc)
            logger.warning(f"[StructuredAgent] SQL error: {exc} — retrying …")
            sql = self._retry_sql(query, schema, sql, error_msg)
            logger.info(f"[StructuredAgent] Retry SQL: {sql}")
            try:
                result_df = con.execute(sql).fetchdf()
                error_msg = None
            except Exception as exc2:
                error_msg = str(exc2)
                logger.error(f"[StructuredAgent] Retry also failed: {exc2}")
        finally:
            con.close()

        if result_df is not None and not result_df.empty:
            result_str = result_df.to_string(index=False)
            summary = self.llm.invoke([
                ("user", SUMMARY_PROMPT.format(query=query, result=result_str))
            ]).content.strip()
            context_chunk = {
                "id": "s1",
                "text": f"Result summary:\n{summary}\n\nData:\n{result_str}",
                "source": "Structured Datasets",
                "source_type": "structured",
                "page_start": 0,
                "section": "Structured query result",
                "ce_score": 1.0,
                "rrf_score": 1.0,
                "metadata": {"sql": sql, "row_count": len(result_df)},
            }
            citation = {"source": "Structured Datasets", "segment": "Structured query result",
                        "row_count": len(result_df)}
            timeline = [f"StructuredAgent: SQL executed ({len(result_df)} rows) → {sql}"]

        elif result_df is not None and result_df.empty:
            context_chunk = {
                "id": "s1",
                "text": "The structured query returned no matching rows for this question.",
                "source": "Structured Datasets",
                "source_type": "structured",
                "page_start": 0,
                "section": "No matching rows",
                "ce_score": 0.5,
                "rrf_score": 0.5,
                "metadata": {"sql": sql, "row_count": 0},
            }
            citation = {"source": "Structured Datasets", "segment": "No matching rows"}
            timeline = [f"StructuredAgent: SQL returned 0 rows — {sql}"]

        else:
            context_chunk = {
                "id": "s1",
                "text": "The structured data lookup could not be completed for this question.",
                "source": "Structured Datasets",
                "source_type": "structured",
                "page_start": 0,
                "section": "Lookup unavailable",
                "ce_score": 0.0,
                "rrf_score": 0.0,
                "metadata": {"sql": sql, "error": error_msg},
            }
            citation = {"source": "Structured Datasets", "segment": "Lookup unavailable"}
            timeline = [f"StructuredAgent: SQL FAILED — {error_msg}"]

        return {
            "retrieved_context": [context_chunk],
            "sql_queries": [sql],
            "citations": [citation],
            "execution_timeline": timeline,
        }