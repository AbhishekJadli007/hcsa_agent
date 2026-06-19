"""
structured_agent.py — Schema-aware Text-to-SQL agent (DuckDB).

Fixes applied per diagnosis (Part 2-4 of the structured-agent audit):
  1. DuckDB-correct date arithmetic rules baked into the prompt — the LLM's
     training data is dominated by Postgres/SQLite, so it defaults to
     julianday() / DATE_PART() / INTERVAL comparisons that DON'T EXIST or
     behave differently in DuckDB. (DATE - DATE) returns BIGINT in DuckDB,
     not INTERVAL, so date_part() on it always fails.
  2. COUNT(*) is always aliased -> no more ugly "count_star()" column.
  3. UPPER()/LOWER() casing rules for enum and status comparisons.
  4. Retry prompt maps SPECIFIC duckdb error substrings to specific fixes,
     instead of just repeating the original rules verbatim.
  5. Summary prompt requires a markdown table for any multi-row result.
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

GENERAL RULES:
- Use only tables and columns listed above.
- DuckDB SQL syntax only (no MySQL / PostgreSQL extensions).
- No markdown, no backticks, no semicolons, no newlines — one line only.
- Always alias aggregates, e.g. COUNT(*) AS count, AVG(x) AS avg_x — never leave
  an unaliased aggregate (DuckDB will otherwise render it as "count_star()").
- If joining tables, use explicit JOIN ... ON syntax.
- Use UPPER(column) = 'VALUE' for enum/category comparisons (ratings, results
  like PASS/FAIL, PLATINUM/GOLD/BRONZE) — real data has inconsistent casing.
- Use LOWER(column) = 'value' for status comparisons (ongoing, inactive,
  completed, active) for the same reason.
- For "highest/lowest" questions, return ALL tied rows, not just one
  (use a CTE with MAX()/MIN() and compare, not ORDER BY ... LIMIT 1).

DUCKDB DATE RULES — DuckDB is NOT PostgreSQL or SQLite. These are the ONLY
correct ways to do date arithmetic in DuckDB:
  ❌ NEVER USE: julianday(...)                         — does not exist in DuckDB
  ❌ NEVER USE: (date_a - date_b) > INTERVAL '6 years'  — DATE - DATE returns BIGINT
                                                            (days), not INTERVAL, in DuckDB
  ❌ NEVER USE: DATE_PART('day', date_a - date_b)        — fails because the
                                                            subtraction is already BIGINT,
                                                            not a DATE/INTERVAL
  ❌ NEVER USE: AGE(...), GETDATE(), NOW() for current date
  ✅ USE: date_diff('year', start_date::DATE, end_date::DATE)   for year differences
  ✅ USE: date_diff('day',  start_date::DATE, end_date::DATE)   for day differences
  ✅ USE: CURRENT_DATE                                          for "today"
  ✅ USE: date_diff('year', start_date::DATE, CURRENT_DATE) > 6  to test "more than 6 years ago"

OUTPUT THE SQL STATEMENT ONLY. Do not explain it. Do not write any words before
or after the statement. The first token of your reply must be SELECT or WITH.
"""

RETRY_PROMPT = """\
The previous SQL query failed with this error:
ERROR: {error}

PREVIOUS SQL: {prev_sql}

{schema}

USER QUERY: {query}

SPECIFIC FIX HINTS based on the error message above:
- If the error mentions "julianday" -> that function does not exist in DuckDB.
  Replace with: date_diff('year', start_col::DATE, end_col::DATE)
- If the error mentions "date_part(STRING_LITERAL, BIGINT)" -> you subtracted
  two DATEs (date_a - date_b), which produces a BIGINT (days) in DuckDB, not an
  INTERVAL or DATE. Do not feed that into date_part(). Use instead:
  date_diff('year', date_a::DATE, date_b::DATE) or
  date_diff('day',  date_a::DATE, date_b::DATE)
- If the error mentions "INTERVAL" comparison failing -> do not compare a BIGINT
  day-difference to an INTERVAL literal. Use date_diff('year', ...) > N instead.
- If the error mentions a missing/unknown column -> re-check the schema above
  and use the EXACT column name shown there.
- If the error mentions "count_star" or an unnamed aggregate column -> alias it:
  COUNT(*) AS count.
- If the error is a binder/catalog error about a table or column not existing ->
  the table/column name is wrong; match it exactly against the schema above.

Write a corrected single-line DuckDB SQL statement for the question above.
OUTPUT THE SQL STATEMENT ONLY — no explanation, no preamble, no markdown.
The first token of your reply must be SELECT or WITH.
"""

SUMMARY_PROMPT = """\
SQL query result for: "{query}"

SQL executed: {sql}

Row count: {row_count}

Result table:
{result}

Write a concise plain-language answer summarising the result.
- If the result has MORE THAN ONE ROW, present it FIRST as a markdown table
  (using the actual column names/values from the result above), THEN add a
  1-2 sentence plain-language summary below the table.
- If the result has exactly one row/one value, just state it directly in 1-2
  sentences (no table needed for a single scalar).
- Do not make up information beyond what appears in the table above.
- Round percentages to 1 decimal place and include the % sign.
- Round averages to 1-2 decimal places and include the unit (days, SGD, etc.)
  where applicable.
"""


# Matches lines that are clearly SQL continuations (used to trim chatty
# prose that some models append after the statement, e.g. "This query first...")
_SQL_CONT = re.compile(
    r"(?i)^\s*(from|where|group|order|having|join|left|right|inner|outer|full|"
    r"cross|natural|on|and|or|not|union|intersect|except|limit|offset|select|"
    r"with|as|using|when|then|else|end|case|in|like|ilike|between|is|distinct|"
    r"[(),*]|\w+\s*[=<>!])"
)


def _extract_sql(raw: str) -> str:
    """
    Isolate a single SQL statement from a possibly-chatty LLM reply.

    Some models wrap SQL in prose ("To address the error...\\nSELECT ...\\n
    This query first...") especially on retry. Only stripping code fences lets
    that prose reach DuckDB and crash it with a syntax error. This strips
    preamble, trailing explanation, fences, and semicolons — returns just the
    statement.
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

    # 3. Keep the first line, then only lines that continue the SQL.
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
        raw = self.llm.invoke([("user", prompt)]).content
        return _extract_sql(raw)

    def _retry_sql(self, query: str, schema: str, prev_sql: str, error: str) -> str:
        prompt = RETRY_PROMPT.format(
            error=error, prev_sql=prev_sql, query=query, schema=schema
        )
        raw = self.llm.invoke([("user", prompt)]).content
        return _extract_sql(raw)

    def execute_query(self, state: AgentState) -> Dict[str, Any]:
        query  = state["query"]
        schema = get_schema_context(DUCKDB_PATH)
        con    = duckdb.connect(DUCKDB_PATH)

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
            summary_prompt = SUMMARY_PROMPT.format(
                query=query, sql=sql, result=result_str, row_count=len(result_df)
            )
            summary = self.llm.invoke([("user", summary_prompt)]).content.strip()

            context_chunk = {
                "id": "s1",
                "text": f"SQL Result Summary:\n{summary}\n\nRaw data:\n{result_str}",
                "source": "Structured Datasets (DuckDB)",
                "source_type": "structured",
                "page_start": 0,
                "section": f"SQL: {sql}",
                "ce_score": 1.0,
                "rrf_score": 1.0,
                "metadata": {"sql": sql, "row_count": len(result_df)},
            }
            citation = {
                "source": "Structured Datasets",
                "segment": f"DuckDB SQL: {sql}",
                "row_count": len(result_df),
            }
            timeline = [
                f"StructuredAgent: SQL executed ({len(result_df)} rows) → {sql}"
            ]

        elif result_df is not None and result_df.empty:
            context_chunk = {
                "id": "s1",
                "text": f"SQL query returned no rows.\nSQL: {sql}",
                "source": "Structured Datasets (DuckDB)",
                "source_type": "structured",
                "page_start": 0,
                "section": "Empty result",
                "ce_score": 0.5,
                "rrf_score": 0.5,
                "metadata": {"sql": sql, "row_count": 0},
            }
            citation = {"source": "Structured Datasets", "segment": f"DuckDB SQL: {sql} (empty)"}
            timeline = [f"StructuredAgent: SQL returned 0 rows — {sql}"]

        else:
            context_chunk = {
                "id": "s1",
                "text": f"SQL execution failed after retry.\nError: {error_msg}\nSQL: {sql}",
                "source": "Structured Datasets (DuckDB)",
                "source_type": "structured",
                "page_start": 0,
                "section": "SQL error",
                "ce_score": 0.0,
                "rrf_score": 0.0,
                "metadata": {"sql": sql, "error": error_msg},
            }
            citation = {"source": "Structured Datasets", "segment": f"SQL error: {error_msg}"}
            timeline = [f"StructuredAgent: SQL FAILED — {error_msg}"]

        return {
            "retrieved_context": [context_chunk],
            "sql_queries": [sql],
            "citations": [citation],
            "execution_timeline": timeline,
        }
