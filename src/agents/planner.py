"""
planner.py — Multi-route query planner with query decomposition.

Responsibilities:
  1. Route the query to structured_agent (SQL), vector_agent (docs), or BOTH.
  2. Decompose the question into focused SUB-QUERIES for retrieval. This is the
     key recall lever: a question like "what severity level AND what reporting
     timelines" is two different information needs that a single embedding cannot
     cover, so we search each aspect separately and union the results.
  3. Detect email-counting intent and extract a thread_id.
  4. Pick an optional source_type filter (sop / email / report).
"""
from __future__ import annotations

import json
import re
from typing import Dict, Any

from loguru import logger

from src.core.state import AgentState
from src.core.llm   import get_llm


SYSTEM_PROMPT = """You are the Query Routing & Decomposition Engine for HCSA's AI Knowledge System.

HCSA has FOUR knowledge sources:
  A) structured_agent — tabular data in DuckDB. Tables: contractor_listing,
     development_projects, permits, inspections. Use ONLY for counts, averages,
     distributions, rankings, statuses, ratings, budgets that come FROM THESE
     FOUR TABLES specifically.
  B) vector_agent — unstructured documents: SOP/policy PDFs (SOP-CO-003,
     POL-CO-003), HDB annual-report PDFs (FS-22, FS-23), and email PDFs.
     Use for policies, procedures, incident rules, permit rules, financial-report
     facts, AND ALL EMAIL CONTENT — including counting or aggregating emails,
     senders, or correspondence (emails are NOT in DuckDB; they only exist as
     PDFs retrieved by vector_agent).

CRITICAL ANTI-MISROUTE RULE — read carefully:
  Questions that LOOK like counting/aggregation questions are NOT automatically
  structured_agent. The deciding factor is WHERE the data lives, not whether the
  question uses the word "how many" or "count".
    - "How many contractors wrote in to request permits in 2025?" -> this counts
      EMAIL SENDERS from correspondence, not rows in the permits table ->
      routes: ["vector_agent"], source_type_filter: ["email"]
    - "How many accidents occurred in 2025?" -> incidents/accidents are reported
      via email and incident reports (IR-62), not a DuckDB table -> 
      routes: ["vector_agent"], source_type_filter: ["email"]
    - "Count the number of unique emails sent per person per topic" -> explicitly
      about emails -> routes: ["vector_agent"], source_type_filter: ["email"]
    - "How many inactive contractors are holding ongoing development projects?"
      -> this DOES map to real columns in contractor_listing + development_projects
      -> routes: ["structured_agent"]
  When in doubt: if the question's subject (permits requested, accidents,
  emails, correspondence, senders) is not one of contractor_listing,
  development_projects, permits, or inspections AS DUCKDB TABLES, route to
  vector_agent, not structured_agent. Only route to structured_agent when the
  question's quantities clearly correspond to columns that exist in those four
  tables (ratings, budgets, statuses, inspection results, completed_projects,
  defects_found, cost, dates already in the table).

ROUTING:
  - tabular data only  -> routes: ["structured_agent"]
  - documents only     -> routes: ["vector_agent"]
  - needs BOTH         -> routes: ["structured_agent","vector_agent"]
    (e.g. "permit status for project X AND the SOP that governs it")

QUERY DECOMPOSITION (search_queries) — REQUIRED when routes include vector_agent:
  Break the question into 1-5 focused retrieval queries, ONE per distinct
  information need. Rules:
    - If the question asks about two+ things (e.g. "severity level AND reporting
      timelines"), emit a SEPARATE sub-query for each thing.
    - Use the vocabulary that will appear in the TARGET document, not the user's
      narrative. For reporting timelines, write terms like
      "incident reporting timeline verbal notification escalation chain written
      report HCSA notification days", NOT "what do I do after the worker fell".
    - Copy specific names, IDs, dates, document codes, project codes, and
      contractor names VERBATIM into a sub-query — exact tokens drive keyword
      (BM25) matching (e.g. "CONTR-2022-047", "UP-2025-088", "Kelang Baru").
    - For incident questions, ALWAYS add a sub-query for the borderline /
      classification protocol (terms: "borderline incident classification
      Level 2 Level 3 potential serious injury review HSE Manager").
    - For "both financial years" questions, emit one sub-query per year.
    - For email-counting/aggregation questions, include the specific topic
      keywords named in the query (e.g. "Ecorebate", "electrical",
      "internet infrastructure", "lease") as separate sub-queries so each
      topic's emails are retrieved.
    - For a simple single-fact question, a single sub-query is fine.
  If routes are structured_agent only, set search_queries to [].

SPECIAL CASE — email counting/aggregation:
  If the query asks specifically "how many emails are in Email_N" (a single
  named thread file), set email_count_intent true and thread_id "Email_N".
  This fast-path is ONLY for counting messages within one named thread PDF.
  Broader questions like "how many contractors wrote in" or "count emails per
  person per topic" are NOT this fast path — leave email_count_intent false and
  instead route to vector_agent with source_type_filter ["email"] so the model
  reasons over the actual retrieved email chunks.

OUTPUT — valid JSON object only, no markdown:
{
  "routes": ["structured_agent"] | ["vector_agent"] | ["structured_agent","vector_agent"],
  "search_queries": ["focused query 1", "focused query 2"],
  "source_type_filter": null | ["sop"] | ["email"] | ["report"] | ["sop","email"] | ["report","sop"],
  "email_count_intent": false | true,
  "thread_id": null | "Email_N",
  "reasoning": "one sentence"
}

EXAMPLES:
Q: "A worker fell 2.5m, sprained ankle, 6h observation, 5 days modified duty.
    What is the severity level and the reporting timelines?"
{"routes":["vector_agent"],
 "search_queries":[
   "incident severity classification level criteria injury hospitalization property damage",
   "incident reporting timeline verbal notification escalation chain written report IR-62 HCSA notification days",
   "borderline incident classification Level 2 Level 3 fall from height potential serious injury review"],
 "source_type_filter":["sop"],"email_count_intent":false,"thread_id":null,
 "reasoning":"Needs classification criteria, reporting timelines, and the borderline protocol."}

Q: "Full performance summary for contractor CONTR-2022-047 (Pulau Ulu Building
    Works): inspections, failed inspections, score, and any policy issues."
{"routes":["structured_agent","vector_agent"],
 "search_queries":[
   "CONTR-2022-047 Pulau Ulu Building Works inspections incidents permits",
   "contractor safety assessment score CSA-72 prequalification threshold enhanced supervision"],
 "source_type_filter":null,"email_count_intent":false,"thread_id":null,
 "reasoning":"Structured tables for inspection/score data plus email/policy context for the contractor."}

Q: "What is the distribution of contractor ratings?"
{"routes":["structured_agent"],"search_queries":[],"source_type_filter":null,
 "email_count_intent":false,"thread_id":null,
 "reasoning":"Pure aggregation over contractor_listing."}

Q: "How many contractors wrote in to request permits in 2025?"
{"routes":["vector_agent"],
 "search_queries":[
   "contractor email permit request 2025 sender",
   "permit application correspondence email 2025"],
 "source_type_filter":["email"],"email_count_intent":false,"thread_id":null,
 "reasoning":"Asks about email senders/correspondence, not a DuckDB table column."}

Q: "How many accidents occurred in 2025?"
{"routes":["vector_agent"],
 "search_queries":["incident accident report 2025 occurrence"],
 "source_type_filter":["email"],"email_count_intent":false,"thread_id":null,
 "reasoning":"Accidents are reported via incident emails/reports, not a DuckDB table."}

Q: "Count the number of unique emails sent per person per topic. I only want
    threads relating to Ecorebate, electrical and internet infrastructural
    issues and lease."
{"routes":["vector_agent"],
 "search_queries":[
   "Ecorebate email correspondence sender",
   "electrical infrastructure issue email correspondence sender",
   "internet infrastructure issue email correspondence sender",
   "lease email correspondence sender"],
 "source_type_filter":["email"],"email_count_intent":false,"thread_id":null,
 "reasoning":"Explicit email aggregation across four named topics, not a DuckDB table."}"""


class PlannerAgent:
    def __init__(self):
        self.llm = get_llm(json_mode=True)

    def route_and_plan(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]

        try:
            response = self.llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("user", query),
            ])
            plan = json.loads(response.content)
        except Exception as exc:
            logger.warning(f"[Planner] JSON parse failed ({exc}) — defaulting to vector_agent")
            plan = {
                "routes": ["vector_agent"],
                "search_queries": [query],
                "source_type_filter": None,
                "email_count_intent": False,
                "thread_id": None,
                "reasoning": "Fallback: could not parse planner output.",
            }

        # Validate routes
        valid = {"structured_agent", "vector_agent"}
        routes = [r for r in plan.get("routes", ["vector_agent"]) if r in valid]
        if not routes:
            routes = ["vector_agent"]
        plan["routes"] = routes

        # ── Deterministic guard against email-counting misroutes ──────────
        # Even with the system-prompt rule above, the LLM occasionally still
        # sends "how many X" straight to structured_agent. These four DuckDB
        # tables never contain people/senders/correspondence/accidents, so a
        # query whose subject is one of those terms cannot be answered by SQL.
        # Force vector_agent (with an email filter) in that case.
        email_subject_terms = (
            "wrote in", "sent in", "emailed", "correspondence", "email",
            "accident", "incident occurred", "complaint", "complaints",
        )
        ql = query.lower()
        looks_email_related = any(t in ql for t in email_subject_terms)
        if looks_email_related and routes == ["structured_agent"]:
            logger.warning(
                f"[Planner] Overriding structured_agent-only route — query mentions "
                f"email/correspondence subject matter not present in DuckDB tables."
            )
            routes = ["vector_agent"]
            plan["routes"] = routes
            if not plan.get("source_type_filter"):
                plan["source_type_filter"] = ["email"]
            plan["reasoning"] = (
                (plan.get("reasoning") or "")
                + " [Overridden: subject matter (emails/correspondence/accidents) "
                "is not represented in the four DuckDB tables.]"
            )

        # ── Deterministic guard: contractor-ID deep-dives need BOTH routes ──
        # A query naming a specific contractor ID (CONTR-YYYY-NNN) alongside
        # counts/rates/averages needs the EXACT numbers from DuckDB, not just
        # the LLM's read of email/SOP context. Force structured_agent into the
        # route set if a contractor ID is present and structured_agent was
        # missed entirely.
        contractor_id_match = re.search(r"\bCONTR-\d{4}-\d{3}\b", query, re.IGNORECASE)
        if contractor_id_match and "structured_agent" not in routes:
            logger.warning(
                f"[Planner] Adding structured_agent — query names a contractor ID "
                f"({contractor_id_match.group()}) and likely needs exact DuckDB figures."
            )
            routes = routes + ["structured_agent"]
            plan["routes"] = routes

        # Validate / default search_queries. Always retrieve on the original query
        # too, so a poor decomposition can never lose the obvious match.
        sub_qs = [s.strip() for s in plan.get("search_queries", []) if isinstance(s, str) and s.strip()]
        if "vector_agent" in routes:
            if query not in sub_qs:
                sub_qs = sub_qs + [query]
        else:
            sub_qs = []
        plan["search_queries"] = sub_qs

        logger.info(
            f"[Planner] routes={routes} | sub_queries={len(sub_qs)} | "
            f"filter={plan.get('source_type_filter')} | {plan.get('reasoning','')[:70]}"
        )

        return {
            "plan": plan,
            "routes": routes,
            "execution_timeline": [
                f"Planner -> routes: {routes} | {len(sub_qs)} sub-queries | "
                f"source_filter: {plan.get('source_type_filter')} | "
                f"email_count: {plan.get('email_count_intent', False)}"
            ],
        }