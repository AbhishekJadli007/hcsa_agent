"""
planner.py — Multi-route query planner with query decomposition.

Responsibilities:
  1. Route to structured_agent (SQL over DuckDB), vector_agent (docs), or BOTH.
  2. Decompose multi-aspect questions into focused SUB-QUERIES for retrieval
     (the recall lever: each information need is searched separately).
  3. Route correspondence COUNTING/AGGREGATION to structured_agent's `emails`
     table — vector search cannot count across a corpus.
  4. Detect single-thread email-counting ("how many emails in Email_N").
"""
from __future__ import annotations

import json
from typing import Dict, Any

from loguru import logger

from src.core.state import AgentState
from src.core.llm   import get_llm


SYSTEM_PROMPT = """You are the Query Routing & Decomposition Engine for HCSA's AI Knowledge System.

HCSA has FOUR knowledge sources:
  A) structured_agent — tabular data in DuckDB:
       - contractor_listing, development_projects, permits, inspections
       - emails  (ONE ROW PER EMAIL: sender, recipients, date_str, email_date,
                  email_year, subject, body) — use this to COUNT or AGGREGATE
                  correspondence.
     Use for counts, averages, distributions, rankings, statuses, ratings, budgets,
     date math, AND for "how many emails / who wrote in / who sent the most /
     count emails per person or per topic / how many incidents reported".
  B) vector_agent — unstructured documents: SOP/policy PDFs (SOP-CO-003,
     POL-CO-003), HDB annual-report PDFs (FS-22, FS-23), and the email text.
     Use for policies, procedures, incident rules, permit rules, financial-report
     facts, and reading/quoting email CONTENT (not counting it).

ROUTING:
  - tabular data or correspondence COUNTING only -> ["structured_agent"]
  - documents / reading content only             -> ["vector_agent"]
  - needs BOTH                                    -> ["structured_agent","vector_agent"]

  COUNTING RULE: if the question asks HOW MANY / HOW OFTEN / WHO SENT THE MOST /
  COUNT ... PER ... over emails or incidents, route to structured_agent and use the
  emails table. Do NOT send counting questions to vector_agent — it cannot count.

QUERY DECOMPOSITION (search_queries) — REQUIRED when routes include vector_agent:
  Break the question into 1-5 focused retrieval queries, ONE per information need.
    - Separate sub-query per distinct thing asked (e.g. "severity level" vs
      "reporting timelines").
    - Use target-document vocabulary, not the user's narrative (for timelines:
      "incident reporting timeline verbal notification escalation chain written
      report IR-62 HCSA notification days").
    - Copy specific names, IDs, dates, document/project codes VERBATIM into a
      sub-query (exact tokens drive BM25): "CONTR-2022-047", "UP-2025-088",
      "Kelang Baru", "Tanjung Pagar".
    - EXPAND domain abbreviations in sub-queries (the documents spell terms out in
      full, so abbreviations will NOT keyword-match). Keep one sub-query with the
      original wording too. Known expansions:
        GD = Group Director;  HMG = Housing Management Group / Housing Management;
        FS = Financial Statements;  AR = Annual Report;  CRO = Chief Risk Officer;
        PTW = Permit to Work;  WAH = Working at Height;  CSE = Confined Space Entry;
        HSE = Health, Safety and Environment;  KAM = Key Audit Matters;
        CSA = Contractor Safety Assessment.
      Example: "Who was the GD of HMG?" -> sub-query "Group Director Housing Management".
    - For incident questions ALWAYS add a borderline-classification sub-query.
    - For "both financial years" emit one sub-query per year.
  If routes are structured_agent only, set search_queries to [].

SINGLE-THREAD email count: if asked "how many emails are in Email_N", set
email_count_intent true and thread_id "Email_N".

OUTPUT — valid JSON object only, no markdown:
{
  "routes": [...],
  "search_queries": [...],
  "source_type_filter": null | ["sop"] | ["email"] | ["report"] | ["sop","email"] | ["report","sop"],
  "email_count_intent": false | true,
  "thread_id": null | "Email_N",
  "reasoning": "one sentence"
}

EXAMPLES:
Q: "Who was the GD of HMG in FY 22/23?"
{"routes":["vector_agent"],
 "search_queries":["Group Director Housing Management","Group Director Housing Management Group FY2022/23","Who was the GD of HMG in FY 22/23?"],
 "source_type_filter":["report"],"email_count_intent":false,"thread_id":null,
 "reasoning":"Leadership name from the annual report; GD=Group Director, HMG=Housing Management Group."}

Q: "How many contractors wrote in to request permits in 2025?"
{"routes":["structured_agent"],"search_queries":[],"source_type_filter":null,
 "email_count_intent":false,"thread_id":null,
 "reasoning":"Count of distinct email senders in 2025 requesting permits — use the emails table."}

Q: "A worker fell 2.5m, sprained ankle. What is the severity level and reporting timelines?"
{"routes":["vector_agent"],
 "search_queries":[
   "incident severity classification level criteria injury hospitalization property damage",
   "incident reporting timeline verbal notification escalation chain written report IR-62 HCSA notification days",
   "borderline incident classification Level 2 Level 3 fall from height potential serious injury review"],
 "source_type_filter":["sop"],"email_count_intent":false,"thread_id":null,
 "reasoning":"Needs classification criteria, reporting timelines, and the borderline protocol."}

Q: "Performance summary for CONTR-2022-047: inspections, failed inspections, score, policy issues."
{"routes":["structured_agent","vector_agent"],
 "search_queries":[
   "CONTR-2022-047 Pulau Ulu Building Works performance review outcome suspension",
   "contractor safety assessment score CSA-72 threshold suspension contract review"],
 "source_type_filter":null,"email_count_intent":false,"thread_id":null,
 "reasoning":"Structured tables for exact inspection/cost/delay stats plus policy/email context."}"""


class PlannerAgent:
    def __init__(self):
        self.llm = get_llm(json_mode=True)

    def route_and_plan(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]

        try:
            response = self.llm.invoke([("system", SYSTEM_PROMPT), ("user", query)])
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

        valid = {"structured_agent", "vector_agent"}
        routes = [r for r in plan.get("routes", ["vector_agent"]) if r in valid]
        if not routes:
            routes = ["vector_agent"]
        plan["routes"] = routes

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