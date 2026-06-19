"""
state.py — Shared AgentState flowing through LangGraph.

All list fields use Annotated[List[...], operator.add] so LangGraph
*appends* values rather than overwriting them across parallel branches.
"""
import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str                                       # original user question

    # ── Planner output ─────────────────────────────────────────────────────
    plan: Dict[str, Any]                             # full routing plan dict
    routes: List[str]                                # ["vector_agent"], or ["structured_agent","vector_agent"]

    # ── Per-agent outputs (accumulated via operator.add) ───────────────────
    retrieved_context: Annotated[List[Dict[str, Any]], operator.add]
    # Each chunk dict: {id, text, source, source_type, page, score, metadata}

    sql_queries: Annotated[List[str], operator.add]  # SQL statements executed

    # ── Synthesiser / Verifier ─────────────────────────────────────────────
    generated_response: str
    citations: Annotated[List[Dict[str, Any]], operator.add]
    confidence_score: float
    is_faithful: bool
    faithfulness_detail: Optional[Dict[str, Any]]    # per-claim breakdown

    # ── Observability ─────────────────────────────────────────────────────
    execution_timeline: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
