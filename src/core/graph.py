"""
graph.py — LangGraph wiring for the HCSA multi-agent RAG pipeline.

Flow:
  planner
    ↓ (conditional: routes list)
  ┌─────────────────────────────────┐
  │  structured_agent  vector_agent │  (both run if hybrid, else one)
  └─────────────────────────────────┘
    ↓ (all branches merge via operator.add on retrieved_context)
  verifier
    ↓
  END

For hybrid queries, LangGraph runs both agents in parallel and the
operator.add annotation on retrieved_context accumulates both results.
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from src.core.state             import AgentState
from src.agents.planner         import PlannerAgent
from src.agents.structured_agent import StructuredAgent
from src.agents.vector_agent    import VectorAgent
from src.agents.verifier        import VerifierAgent


# ─── Routing helper ───────────────────────────────────────────────────────────

def _route_after_planner(state: AgentState) -> list[str]:
    """
    LangGraph conditional edge: return list of next node(s) to execute.
    Returning a list triggers parallel execution.
    """
    routes = state.get("routes", ["vector_agent"])
    # Map route names to actual node names (they're the same here)
    valid = {"structured_agent", "vector_agent"}
    chosen = [r for r in routes if r in valid]
    return chosen if chosen else ["vector_agent"]


# ─── Graph compiler ───────────────────────────────────────────────────────────

def compile_agentic_workflow():
    planner    = PlannerAgent()
    structured = StructuredAgent()
    vector     = VectorAgent()
    verifier   = VerifierAgent()

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("planner",          planner.route_and_plan)
    workflow.add_node("structured_agent", structured.execute_query)
    workflow.add_node("vector_agent",     vector.retrieve_chunks)
    workflow.add_node("verifier",         verifier.verify_and_generate)

    # Entry point
    workflow.set_entry_point("planner")

    # Conditional fan-out from planner (supports parallel multi-agent)
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "structured_agent": "structured_agent",
            "vector_agent":     "vector_agent",
        },
    )

    # Both agents → verifier (LangGraph merges parallel branches via state reducers)
    workflow.add_edge("structured_agent", "verifier")
    workflow.add_edge("vector_agent",     "verifier")

    # Verifier → END
    workflow.add_edge("verifier", END)

    return workflow.compile()
