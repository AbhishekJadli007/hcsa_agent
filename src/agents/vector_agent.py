"""
vector_agent.py — Real hybrid retrieval agent.

Replaces the mock that returned hardcoded Level-9 text.

Features:
  - Uses HybridRetriever (dense + BM25 + RRF + cross-encoder).
  - Respects source_type_filter from planner.
  - Handles email-counting intent as a special fast path (no LLM needed).
  - Formats each chunk as a structured context dict for the verifier.
"""
from __future__ import annotations

from typing import Dict, Any, List

from loguru import logger

from src.core.state        import AgentState
from src.core.config       import TOP_K_FINAL
from src.retrieval.retriever import HybridRetriever


# Module-level singleton (lazy init on first call)
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _format_chunk(chunk: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Convert a raw retriever result into the standard context dict."""
    payload = chunk.get("payload", chunk)
    meta = {}
    for k in ("sender", "recipients", "date_str", "subject", "email_index",
              "total_in_thread", "thread_id"):
        if k in payload:
            meta[k] = payload[k]

    return {
        "id": f"v{rank}",
        "text": payload.get("text", ""),
        "source": payload.get("source", "unknown"),
        "source_type": payload.get("source_type", "unknown"),
        "page_start": payload.get("page_start", 0),
        "section": payload.get("section", ""),
        "ce_score": chunk.get("ce_score", 0.0),
        "rrf_score": chunk.get("rrf_score", 0.0),
        "metadata": meta,
    }


class VectorAgent:
    def __init__(self):
        self.retriever = _get_retriever()

    def retrieve_chunks(self, state: AgentState) -> Dict[str, Any]:
        query        = state["query"]
        plan         = state.get("plan", {})
        source_types = plan.get("source_type_filter")   # None = all sources

        # ── Special fast path: email counting ─────────────────────────────
        if plan.get("email_count_intent") and plan.get("thread_id"):
            thread_id = plan["thread_id"]
            count = self.retriever.count_emails_in_thread(thread_id)
            if count is not None:
                text = (
                    f"[Email Count Result] {thread_id} contains {count} email(s) in the thread. "
                    f"Thread ID: {thread_id}."
                )
                return {
                    "retrieved_context": [
                        {
                            "id": "v0",
                            "text": text,
                            "source": f"{thread_id}.pdf",
                            "source_type": "email",
                            "page_start": 1,
                            "section": "Thread Summary",
                            "ce_score": 1.0,
                            "rrf_score": 1.0,
                            "metadata": {"total_in_thread": count, "thread_id": thread_id},
                        }
                    ],
                    "citations": [
                        {"source": f"{thread_id}.pdf", "segment": "Thread summary chunk"}
                    ],
                    "execution_timeline": [
                        f"VectorAgent: email count fast-path → {thread_id} = {count} emails"
                    ],
                }

        # ── Standard retrieval: multi-query (decomposed) for recall ───────
        search_queries = plan.get("search_queries") or [query]
        try:
            results = self.retriever.retrieve_multi(
                queries=search_queries,
                source_types=source_types,
            )
        except Exception as exc:
            logger.error(f"[VectorAgent] Retrieval failed: {exc}")
            return {
                "retrieved_context": [],
                "citations": [],
                "execution_timeline": [f"VectorAgent: retrieval error — {exc}"],
                "errors": [str(exc)],
            }

        context_chunks: List[Dict[str, Any]] = [
            _format_chunk(r, i) for i, r in enumerate(results, start=1)
        ]
        citations = [
            {
                "source": c["source"],
                "segment": c["section"] or c["source"],
                "page": c["page_start"],
                "score": round(c["ce_score"], 4),
            }
            for c in context_chunks
        ]

        logger.info(
            f"[VectorAgent] Retrieved {len(context_chunks)} chunks "
            f"(filter={source_types})"
        )

        return {
            "retrieved_context": context_chunks,
            "citations": citations,
            "execution_timeline": [
                f"VectorAgent: multi-query ({len(search_queries)} sub-queries) "
                f"dense+BM25+RRF+rerank → {len(context_chunks)} chunks "
                f"(source_filter={source_types})"
            ],
        }