"""
retriever.py — Hybrid retrieval: dense (Qdrant) + sparse (BM25) → RRF fusion
               → optional source-type filter → cross-encoder rerank.

Public API:
    retriever = HybridRetriever()
    chunks = retriever.retrieve(query, top_k=5, source_types=["sop","email"])
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from src.core.config import (
    QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_MODEL,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL, RERANKER_MODEL,
    MULTIQUERY_PER_K, MULTIQUERY_MAX,
)

BM25_CORPUS_PATH = Path(QDRANT_PATH).parent / "bm25_corpus.pkl"

# RRF constant (standard value)
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Lazy-initialised so importing the module doesn't force model loads.
    Call .retrieve() and everything spins up on first use.
    """

    def __init__(self):
        self._qdrant   = None
        self._embed    = None
        self._bm25     = None
        self._corpus   = None
        self._reranker = None
        self._ready    = False

    # ── Initialisation ────────────────────────────────────────────────────

    def _ensure_ready(self):
        if self._ready:
            return
        logger.info("[Retriever] Loading models and indexes …")

        from qdrant_client import QdrantClient
        from fastembed import TextEmbedding
        from sentence_transformers import CrossEncoder

        self._qdrant  = QdrantClient(path=QDRANT_PATH)
        self._embed   = TextEmbedding(model_name=EMBEDDING_MODEL)

        if BM25_CORPUS_PATH.exists():
            with open(BM25_CORPUS_PATH, "rb") as f:
                data = pickle.load(f)
            self._bm25   = data["index"]
            self._corpus = data["corpus"]
            logger.info(f"[Retriever] BM25 loaded: {len(self._corpus)} docs")
        else:
            logger.warning("[Retriever] BM25 corpus not found — dense-only mode")

        self._reranker = CrossEncoder(RERANKER_MODEL)
        self._ready    = True
        logger.info("[Retriever] Ready.")
        
        # ── Dense retrieval ───────────────────────────────────────────────────

    def _dense_search(
        self,
        query: str,
        top_k: int,
        source_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns list of {id, score, payload} dicts."""
        vec = list(self._embed.embed([query]))[0].tolist()

        filter_cond = None
        if source_types:
            from qdrant_client.models import Filter, FieldCondition, MatchAny

            filter_cond = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchAny(any=source_types),
                    )
                ]
            )

        response = self._qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            limit=top_k,
            query_filter=filter_cond,
            with_payload=True,
        )

        hits = response.points

        return [
            {
                "id": h.id,
                "score": h.score,
                "payload": h.payload,
            }
            for h in hits
        ]

    # ── BM25 retrieval ────────────────────────────────────────────────────

    def _bm25_search(
        self, query: str, top_k: int, source_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Apply source_type filter
        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            payload = self._corpus[idx]
            if source_types and payload.get("source_type") not in source_types:
                continue
            results.append({"id": idx, "score": float(score), "payload": payload})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── RRF fusion ────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        dense_hits: List[Dict], bm25_hits: List[Dict], k: int = RRF_K
    ) -> List[Dict]:
        """Reciprocal Rank Fusion over two ranked lists."""
        scores: Dict[int, float] = {}
        payloads: Dict[int, Any] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            doc_id = hit["id"]
            scores[doc_id]   = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            payloads[doc_id] = hit["payload"]

        for rank, hit in enumerate(bm25_hits, start=1):
            doc_id = hit["id"]
            scores[doc_id]   = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            payloads[doc_id] = payloads.get(doc_id, hit["payload"])

        fused = [
            {"id": doc_id, "rrf_score": score, "payload": payloads[doc_id]}
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return fused

    # ── Cross-encoder reranking ───────────────────────────────────────────

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not candidates:
            return []
        texts  = [c["payload"].get("text", "") for c in candidates]
        pairs  = [[query, t] for t in texts]
        ce_scores = self._reranker.predict(pairs)
        for c, s in zip(candidates, ce_scores):
            c["ce_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        return ranked[:top_k]

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        source_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieve + rerank.

        Args:
            query:        natural language question
            top_k:        number of chunks to return after reranking
            source_types: optional filter e.g. ["sop","email"] or ["report"]

        Returns:
            List of chunk dicts with keys:
            text, source, source_type, page_start, section, chunk_index,
            rrf_score, ce_score, and any metadata fields.
        """
        self._ensure_ready()

        dense_hits = self._dense_search(query, TOP_K_DENSE, source_types)
        bm25_hits  = self._bm25_search(query, TOP_K_BM25,  source_types)
        fused      = self._rrf_fuse(dense_hits, bm25_hits)
        reranked   = self._rerank(query, fused[: TOP_K_DENSE * 2], top_k)

        logger.debug(
            f"[Retriever] query='{query[:60]}' → {len(dense_hits)} dense, "
            f"{len(bm25_hits)} BM25, {len(reranked)} after rerank"
        )
        return reranked

    def retrieve_multi(
        self,
        queries: List[str],
        source_types: Optional[List[str]] = None,
        per_query_k: int = MULTIQUERY_PER_K,
        max_total: int = MULTIQUERY_MAX,
    ) -> List[Dict[str, Any]]:
        """
        Recall-oriented retrieval for multi-aspect questions.

        Runs each sub-query through the full hybrid+rerank pipeline (so each
        aspect is reranked against ITS OWN sub-query, not the whole question),
        then unions the per-sub-query top-k by chunk id. This guarantees that
        every information need in the question is represented — e.g. a question
        asking for both incident *classification* and *reporting timelines*
        yields chunks for each, instead of the timeline chunks being starved by
        the more salient classification text.

        Dedup is by chunk id (Qdrant point id == BM25 corpus index == global
        chunk index, since build_index assigns them in lockstep). On collision
        we keep the higher cross-encoder score.
        """
        self._ensure_ready()

        # De-dup the sub-queries themselves (case-insensitive) to avoid wasted work.
        seen_q, uniq_queries = set(), []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen_q:
                seen_q.add(key)
                uniq_queries.append(q.strip())
        if not uniq_queries:
            return []

        merged: Dict[Any, Dict[str, Any]] = {}
        for q in uniq_queries:
            for r in self.retrieve(q, top_k=per_query_k, source_types=source_types):
                cid = r.get("id")
                if cid not in merged or r.get("ce_score", 0.0) > merged[cid].get("ce_score", 0.0):
                    merged[cid] = r

        ranked = sorted(merged.values(), key=lambda x: x.get("ce_score", 0.0), reverse=True)
        logger.info(
            f"[Retriever] multi-query: {len(uniq_queries)} sub-queries -> "
            f"{len(ranked)} unique chunks (cap {max_total})"
        )
        return ranked[:max_total]

    def count_emails_in_thread(self, thread_id: str) -> Optional[int]:
        """
        Direct lookup for 'how many emails in Email_N.pdf' queries.
        Returns the total_in_thread value from the thread-summary chunk.
        """
        self._ensure_ready()
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        hits = self._qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="thread_id",  match=MatchValue(value=thread_id)),
                    FieldCondition(key="email_index", match=MatchValue(value=0)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        records, _ = hits
        if records:
            return records[0].payload.get("total_in_thread")
        return None