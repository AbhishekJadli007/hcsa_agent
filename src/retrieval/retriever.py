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

# ── Multi-paragraph recall knobs (token cost lives here) ──────────────────────
# NEIGHBOR_WINDOW: after picking the best chunks, also pull each one's ±N
#   contiguous same-document siblings. This stitches a section that got split
#   across several small chunks (e.g. a Key Audit Matters block) back together,
#   so an answer spanning adjacent chunks isn't truncated to whichever single
#   chunk ranked highest. Use 2 for 512-char chunks (matters sit ~2 chunks
#   apart); drop to 1 if you raise CHUNK_SIZE to ~1024, to save tokens.
# MULTIQUERY_HARD_CEILING: absolute cap on chunks returned AFTER expansion, so a
#   wide window can't blow the LLM context / Groq token budget. Trimmed by
#   cross-encoder score, so the most relevant anchors keep their neighbours.
NEIGHBOR_WINDOW = 2
MULTIQUERY_HARD_CEILING = 22


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

    # ── Neighbour expansion (multi-paragraph recall) ──────────────────────

    def _expand_neighbors(
        self, chunks: List[Dict[str, Any]], window: int = NEIGHBOR_WINDOW
    ) -> List[Dict[str, Any]]:
        """
        For each chunk, pull its ±window contiguous siblings from the SAME source
        document and merge them in. Point-ids are sequential per build_index, so
        id±d are the neighbouring chunks; we guard on `source` so a neighbour is
        only kept if it belongs to the same document as the anchor that wanted it
        (this stops us bleeding across a document boundary at the id seam).

        Newly-pulled neighbours inherit a score just below their anchor, so they
        sort directly after it and the final ceiling trims neighbours of the
        weakest anchors first.
        """
        if window <= 0 or not chunks:
            return chunks

        have = {c["id"] for c in chunks if isinstance(c.get("id"), int)}
        want_src: Dict[int, Any] = {}    # neighbour id -> source it MUST match
        want_score: Dict[int, float] = {}  # neighbour id -> anchor ce_score
        for c in chunks:
            cid = c.get("id")
            if not isinstance(cid, int):
                continue
            src = (c.get("payload") or {}).get("source")
            sc  = c.get("ce_score", 0.0)
            for d in range(1, window + 1):
                for nid in (cid - d, cid + d):
                    if nid >= 0 and nid not in have:
                        want_src[nid] = src
                        want_score[nid] = max(want_score.get(nid, -1e9), sc)

        if not want_src:
            return chunks

        try:
            recs = self._qdrant.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=sorted(want_src.keys()),
                with_payload=True,
            )
        except Exception as exc:
            logger.warning(f"[Retriever] neighbour fetch failed: {exc}")
            return chunks

        added = 0
        for rec in recs:
            nid = rec.id
            payload = rec.payload or {}
            if payload.get("source") != want_src.get(nid):
                continue  # different document at the id seam — skip
            chunks.append({
                "id": nid,
                "payload": payload,
                "rrf_score": 0.0,
                "ce_score": want_score.get(nid, 0.0) - 0.001,
                "is_neighbor": True,
            })
            added += 1

        if added:
            logger.info(
                f"[Retriever] neighbour expansion: +{added} contiguous chunks "
                f"(window={window})"
            )
        return chunks

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
        then selects chunks with GUARANTEED per-sub-query coverage and stitches
        contiguous sections back together via neighbour expansion.

        Why round-robin instead of a global score sort: a global top-N by
        cross-encoder score lets the most salient aspect monopolise the slots —
        e.g. the FY2022/23 Key Audit Matters chunks crowd out the FY2023/24 ones,
        so the second year contributes nothing. Round-robin takes the #1 hit from
        every sub-query, then the #2 from every sub-query, etc., so each
        information need is represented before the cap is reached.

        Why neighbour expansion: an answer can span several adjacent chunks (both
        Key Audit Matters sit in contiguous chunks). Once any chunk in that block
        is selected, pulling its id±window siblings recovers the rest of the block
        even though those siblings ranked below the cutoff on their own.

        Dedup is by chunk id (Qdrant point id == BM25 corpus index == global chunk
        index, since build_index assigns them in lockstep). On collision we keep
        the higher cross-encoder score.
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

        # 1. Run every sub-query; keep its ranked hits separately, and track the
        #    best-scoring copy of each chunk for dedup.
        per_query_hits: List[List[Dict[str, Any]]] = []
        best: Dict[Any, Dict[str, Any]] = {}
        for q in uniq_queries:
            hits = self.retrieve(q, top_k=per_query_k, source_types=source_types)
            per_query_hits.append(hits)
            for r in hits:
                cid = r.get("id")
                if cid not in best or r.get("ce_score", 0.0) > best[cid].get("ce_score", 0.0):
                    best[cid] = r

        # 2. Round-robin selection — guarantees every sub-query contributes.
        selected: Dict[Any, Dict[str, Any]] = {}
        depth = 0
        while len(selected) < max_total:
            progressed = False
            for hits in per_query_hits:
                if depth < len(hits):
                    cid = hits[depth]["id"]
                    if cid not in selected:
                        selected[cid] = best[cid]
                        progressed = True
                        if len(selected) >= max_total:
                            break
            if not progressed:
                break
            depth += 1

        chosen = list(selected.values())
        anchors = len(chosen)

        # 3. Stitch contiguous sections back together.
        chosen = self._expand_neighbors(chosen)

        # 4. Order by relevance; neighbours sit just under their anchor. Hard
        #    ceiling guards the token budget when the window is wide.
        chosen.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
        chosen = chosen[:MULTIQUERY_HARD_CEILING]

        logger.info(
            f"[Retriever] multi-query: {len(uniq_queries)} sub-queries -> "
            f"{anchors} anchors (round-robin, cap {max_total}) -> "
            f"{len(chosen)} chunks after neighbour expansion (ceiling {MULTIQUERY_HARD_CEILING})"
        )
        return chosen

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