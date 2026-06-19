"""
build_index.py — Master ingestion orchestrator.

Run once (or re-run after adding new documents):
    python -m src.ingestion.build_index

What it does:
  1. Chunks all SOP PDFs           → Qdrant + BM25
  2. Parses all Email PDFs         → Qdrant + BM25
  3. Chunks all Report PDFs        → Qdrant + BM25
  4. Loads all Excel files         → DuckDB
  5. Persists BM25 corpus to disk  → bm25_corpus.pkl
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from tqdm import tqdm

from src.core.config import (
    SOP_DIR, EMAIL_DIR, REPORTS_DIR,
    QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_MODEL, EMBED_DIM,
    SRC_SOP, SRC_EMAIL, SRC_REPORT,
)
from src.ingestion.pdf_chunker   import chunk_directory, Chunk
from src.ingestion.email_parser  import parse_email_directory, EmailChunk
from src.ingestion.db_loader     import load_all_tables

# Lazy imports so build_index can be imported without optional deps loaded
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams, TokenizerType,
)
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

BM25_CORPUS_PATH = Path(QDRANT_PATH).parent / "bm25_corpus.pkl"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _chunk_to_payload(chunk) -> Dict[str, Any]:
    """Convert a Chunk or EmailChunk to a flat Qdrant payload dict."""
    if isinstance(chunk, EmailChunk):
        d = chunk.to_dict()
        meta = d.pop("metadata", {})
        d.update(meta)
        return d
    elif isinstance(chunk, Chunk):
        return chunk.to_dict()
    else:
        return dict(chunk)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer for BM25."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


# ─── Main build ───────────────────────────────────────────────────────────────

def build_all(force_rebuild: bool = False):
    """
    Full ingestion pipeline.
    Set force_rebuild=True to wipe and recreate the Qdrant collection.
    """
    logger.info("=== HCSA Ingestion Pipeline Starting ===")

    # ── 1. Collect all chunks ──────────────────────────────────────────────
    all_chunks = []

    logger.info("Chunking SOP/Policy PDFs …")
    sop_chunks = chunk_directory(SOP_DIR, source_type=SRC_SOP)
    all_chunks.extend(sop_chunks)

    logger.info("Parsing Email PDFs …")
    email_chunks = parse_email_directory(EMAIL_DIR)
    all_chunks.extend(email_chunks)

    logger.info("Chunking Annual Report PDFs …")
    report_chunks = chunk_directory(REPORTS_DIR, source_type=SRC_REPORT)
    all_chunks.extend(report_chunks)

    logger.info(f"Total chunks to index: {len(all_chunks)}")

    if not all_chunks:
        logger.warning("No chunks found — check your data/ directories contain PDF files.")
        return

    # ── 2. Build Qdrant index ─────────────────────────────────────────────
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing and force_rebuild:
        client.delete_collection(QDRANT_COLLECTION)
        logger.info(f"Deleted existing collection '{QDRANT_COLLECTION}'")

    if QDRANT_COLLECTION not in existing or force_rebuild:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        # Enable full-text index on the text payload for optional keyword pre-filter
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="source_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}'")
    else:
        logger.info(f"Collection '{QDRANT_COLLECTION}' already exists — skipping recreation (pass force_rebuild=True to reset)")
        # Still rebuild BM25 and DuckDB
        _build_bm25(all_chunks)
        load_all_tables()
        return

    # ── 3. Embed and upsert in batches ────────────────────────────────────
    embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts  = [_chunk_to_payload(c)["text"] for c in all_chunks]
    payloads = [_chunk_to_payload(c) for c in all_chunks]

    BATCH = 64
    logger.info(f"Embedding {len(texts)} chunks in batches of {BATCH} …")
    for start in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch_texts    = texts[start : start + BATCH]
        batch_payloads = payloads[start : start + BATCH]
        batch_vecs     = list(embed_model.embed(batch_texts))

        points = [
            PointStruct(
                id=start + i,
                vector=vec.tolist(),
                payload=payload,
            )
            for i, (vec, payload) in enumerate(zip(batch_vecs, batch_payloads))
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    logger.info(f"Indexed {len(texts)} vectors in Qdrant.")

    # ── 4. Build BM25 index ───────────────────────────────────────────────
    _build_bm25(all_chunks)

    # ── 5. Load structured data ───────────────────────────────────────────
    load_all_tables()

    logger.info("=== Ingestion complete ===")


def _build_bm25(all_chunks):
    bm25_corpus = [_chunk_to_payload(c) for c in all_chunks]
    tokenized   = [_tokenize(c["text"]) for c in bm25_corpus]
    bm25_index  = BM25Okapi(tokenized)
    BM25_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump({"corpus": bm25_corpus, "index": bm25_index}, f)
    logger.info(f"BM25 index persisted → {BM25_CORPUS_PATH}")


if __name__ == "__main__":
    build_all(force_rebuild=False)
