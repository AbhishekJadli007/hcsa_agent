"""
config.py — Single source of truth for all pipeline constants.
Edit here; nothing else hardcodes paths or model names.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent.parent  # repo root
DATA_DIR    = ROOT_DIR / "data"

# Match your ACTUAL folder names exactly (case-sensitive on macOS/Linux)
SOP_DIR     = DATA_DIR / "SOPs & Policies"
EMAIL_DIR   = DATA_DIR / "Email Repository"
REPORTS_DIR = DATA_DIR / "Reports"
STRUCT_DIR  = DATA_DIR / "Structured Datasets"

DUCKDB_PATH = str(ROOT_DIR / "hcsa_database.db")

# ── Groq LLM ──────────────────────────────────────────────────────────────
# NOTE: llama3-70b-8192 was DECOMMISSIONED by Groq. Current production 70B model
# is llama-3.3-70b-versatile (the 3.3, NOT the also-deprecated 3.1). Override via
# the GROQ_MODEL_NAME env var if you prefer e.g. openai/gpt-oss-120b (stronger) or
# llama-3.1-8b-instant (faster / higher free-tier limits).
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL      = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.0

# ── Qdrant (local in-process) ──────────────────────────────────────────────
QDRANT_PATH       = str(ROOT_DIR / "qdrant_store")
QDRANT_COLLECTION = "hcsa_knowledge"
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"   # fastembed model, no GPU needed
EMBED_DIM         = 384

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE         = 512    # characters
CHUNK_OVERLAP      = 80
MAX_CHUNKS_PER_DOC = 2000   # safety cap per document

# ── Retrieval ──────────────────────────────────────────────────────────────
# Bumped from 10/10/5: several benchmark answers (e.g. financial-risk sub-types,
# incident-classification timelines) require 4-9 supporting paragraphs, so a
# final cut of 5 caps recall too aggressively. Larger candidate pools also give
# the cross-encoder more to rerank. Lower these if you hit Groq token limits.
TOP_K_DENSE  = 20    # candidates from Qdrant dense search
TOP_K_BM25   = 20    # candidates from BM25 sparse search
TOP_K_FINAL  = 8     # final chunks after RRF + cross-encoder rerank (single-query path)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Multi-query retrieval (query decomposition) ────────────────────────────
# The planner decomposes a multi-aspect question into focused sub-queries; the
# retriever runs each one, reranks its candidates against that sub-query, then
# unions the results. This guarantees every aspect of a question is represented
# (e.g. incident *classification* AND *reporting timelines*), which is the main
# lever for recall on benchmark questions that draw on 8-13 paragraphs.
# Lower these if you hit Groq token-per-minute limits.
MULTIQUERY_PER_K = 5    # top chunks kept per sub-query (reranked against that sub-query)
MULTIQUERY_MAX   = 12   # hard cap on total unique chunks passed to the synthesiser

# ── Source type tags (stored in Qdrant payload) ────────────────────────────
SRC_SOP    = "sop"
SRC_EMAIL  = "email"
SRC_REPORT = "report"

# ── Structured DB ─────────────────────────────────────────────────────────
# Maps Excel filename stem (lowercase, spaces/underscores both OK) → DuckDB table name
EXCEL_TABLE_MAP = {
    "contractor_listing"   : "contractor_listing",
    "contractor listing"   : "contractor_listing",
    "development_projects" : "development_projects",
    "development projects" : "development_projects",
    "permits"              : "permits",
    "inspections"          : "inspections",
}