# HCSA AI Knowledge Management Chatbot

Multi-agent RAG pipeline using LangGraph + Groq + Qdrant + BM25 + DuckDB.

## Architecture

```
User Query
    │
    ▼
┌─────────┐    JSON routing plan
│ Planner │──────────────────────────────────────┐
└─────────┘                                       │
    │ routes: ["structured_agent"]                │ routes: ["vector_agent"]
    │ routes: ["structured_agent","vector_agent"] │ (hybrid = both in parallel)
    ▼                                             ▼
┌──────────────────┐               ┌──────────────────────────────┐
│ Structured Agent │               │      Vector Agent             │
│ (DuckDB / SQL)   │               │  Dense (Qdrant/fastembed)     │
│ Real schema      │               │  + BM25 (rank-bm25)           │
│ Retry on error   │               │  + RRF fusion                 │
└──────────────────┘               │  + Cross-encoder rerank       │
         │                         │  + Email-count fast path      │
         │                         └──────────────────────────────┘
         │                                        │
         └──────────────┬─────────────────────────┘
                        ▼
              ┌──────────────────┐
              │     Verifier     │
              │  Synthesise      │
              │  Claim decompose │
              │  Per-claim check │
              │  Faithfulness %  │
              └──────────────────┘
                        │
                        ▼
                   Final Response
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Place your data files

```
data/
├── sop_policies/          ← Drop SOP & policy PDFs here
├── emails/                ← Drop Email_1.pdf … Email_70.pdf here
├── reports/               ← Drop HDB annual report PDFs here
└── structured_datasets/   ← Drop the 4 Excel files here
    ├── Contractor listing.xlsx
    ├── Development Projects.xlsx
    ├── Permits.xlsx
    └── Inspections.xlsx
```

### 4. Build the index (once)
```bash
python -m src.ingestion.build_index
```

Re-run with `force_rebuild=True` after adding/changing documents.

### 5. Launch the app
```bash
streamlit run app.py
```

The app also has a **Build Index Now** button if the index hasn't been built yet.

## Key design decisions

| Choice | Rationale |
|---|---|
| **fastembed** (BAAI/bge-small-en-v1.5) | No GPU needed, 384-dim, strong retrieval quality |
| **Qdrant local** | File-based, no server to run, fast cosine search |
| **BM25 + RRF** | Catches exact-match keywords (policy numbers, contractor names) that dense embeddings miss |
| **Cross-encoder reranker** | ms-marco-MiniLM-L-6-v2 — reranks top-20 candidates, very fast on CPU |
| **Email thread summary chunks** | Enables "how many emails in Email_N" counting without iterating all chunks |
| **Runtime schema introspection** | SQL agent calls `DESCRIBE <table>` on every query — no hardcoded columns |
| **SQL retry** | LLM gets its own error message fed back; fixes ~90% of first-attempt SQL errors |
| **Claim-level faithfulness** | Decomposes response → atomic claims → verifies each against evidence; replaces broken substring check |

## Metrics supported (Annex A)

| Metric | Implementation |
|---|---|
| Accuracy | Manual / eval script comparison to expected answers |
| Recall rate | Count relevant chunks retrieved vs total relevant in KB |
| Precision rate | Count relevant chunks retrieved vs total retrieved |
| Completeness rate | Key-point overlap between generated and expected answer |
| **Faithfulness score** | `supported_claims / total_claims` (per-claim LLM verification) |
