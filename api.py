"""
api.py — FastAPI wrapper exposing the HCSA agentic RAG pipeline as an HTTP API.

Place this at the repo ROOT (next to app.py). Run it WHERE YOUR INDEXES LIVE
(qdrant_store/, bm25_corpus.pkl, hcsa_database.db — i.e. your Mac):

    pip install fastapi "uvicorn[standard]"
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

A web frontend POSTs to /api/chat and renders the JSON response.
The workflow is compiled ONCE at startup so every subsequent request is fast.

NOTE: Bolt's hosted preview cannot reach http://localhost on your machine.
Expose with a tunnel first:
    ngrok http 8000  ->  https://<something>.ngrok-free.app
Then set VITE_API_BASE_URL to that URL in the Bolt environment.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Any, Dict

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from loguru import logger

from src.core.graph import compile_agentic_workflow


# ── Request / response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > 4000:
            raise ValueError("message exceeds 4000-character limit")
        return v


class Source(BaseModel):
    source: str
    source_type: str          # "sop" | "email" | "report" | "structured"
    section: str
    text: str
    score: float


class Plan(BaseModel):
    routes: List[str] = []
    search_queries: List[str] = []
    source_type_filter: Optional[List[str]] = None
    reasoning: str = ""
    email_count_intent: bool = False
    thread_id: Optional[str] = None


class Claim(BaseModel):
    claim: str
    supported: bool


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    is_faithful: bool
    sources: List[Source] = []
    plan: Plan = Plan()
    timeline: List[str] = []
    claims: List[Claim] = []
    errors: List[str] = []
    latency_ms: Optional[int] = None   # server-side wall-clock time for the full pipeline


EXAMPLES = [
    "What permits are required for working at height on a scaffold?",
    "A worker fell 2.5 m from a scaffold and sprained their ankle. What is the incident severity level and what are the reporting timelines?",
    "What were HDB's Key Audit Matters for FY 2022/23 and FY 2023/24?",
    "What is the performance summary for contractor CONTR-2022-047?",
    "How many emails are in Email_1?",
    "What is the distribution of contractor performance ratings?",
    "Summarise the main findings from the last safety inspection report.",
    "What is the policy on confined space entry permits?",
]


# ── Lifespan: compile the workflow once at startup ───────────────────────────

_workflow = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _workflow
    logger.info("[API] Compiling agentic workflow (loading models / indexes)…")
    try:
        _workflow = compile_agentic_workflow()
        logger.info("[API] Ready — workflow compiled successfully.")
    except Exception as exc:
        logger.error(f"[API] STARTUP FAILED: {exc}")
        # Don't crash the process; surface the error on the first /api/chat call.
    yield
    logger.info("[API] Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HCSA Intelligent Assistant API",
    version="1.1",
    description="Agentic RAG pipeline over HCSA safety, permit, contractor, and financial records.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your frontend domain in production
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Global exception handler — never let a 500 reach the browser without CORS ─

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"[API] Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "answer": "An unexpected server error occurred. Please try again.",
            "confidence": 0.0,
            "is_faithful": False,
            "sources": [],
            "plan": {"routes": [], "search_queries": [], "source_type_filter": None, "reasoning": "", "email_count_intent": False, "thread_id": None},
            "timeline": [],
            "claims": [],
            "errors": [str(exc)],
        },
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _initial_state(query: str) -> Dict[str, Any]:
    """Build the initial AgentState dict — keys must match state.py exactly."""
    return {
        "query":               query,
        "plan":                {},
        "routes":              [],
        "retrieved_context":   [],
        "sql_queries":         [],
        "generated_response":  "",
        "citations":           [],
        "confidence_score":    0.0,
        "is_faithful":         False,
        "faithfulness_detail": {},
        "execution_timeline":  [],
        "errors":              [],
    }


def _extract_sources(out: Dict[str, Any]) -> List[Source]:
    """Flatten retrieved_context chunks into Source objects, deduplicating by text."""
    seen_texts: set[str] = set()
    sources: List[Source] = []
    for c in out.get("retrieved_context", []) or []:
        text = str(c.get("text", "")).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        # Prefer cross-encoder score (ce_score); fall back to rrf_score.
        raw_score = c.get("ce_score") or c.get("rrf_score") or 0.0
        sources.append(Source(
            source=str(c.get("source", "unknown")),
            source_type=str(c.get("source_type", "unknown")),
            section=str(c.get("section", "")),
            text=text[:800],   # cap snippet length for the UI
            score=round(float(raw_score), 4),
        ))
    return sources


def _extract_plan(out: Dict[str, Any]) -> Plan:
    p = out.get("plan", {}) or {}
    return Plan(
        routes=p.get("routes", []) or [],
        search_queries=p.get("search_queries", []) or [],
        source_type_filter=p.get("source_type_filter"),
        reasoning=str(p.get("reasoning", "") or ""),
        email_count_intent=bool(p.get("email_count_intent", False)),
        thread_id=p.get("thread_id"),
    )


def _extract_claims(out: Dict[str, Any]) -> List[Claim]:
    fd = out.get("faithfulness_detail") or {}
    return [
        Claim(claim=str(c.get("claim", "")), supported=bool(c.get("supported", False)))
        for c in (fd.get("claims", []) or [])
        if str(c.get("claim", "")).strip()
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "workflow_ready": _workflow is not None,
    }


@app.get("/api/examples")
def examples():
    return {"examples": EXAMPLES}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _workflow is None:
        return ChatResponse(
            answer="The pipeline failed to initialise at startup. Check the server logs.",
            confidence=0.0,
            is_faithful=False,
            errors=["Workflow not initialised — see server logs for the startup error."],
        )

    query = req.message   # already stripped and validated by the Pydantic model
    logger.info(f"[API] Query: {query!r}")

    t0 = time.perf_counter()
    try:
        out = _workflow.invoke(_initial_state(query))
    except Exception as exc:
        logger.error(f"[API] Pipeline error: {exc}")
        return ChatResponse(
            answer="The pipeline encountered an error. Please try again or rephrase your question.",
            confidence=0.0,
            is_faithful=False,
            errors=[str(exc)],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"[API] Pipeline completed in {latency_ms} ms")

    answer = (out.get("generated_response") or "").strip() or "No response was generated."
    confidence = float(out.get("confidence_score") or 0.0)

    pipeline_errors: List[str] = out.get("errors", []) or []
    if pipeline_errors:
        logger.warning(f"[API] Pipeline reported {len(pipeline_errors)} error(s): {pipeline_errors}")

    return ChatResponse(
        answer=answer,
        confidence=confidence,
        is_faithful=bool(out.get("is_faithful", False)),
        sources=_extract_sources(out),
        plan=_extract_plan(out),
        timeline=out.get("execution_timeline", []) or [],
        claims=_extract_claims(out),
        errors=pipeline_errors,
        latency_ms=latency_ms,
    )
