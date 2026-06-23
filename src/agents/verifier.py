"""
verifier.py — Two-stage node: synthesis + claim-level faithfulness verification.

Stage 1 (Synthesiser):
  Generates a grounded response from the merged evidence context, following the
  HCSA response-format rules (answer-first, precise citations, tables for
  structured data, separate sections for hybrid queries, no hallucination).

Stage 2 (Verifier):
  Decomposes the response into atomic claims and checks each against the evidence
  in a SINGLE batched LLM call. faithfulness = supported_claims / total_claims.

Why batched: Groq's free tier is ~30 requests/minute. The old code made one LLM
call PER claim (10-14 calls for a typical answer), which throttles/fails on a full
benchmark run. Batching brings it to 3 calls/query (synthesis + decompose + verify).
"""
from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from loguru import logger

from src.core.state import AgentState
from src.core.llm   import get_llm


# ── HCSA response-format rules (injected as the synthesis system prompt) ──────
# Condensed from the tender's HCSA Intelligent Assistant system prompt.
HCSA_SYSTEM_PROMPT = """\
You are the HCSA (Housing and Construction Safety Authority) Intelligent Assistant.
You answer strictly from the EVIDENCE CONTEXT provided in the user message, which is
drawn from four sources: Policy/SOP PDFs, HDB annual-report PDFs, email PDFs, and
structured datasets (DuckDB/Excel).

CORE PRINCIPLE — BE COMPLETE. These answers are graded on coverage of key points,
so do NOT stop at the first relevant fact. Mine ALL of the evidence and include
every applicable criterion, threshold, timeline, requirement, and source document
that bears on the question. A thorough, well-structured answer is the goal.

RESPONSE RULES:
1. Lead with the direct answer in the first 1-2 sentences. Never open with
   "Based on the documents..." or "According to my retrieval...". If the evidence
   shows the answer is genuinely conditional or borderline, then the direct answer
   is to state that and name the candidate outcomes (see rule A).
2. Cite every factual claim to its source using these formats:
   - SOP/Policy:  [SOP-CO-003 Section X.X]  or  [POL-CO-003 Section X.X]
   - Financials:  [HDB FS-22, Note X / Page X]  (FY2022/23)
                  [HDB FS-23, Note X / Page X]  (FY2023/24)
   - Emails:      [Email XX.pdf - Sender to Recipient, Date]
   - Structured:  [Contractor listing.xlsx] / [Inspections.xlsx] / etc.
3. Present structured/tabular results as a markdown table, then a one-line summary.
   Do not bury numbers in prose.
4. For hybrid queries that use both documents AND structured data, use clear
   section headers per source type, then a Conclusion.
5. For timelines/investigations, list events in date order and attribute each fact
   to its source. Separate "what happened" from "what policy required".
6. Disambiguate financial years: FY2022/23 = year ended 31 Mar 2023 (HDB FS-22);
   FY2023/24 = year ended 31 Mar 2024 (HDB FS-23). Never mix the two. If asked
   about both years, give BOTH figures, ideally as a two-column comparison.
7. HDB and HCSA may be used interchangeably in financial contexts.
8. Do NOT invent figures, email senders, permit IDs, contractor names, thresholds,
   or section numbers. Use ONLY values that appear in the evidence (e.g. if the
   evidence says escalation is triggered by falls exceeding TWO metres, do not
   write three metres). If the evidence is insufficient, say EXACTLY:
   "I do not have sufficient information in the provided records to answer this question."
9. State what the policy says; do not give legal advice.

DOMAIN RULES:
A. INCIDENT SEVERITY CLASSIFICATION. First classify by the ACTUAL consequence
   (injury severity, hospitalization duration, property damage). Then check the
   borderline protocol (POL-CO-003 Section 5.2): if the incident TYPE carries
   potential for serious injury — falls from height, struck-by heavy equipment,
   electrical contact, chemical exposure, or structural failure — it is a
   BORDERLINE case. For borderline cases you MUST present BOTH the actual-
   consequence level AND the higher potential-consequence level, give the criteria
   for each, and state that the Contractor's HSE Manager (or equivalent) makes the
   final classification within one hour using Document IC-65. Do not collapse a
   borderline incident into a single level.
B. REPORTING TIMELINES. When asked for reporting/notification timelines, give the
   COMPLETE set for EACH applicable level: (i) verbal notification — who is
   notified, within what time, and the escalation chain; (ii) written incident
   report — which document (e.g. IR-62) and within what time; (iii) HCSA/external
   notification — within what time. If a borderline case spans two levels, give the
   timelines for both levels.
C. CROSS-SOURCE. POL-CO-003 (policy) and SOP-CO-003 (procedure) often both cover a
   topic and sometimes state DIFFERENT figures (e.g. HCSA notification within 7
   calendar days in the SOP vs 14 days in the policy). Cite BOTH and report BOTH
   figures explicitly rather than silently choosing one."""

SYNTHESIS_USER_TEMPLATE = """\
EVIDENCE CONTEXT:
{context}

USER QUESTION: {query}

Answer the question using ONLY the evidence above, following all response rules."""

DECOMPOSE_PROMPT = """\
Extract every individual factual claim made in the RESPONSE below.
Each claim must be a single, atomic, standalone statement.
Return ONLY a JSON object of this exact form (no markdown, no other text):
{{"claims": ["claim one", "claim two", "..."]}}

RESPONSE:
{response}"""

BATCH_VERIFY_PROMPT = """\
You are a strict fact-checker. Decide, for each claim, whether it is supported.
A claim is SUPPORTED if the EVIDENCE states it, OR if it merely restates a fact
from the USER-STATED SCENARIO block inside the evidence (the user's own premises
count as given facts). A claim is NOT supported if it adds a figure, threshold,
name, or detail that appears in neither. Do not rely on outside knowledge.

EVIDENCE:
{context}

CLAIMS:
{numbered_claims}

Return ONLY a JSON object of this exact form, one entry per claim, in order
(no markdown, no other text):
{{"results": [{{"index": 1, "supported": true}}, {{"index": 2, "supported": false}}]}}"""

VERIFY_ONE_PROMPT = """\
EVIDENCE:
{context}

CLAIM: {claim}

A claim is supported if the EVIDENCE states it OR it restates a fact from the
USER-STATED SCENARIO block. Reply ONLY with a JSON object:
{{"supported": true}} or {{"supported": false}}"""


def _strip_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw).strip()


def _build_context_string(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        header = (
            f"[Source: {c.get('source', '?')} | Type: {c.get('source_type', '?')} "
            f"| Section: {c.get('section', '')}]"
        )
        # Surface email metadata so the model can build precise citations.
        if c.get("source_type") == "email" and (meta.get("sender") or meta.get("date_str")):
            header += (
                f"\n[Email meta: From {meta.get('sender', '?')} "
                f"to {meta.get('recipients', '?')} on {meta.get('date_str', '?')} "
                f"| Subject: {meta.get('subject', '')}]"
            )
        parts.append(f"{header}\n{c.get('text', '')}")
    return "\n\n---\n\n".join(parts)


class VerifierAgent:
    def __init__(self):
        self.llm      = get_llm()
        self.llm_json = get_llm(json_mode=True)

    # ── Stage 1: Synthesis ────────────────────────────────────────────────
    def _synthesise(self, query: str, context_str: str) -> str:
        return self.llm.invoke([
            ("system", HCSA_SYSTEM_PROMPT),
            ("user",   SYNTHESIS_USER_TEMPLATE.format(context=context_str, query=query)),
        ]).content.strip()

    # ── Stage 2a: Claim decomposition ─────────────────────────────────────
    def _decompose(self, response: str) -> List[str]:
        try:
            raw   = self.llm_json.invoke([
                ("user", DECOMPOSE_PROMPT.format(response=response))
            ]).content
            data  = json.loads(_strip_fences(raw))
            # Groq json_object mode returns an object; accept array too for safety.
            if isinstance(data, dict):
                claims = data.get("claims", [])
            elif isinstance(data, list):
                claims = data
            else:
                claims = []
            claims = [str(c).strip() for c in claims if str(c).strip()]
            if claims:
                return claims
        except Exception as exc:
            logger.warning(f"[Verifier] Claim decompose failed: {exc}")
        # Fallback: naive sentence split.
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if len(s.strip()) > 20]

    # ── Stage 2b: Batched claim verification (1 call) ─────────────────────
    def _verify_claims_batch(self, claims: List[str], context_str: str) -> List[bool]:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
        try:
            raw  = self.llm_json.invoke([
                ("user", BATCH_VERIFY_PROMPT.format(context=context_str, numbered_claims=numbered))
            ]).content
            data = json.loads(_strip_fences(raw))
            results = data.get("results", []) if isinstance(data, dict) else []
            verdict = {int(r["index"]): bool(r.get("supported", False))
                       for r in results if "index" in r}
            if len(verdict) == len(claims):
                return [verdict.get(i, False) for i in range(1, len(claims) + 1)]
            logger.warning(
                f"[Verifier] Batch verify count mismatch "
                f"({len(verdict)} vs {len(claims)}) — falling back to per-claim"
            )
        except Exception as exc:
            logger.warning(f"[Verifier] Batch verify failed: {exc} — falling back to per-claim")
        # Fallback: per-claim (only hit when batch JSON is malformed).
        return [self._verify_one(c, context_str) for c in claims]

    def _verify_one(self, claim: str, context_str: str) -> bool:
        try:
            raw  = self.llm_json.invoke([
                ("user", VERIFY_ONE_PROMPT.format(context=context_str, claim=claim))
            ]).content
            return bool(json.loads(_strip_fences(raw)).get("supported", False))
        except Exception:
            return False

    # ── Main node ─────────────────────────────────────────────────────────
    def verify_and_generate(self, state: AgentState) -> Dict[str, Any]:
        query  = state["query"]
        chunks = state.get("retrieved_context", [])

        if not chunks:
            return {
                "generated_response": "I do not have sufficient information in the provided records to answer this question.",
                "confidence_score": 0.0,
                "is_faithful": False,
                "faithfulness_detail": {"claims": [], "supported": 0, "total": 0},
                "execution_timeline": ["Verifier: no context chunks — returning insufficient-info response"],
            }

        context_str = _build_context_string(chunks)

        # Stage 1: synthesise grounded answer.
        response = self._synthesise(query, context_str)
        logger.info(f"[Verifier] Synthesised response ({len(response)} chars)")

        # Short-circuit: model self-reported insufficient info.
        if "do not have sufficient information" in response.lower():
            return {
                "generated_response": response,
                "confidence_score": 0.0,
                "is_faithful": False,
                "faithfulness_detail": {"claims": [], "supported": 0, "total": 0},
                "execution_timeline": ["Verifier: model self-reported insufficient information"],
            }

        # Stage 2: decompose + batch-verify claims. The verification context adds
        # the user's own scenario as given premises, so the model is not penalised
        # for restating facts the user supplied (e.g. "the worker fell 2.5m").
        claims = self._decompose(response)
        logger.info(f"[Verifier] {len(claims)} claims to verify (batched)")
        verify_context = (
            f"{context_str}\n\n---\n\n"
            f"[USER-STATED SCENARIO — treat these as given facts]\n{query}"
        )
        verdicts = self._verify_claims_batch(claims, verify_context) if claims else []

        supported     = sum(1 for v in verdicts if v)
        total         = len(claims) if claims else 1
        faithfulness  = round(supported / total, 4)
        is_faithful   = faithfulness >= 0.5
        claim_results = [{"claim": c, "supported": v} for c, v in zip(claims, verdicts)]

        logger.info(f"[Verifier] Faithfulness: {faithfulness:.2%} ({supported}/{total} claims)")

        return {
            "generated_response": response,
            "confidence_score": faithfulness,
            "is_faithful": is_faithful,
            "faithfulness_detail": {
                "claims": claim_results,
                "supported": supported,
                "total": total,
            },
            "execution_timeline": [
                f"Verifier: faithfulness {faithfulness:.2%} ({supported}/{total} claims supported, batched)"
            ],
        }