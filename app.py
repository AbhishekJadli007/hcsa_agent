"""
app.py — HCSA AI Knowledge Management Chatbot
Streamlit UI with:
  - Chat interface (left)
  - Multi-agent telemetry + faithfulness drill-down (right)
  - First-run index check with build button
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCSA Knowledge Management Chatbot",
    page_icon="🏛️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-badge {
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:0.75rem; font-weight:600; margin:2px;
}
.badge-sop     { background:#dbeafe; color:#1e40af; }
.badge-email   { background:#dcfce7; color:#166534; }
.badge-report  { background:#fef9c3; color:#854d0e; }
.badge-structured { background:#f3e8ff; color:#6b21a8; }
.claim-ok  { color:#16a34a; }
.claim-bad { color:#dc2626; }
</style>
""", unsafe_allow_html=True)

# ── Index readiness check ─────────────────────────────────────────────────────
from src.core.config import QDRANT_PATH  # noqa: E402

# Compute BM25 path the same way build_index does
from pathlib import Path as _Path
_bm25_path = _Path(QDRANT_PATH).parent / "bm25_corpus.pkl"
_qdrant_ok = (_Path(QDRANT_PATH) / "collection").exists() or any(_Path(QDRANT_PATH).iterdir()) if _Path(QDRANT_PATH).exists() else False

if not _qdrant_ok or not _bm25_path.exists():
    st.warning(
        "⚠️ Knowledge base not yet indexed. "
        "Place your PDF and Excel files in the `data/` subfolders, then click **Build Index**."
    )
    if st.button("🔨 Build Index Now"):
        with st.spinner("Indexing documents — this may take a few minutes …"):
            from src.ingestion.build_index import build_all
            build_all(force_rebuild=False)
        st.success("Index built! Refresh the page.")
        st.stop()
    st.stop()

# ── Graph (lazy init once per session) ───────────────────────────────────────
if "app_graph" not in st.session_state:
    with st.spinner("Loading agent pipeline …"):
        from src.core.graph import compile_agentic_workflow
        st.session_state.app_graph = compile_agentic_workflow()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("## 🏛️ HCSA AI Knowledge Management Chatbot")
st.caption("Ask about SOPs, emails, annual reports, contractors, projects, permits, or inspections.")
st.divider()

chat_col, telemetry_col = st.columns([1.3, 0.7], gap="large")

# ═════════════════════════════════════════════════════════════════════════════
# LEFT — Chat
# ═════════════════════════════════════════════════════════════════════════════
with chat_col:
    st.subheader("💬 Chat Interface")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                meta = msg.get("metadata", {})
                faith = meta.get("confidence_score", 1.0)
                if isinstance(faith, float) and faith < 0.5:
                    st.warning(f"⚠️ Low faithfulness score: {faith:.0%} — verify against source documents.")

    if user_query := st.chat_input("Ask a question …"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        initial_state = {
            "query": user_query,
            "plan": {},
            "routes": [],
            "retrieved_context": [],
            "sql_queries": [],
            "generated_response": "",
            "citations": [],
            "confidence_score": 0.0,
            "is_faithful": False,
            "faithfulness_detail": {},
            "execution_timeline": [],
            "errors": [],
        }

        with st.spinner("Routing query across agents …"):
            try:
                output = st.session_state.app_graph.invoke(initial_state)
                response_text = output.get("generated_response", "No response generated.")
                metadata = output
            except Exception as exc:
                response_text = f"❌ Pipeline error: {exc}"
                metadata = {}

        with st.chat_message("assistant"):
            st.markdown(response_text)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text,
            "metadata": metadata,
        })
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# RIGHT — Telemetry
# ═════════════════════════════════════════════════════════════════════════════
with telemetry_col:
    st.subheader("🛡️ Agent Telemetry")

    last = next(
        (t for t in reversed(st.session_state.chat_history) if t["role"] == "assistant"),
        None,
    )

    if not last or "metadata" not in last:
        st.info("Send a message to see pipeline telemetry here.")
    else:
        meta = last["metadata"]

        # ── Faithfulness metric ──────────────────────────────────────────
        faith_score = meta.get("confidence_score", 0.0)
        faith_label = "✅ Grounded" if faith_score >= 0.5 else "⚠️ Low Grounding"
        st.metric(
            "Faithfulness Score",
            f"{faith_score:.0%}",
            delta=faith_label,
            delta_color="normal" if faith_score >= 0.5 else "inverse",
        )

        # ── 1. Routing plan ─────────────────────────────────────────────
        with st.expander("1️⃣ Query Router Plan", expanded=True):
            plan = meta.get("plan", {})
            if plan:
                st.json(plan)
            sql_list = meta.get("sql_queries", [])
            if sql_list:
                st.markdown("**SQL Executed:**")
                for sql in sql_list:
                    st.code(sql, language="sql")

        # ── 2. Evidence chunks ───────────────────────────────────────────
        with st.expander("2️⃣ Retrieved Evidence Chunks", expanded=True):
            chunks = meta.get("retrieved_context", [])
            if not chunks:
                st.info("No evidence chunks retrieved.")
            for i, chunk in enumerate(chunks, 1):
                src_type = chunk.get("source_type", "unknown")
                badge_cls = {
                    "sop": "badge-sop",
                    "email": "badge-email",
                    "report": "badge-report",
                    "structured": "badge-structured",
                }.get(src_type, "badge-sop")

                score = chunk.get("ce_score", 0)
                st.markdown(
                    f"**Chunk {i}** — "
                    f"<span class='source-badge {badge_cls}'>{src_type.upper()}</span> "
                    f"`{chunk.get('source','?')}` | Section: *{chunk.get('section','')[:60]}* | "
                    f"Score: `{score:.3f}`",
                    unsafe_allow_html=True,
                )
                st.caption(chunk.get("text", "")[:400] + ("…" if len(chunk.get("text","")) > 400 else ""))

                # Show email metadata if present
                emeta = chunk.get("metadata", {})
                if emeta.get("sender"):
                    st.caption(
                        f"📧 From: {emeta.get('sender','')} | To: {emeta.get('recipients','')} | "
                        f"Date: {emeta.get('date_str','')} | Subject: {emeta.get('subject','')}"
                    )
                st.divider()

        # ── 3. Faithfulness drill-down ────────────────────────────────────
        with st.expander("3️⃣ Claim-Level Faithfulness", expanded=False):
            detail = meta.get("faithfulness_detail", {})
            claims = detail.get("claims", [])
            if not claims:
                st.info("No claim-level breakdown available.")
            else:
                for c in claims:
                    icon = "✅" if c.get("supported") else "❌"
                    colour = "claim-ok" if c.get("supported") else "claim-bad"
                    st.markdown(
                        f"<span class='{colour}'>{icon}</span> {c.get('claim','')}",
                        unsafe_allow_html=True,
                    )

        # ── 4. Execution timeline ─────────────────────────────────────────
        with st.expander("4️⃣ Execution Timeline", expanded=False):
            for step in meta.get("execution_timeline", []):
                st.text(f"⏱ {step}")

        # ── 5. Errors ─────────────────────────────────────────────────────
        errors = meta.get("errors", [])
        if errors:
            with st.expander("⚠️ Errors", expanded=True):
                for e in errors:
                    st.error(e)

        # ── 6. Raw state (debug) ──────────────────────────────────────────
        with st.expander("🔍 Raw State (Debug)", expanded=False):
            st.json({k: v for k, v in meta.items() if k != "retrieved_context"})
