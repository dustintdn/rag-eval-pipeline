import json
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from chain.qa_chain import ask
from eval.runner import EVAL_LOGS_DIR, run_eval
from ingest.chunker import chunk_documents
from ingest.embedder import embed_and_store
from ingest.loader import load_file
from prompts.registry import list_versions

DEFAULT_DATASET = Path("eval/sample_dataset.json")

st.set_page_config(page_title="RAG Eval Pipeline", layout="wide")
st.title("RAG Eval Pipeline")

tab_ingest, tab_qa, tab_eval = st.tabs(["Ingest", "Q&A", "Eval Dashboard"])

# ── Ingest ──────────────────────────────────────────────────────────────────
with tab_ingest:
    st.header("Document Ingestion")
    uploaded = st.file_uploader(
        "Upload a PDF, TXT, or MD file",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not uploaded):
        for uf in uploaded:
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            with st.spinner(f"Processing {uf.name}…"):
                docs = load_file(tmp_path)
                chunks = chunk_documents(docs)
                count = embed_and_store(chunks)
                Path(tmp_path).unlink(missing_ok=True)
            st.success(f"{uf.name} → {count} chunks added")

# ── Q&A ─────────────────────────────────────────────────────────────────────
with tab_qa:
    st.header("Ask a Question")
    question = st.text_input("Question")
    prompt_version = st.selectbox("Prompt version", list_versions())
    if st.button("Ask", disabled=not question):
        with st.spinner("Thinking…"):
            result = ask(question, prompt_version=prompt_version)
        st.caption(f"Prompt: `{result.prompt_version}`")
        st.markdown(f"**Answer:** {result.answer}")
        with st.expander(f"Source chunks ({len(result.source_documents)})"):
            for i, doc in enumerate(result.source_documents, 1):
                st.markdown(f"**Chunk {i}** — `{doc.metadata.get('source_file', 'unknown')}`")
                st.text(doc.page_content[:500])

# ── Eval Dashboard ───────────────────────────────────────────────────────────
with tab_eval:
    st.header("Eval Dashboard")

    col_run, col_compare = st.columns([1, 2])

    with col_run:
        if st.button("Run Eval"):
            if not DEFAULT_DATASET.exists():
                st.error("No sample_dataset.json found in eval/")
            else:
                with st.spinner("Running evaluation…"):
                    run_id, results = run_eval(DEFAULT_DATASET)
                st.success(f"Run complete: `{run_id}`")

    # List available runs
    run_files = sorted(EVAL_LOGS_DIR.glob("*_results.json"), reverse=True) if EVAL_LOGS_DIR.exists() else []
    run_ids = [f.stem.replace("_results", "") for f in run_files]

    if run_ids:
        selected = st.selectbox("View run", run_ids)
        run_path = EVAL_LOGS_DIR / f"{selected}_results.json"
        data = json.loads(run_path.read_text())
        scores = data["scores"]

        st.subheader("Scores")
        st.dataframe(
            {"Metric": list(scores.keys()), "Score": [round(v, 4) for v in scores.values()]},
            use_container_width=True,
        )

        fig = go.Figure(go.Bar(x=list(scores.keys()), y=list(scores.values())))
        fig.update_layout(yaxis_range=[0, 1], title="Metric Scores")
        st.plotly_chart(fig, use_container_width=True)

        # Side-by-side comparison
        st.subheader("Compare two runs")
        if len(run_ids) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                run_a = st.selectbox("Run A", run_ids, key="a")
            with c2:
                run_b = st.selectbox("Run B", run_ids, index=1, key="b")

            if run_a != run_b:
                scores_a = json.loads((EVAL_LOGS_DIR / f"{run_a}_results.json").read_text())["scores"]
                scores_b = json.loads((EVAL_LOGS_DIR / f"{run_b}_results.json").read_text())["scores"]
                metrics = list(scores_a.keys())
                fig2 = go.Figure([
                    go.Bar(name=run_a, x=metrics, y=[scores_a[m] for m in metrics]),
                    go.Bar(name=run_b, x=metrics, y=[scores_b[m] for m in metrics]),
                ])
                fig2.update_layout(barmode="group", yaxis_range=[0, 1], title="Run Comparison")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run at least two evals to enable comparison.")
    else:
        st.info("No eval runs yet. Click 'Run Eval' to start.")
