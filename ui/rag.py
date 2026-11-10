"""
rag.py - AI Clinical Chat (Clinical RAG Assistant) UI.

This module owns presentation and Streamlit session state only.  Ingestion,
indexing, retrieval, and answer generation live in ``application.rag`` and
``backend``; the UI calls the orchestrator and renders its results.
"""

import streamlit as st

from backend.citation_manager import page_label
from ui.components import (
    render_empty_state,
    render_pills,
    render_section,
    render_view_header,
)
from ui.session import get_intelligence_service, resolve_api_key

SUGGESTED_QUESTIONS = (
    "What is the patient's primary diagnosis?",
    "Summarize the pathology report.",
    "What biomarkers are documented?",
    "Compare the findings across uploaded reports.",
    "What treatments has the patient received?",
    "What information is missing from the documents?",
)


def _score_caption(item):
    """Describe retrieval scores without presenting distances as percentages."""
    parts = []
    score_type = item.get("score_type")

    if score_type == "cross_encoder_score" and item.get("rerank_score") is not None:
        parts.append(f"Cross-encoder score: {item['rerank_score']:.4f}")
    elif score_type == "rrf":
        parts.append(f"RRF score: {item.get('rrf_score', 0.0):.5f}")
        if item.get("semantic_rank"):
            parts.append(f"semantic rank {item['semantic_rank']}")
        if item.get("keyword_rank"):
            parts.append(f"BM25 rank {item['keyword_rank']}")

    if item.get("similarity") is not None:
        parts.append(f"cosine similarity: {item['similarity']:.4f}")
    if item.get("keyword_score") is not None:
        parts.append(f"BM25 score: {item['keyword_score']:.4f}")
    return " · ".join(parts)


def _source_page(source):
    """Return a validated page label for a citation or evidence item."""
    label = source.get("page_label")
    if label:
        return label
    return page_label(source.get("page_number"))


def _render_rag_answer(turn):
    """Render one assistant turn, including citations and evidence."""
    result = turn.get("result") or {}

    if not result.get("success"):
        st.error(result.get("error") or "The RAG pipeline could not produce an answer.")
        for warning in result.get("warnings", []):
            st.warning(warning)
        return

    answer = result.get("answer") or ""
    st.markdown(answer)

    sources = result.get("sources") or []
    contradictions = result.get("unresolved_contradictions") or []
    indicators = []
    if sources:
        indicators.append((f"{len(sources)} validated citation(s)", "info"))
    if contradictions:
        indicators.append((f"{len(contradictions)} contradiction(s)", "warn"))
    if result.get("insufficient_evidence"):
        indicators.append(("Insufficient evidence", "warn"))
    if indicators:
        render_pills(indicators)

    if result.get("insufficient_evidence"):
        st.warning(
            "The retrieved documents do not fully answer this question. The answer "
            "was not completed from general medical knowledge."
        )

    if contradictions:
        with st.expander("Conflicting evidence", expanded=True):
            for contradiction in contradictions:
                st.markdown(f"- {contradiction}")

    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=True):
            for index, source in enumerate(sources, start=1):
                st.markdown(
                    f"**[{index}] {source.get('filename', 'Unknown document')} - "
                    f"{_source_page(source)}**"
                )
                if source.get("section_heading"):
                    st.caption(f"Section: {source['section_heading']}")
                quote = source.get("quote") or ""
                if quote:
                    st.markdown(f"> {quote}")
                score = _score_caption(source)
                if score:
                    st.caption(score)
                st.divider()
    else:
        st.info("No validated source citations were returned for this answer.")

    evidence = result.get("retrieved_evidence") or []
    if evidence:
        with st.expander(f"Retrieved evidence ({len(evidence)})"):
            for item in evidence:
                st.markdown(
                    f"**{item.get('evidence_id', '')} - "
                    f"{item.get('filename', 'Unknown document')} - "
                    f"{_source_page(item)}**"
                )
                score = _score_caption(item)
                if score:
                    st.caption(score)
                if item.get("section_heading"):
                    st.caption(f"Section: {item['section_heading']}")
                st.text((item.get("text") or "")[:2000])
                st.divider()

    for warning in result.get("warnings", []):
        st.warning(warning)


def _render_index_summary(index_status, patient_id):
    """Show a compact view of the index that answers are grounded in."""
    if not index_status.success:
        st.error(f"Could not read the RAG index: {index_status.error}")
        return

    if index_status.documents:
        render_pills(
            [
                (f"{len(index_status.documents)} indexed document(s)", "ok"),
                (f"{index_status.chunk_count} chunk(s)", "neutral"),
                (f"case {patient_id}", "neutral"),
            ]
        )
        with st.expander("Indexed documents", expanded=False):
            for document in index_status.documents:
                st.markdown(
                    f"- **{document['filename']}** · "
                    f"{document['chunk_count']} chunk(s) · "
                    f"{document.get('document_date') or 'date unknown'}"
                )
        return

    render_pills([("No index yet", "neutral")])
    st.caption(
        "The first question builds the index automatically, or use the settings "
        "below to build it now."
    )


def _render_retrieval_settings(config, settings):
    """Render retrieval controls and persist the chosen settings."""
    with st.expander("Chat & retrieval settings", expanded=False):
        mode_label = st.radio(
            "Retrieval mode",
            ["Hybrid (semantic + BM25)", "Semantic only"],
            index=0 if settings.get("mode", "hybrid") == "hybrid" else 1,
            horizontal=True,
            key="rag_mode_widget",
        )
        top_k = st.slider(
            "Evidence chunks to keep (top-k)",
            min_value=1,
            max_value=10,
            value=max(1, min(10, int(settings.get("top_k", config.top_k)))),
            key="rag_top_k_widget",
        )
        use_rerank = st.checkbox(
            "Cross-encoder reranking (slower, optional)",
            value=bool(settings.get("rerank", config.rerank_enabled)),
            key="rag_rerank_widget",
        )
        threshold = st.slider(
            "Minimum cosine similarity (-1 = off)",
            min_value=-1.0,
            max_value=1.0,
            value=max(-1.0, min(1.0, float(settings.get("threshold", -1.0)))),
            step=0.05,
            key="rag_threshold_widget",
        )
        st.caption(
            "Scores shown in answers are retrieval scores, not calibrated "
            "confidence or probability of correctness."
        )

    mode = "hybrid" if mode_label.startswith("Hybrid") else "semantic"
    st.session_state["rag_settings"] = {
        "mode": mode,
        "top_k": top_k,
        "rerank": use_rerank,
        "threshold": threshold,
    }
    return mode, top_k, use_rerank, threshold


def _render_index_controls(service):
    """Render index build/update and delete controls."""
    index_col, force_col = st.columns([2, 1])
    with index_col:
        index_clicked = st.button(
            "Build / update RAG index",
            type="primary",
            key="rag_index_button",
            use_container_width=True,
        )
    with force_col:
        force_rebuild = st.checkbox(
            "Rebuild from scratch",
            value=False,
            key="rag_force_rebuild",
        )

    if index_clicked:
        with st.spinner("Chunking and embedding documents..."):
            index_result = service.ensure_rag_index(force=force_rebuild)
        if index_result.get("success"):
            st.session_state["rag_index_notice"] = (
                f"Index ready: {index_result.get('indexed', 0)} new, "
                f"{index_result.get('updated', 0)} updated, "
                f"{index_result.get('skipped', 0)} unchanged, "
                f"{index_result.get('chunks_indexed', 0)} chunks stored."
            )
            st.rerun()
        for error in index_result.get("errors", []):
            st.error(error)
        for warning in index_result.get("warnings", []):
            st.warning(warning)

    st.divider()
    st.markdown("**Remove this case's indexed data**")
    st.caption(
        "This deletes only the chunks indexed for the current case ID. "
        "Uploaded documents and the structured record are not affected."
    )
    confirm_delete = st.checkbox(
        "I understand this removes the current case's RAG index.",
        key="rag_confirm_delete",
    )
    if st.button(
        "Delete indexed data for this case",
        disabled=not confirm_delete,
        key="rag_delete_button",
    ):
        deleted = service.delete_rag_index()
        st.session_state["rag_chat"] = []
        st.session_state["rag_index_notice"] = (
            f"Deleted {deleted} chunk(s) for case {service.patient_id}."
        )
        st.rerun()


def _render_suggestions(has_chat):
    """Render the suggested questions as buttons."""
    if has_chat:
        with st.expander("Suggested questions", expanded=False):
            _render_suggestion_buttons()
        return

    render_section(
        "Suggested questions",
        "Start with a question about the documents in this case.",
    )
    _render_suggestion_buttons()


def _render_suggestion_buttons():
    """Render six suggestion buttons without calling the model."""
    columns = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        with columns[index % 2]:
            if st.button(
                question,
                key=f"rag_suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state["rag_pending_question"] = question
                st.rerun()


def _render_chat_history():
    """Render previous chat turns."""
    if not st.session_state["rag_chat"]:
        return

    for turn in st.session_state["rag_chat"]:
        with st.chat_message(turn.get("role", "assistant")):
            if turn.get("role") == "user":
                st.markdown(turn.get("content", ""))
            else:
                _render_rag_answer(turn)


def _render_clear_chat():
    """Render the clear-chat control when there is history."""
    if not st.session_state["rag_chat"]:
        return

    _spacer, control = st.columns([4, 1])
    with control:
        if st.button("Clear chat", key="rag_clear_chat", use_container_width=True):
            st.session_state["rag_chat"] = []
            st.rerun()


def _answer_question(service, question, top_k, mode, use_rerank, threshold):
    """Handle one question: retrieve, answer, and store the turn."""
    st.session_state["rag_chat"].append({"role": "user", "content": question})

    with st.spinner("Retrieving evidence and generating a grounded answer..."):
        try:
            answer_result = service.ask(
                question=question,
                top_k=top_k,
                mode=mode,
                rerank=use_rerank,
                score_threshold=None if threshold <= -1.0 else threshold,
            )
        except Exception as error:  # a transport error must not crash the view
            answer_result = {
                "success": False,
                "error": f"RAG answer generation failed: {error}",
                "warnings": [],
            }

    st.session_state["rag_chat"].append(
        {
            "role": "assistant",
            "content": answer_result.get("answer", ""),
            "result": answer_result,
        }
    )
    st.rerun()


def _handle_rag_question(service, top_k, mode, use_rerank, threshold):
    """Read the chat input or a pending suggestion and answer it."""
    typed_question = st.chat_input(
        "Ask a question about the uploaded patient documents...",
        key="rag_question_input",
    )
    pending_question = st.session_state.pop("rag_pending_question", None)
    question = typed_question or pending_question
    if not question:
        return

    _answer_question(service, question, top_k, mode, use_rerank, threshold)


def create_rag_assistant_section():
    """
    Render AI Clinical Chat (the Clinical RAG Assistant).

    The assistant uses the same canonical documents as the structured record,
    but retrieval is always filtered to the current session's case ID.
    """
    render_view_header(
        "AI Clinical Chat",
        "Ask evidence-grounded questions about this patient's documents. The "
        "Clinical RAG Assistant answers only from retrieved passages and rebuilds "
        "every citation from stored document metadata.",
    )

    if "processed_files" not in st.session_state:
        render_empty_state(
            "No documents in this case yet",
            "Upload and process clinical documents before starting a conversation.",
        )
        return

    if resolve_api_key() is None:
        st.warning(
            "NVIDIA API key not configured. You can build and inspect the RAG "
            "index, but answer generation will fail until a key is entered in "
            "the sidebar."
        )

    service = get_intelligence_service()
    components = service.rag.components
    if components is None:
        st.error(
            "The RAG knowledge base could not be initialised. "
            f"{service.rag.init_error}"
        )
        return

    notice = st.session_state.pop("rag_index_notice", None)
    if notice:
        st.success(notice)

    config = components["config"]
    settings = st.session_state.get("rag_settings", {})
    mode, top_k, use_rerank, threshold = _render_retrieval_settings(config, settings)

    index_status = service.index_status()
    _render_index_summary(index_status, service.patient_id)

    with st.expander("Index controls", expanded=False):
        _render_index_controls(service)

    _render_clear_chat()
    _render_suggestions(has_chat=bool(st.session_state["rag_chat"]))
    _render_chat_history()
    _handle_rag_question(service, top_k, mode, use_rerank, threshold)
