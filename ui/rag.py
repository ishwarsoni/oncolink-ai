"""
rag.py - Clinical RAG assistant UI.

This module owns presentation and Streamlit session state only.  Hashing,
indexing, retrieval, and answer generation live in ``backend/``.
"""

import streamlit as st

from backend.citation_manager import page_label
from backend.config import RAGConfig
from backend.rag_indexer import (
    build_document_record,
    build_rag_components,
    document_content_hash,
    ensure_documents_indexed,
    index_signature,
    remove_patient_index,
)
from ui.session import resolve_api_key


def _get_rag_components():
    """Build and cache the RAG stack for this browser session."""
    components = st.session_state.get("rag_components")
    if components is not None:
        return components

    try:
        # Instantiate the config after load_dotenv() has run so overrides from
        # .env are picked up.
        components = build_rag_components(config=RAGConfig())
        st.session_state["rag_components"] = components
        st.session_state["rag_init_error"] = None
        return components
    except Exception as error:
        st.session_state["rag_init_error"] = str(error)
        return None


def _current_rag_documents():
    """
    Convert the processed uploads into patient-scoped RAG documents.

    Building a record hashes the full document text.  Uploads do not change
    while they stay in session state, so the result is cached per patient and
    file set instead of being rebuilt for every question.
    """
    patient_id = st.session_state.get("patient_id", "")
    processed = [
        item for item in st.session_state.get("processed_files", []) if item.get("success")
    ]
    cache_key = (
        patient_id,
        tuple((item.get("filename", ""), _cached_content_hash(item)) for item in processed),
    )

    cached = st.session_state.get("rag_documents_cache")
    if cached and cached.get("key") == cache_key:
        return cached["documents"], cached["errors"]

    documents = []
    errors = []
    for item in processed:
        try:
            documents.append(build_document_record(item, patient_id))
        except Exception as error:
            errors.append(f"{item.get('filename', 'Unknown file')}: {error}")

    st.session_state["rag_documents_cache"] = {
        "key": cache_key,
        "documents": documents,
        "errors": errors,
    }
    return documents, errors


def _cached_content_hash(item):
    """Memoize the document hash on the processed-file dict."""
    cached = item.get("_rag_content_hash")
    if cached:
        return cached
    digest = document_content_hash(item.get("text", ""))
    item["_rag_content_hash"] = digest
    return digest


def _ensure_rag_index(components, force=False):
    """
    Index the currently processed documents if needed.

    Unchanged documents are skipped by the indexer's content-hash check, so
    this is safe to call before each question.  The session cache avoids
    repeating the bookkeeping work on every rerun.
    """
    documents, document_errors = _current_rag_documents()
    if not documents:
        result = ensure_documents_indexed([], components)
        result["warnings"].extend(document_errors)
        return result

    signature = index_signature(documents)
    patient_id = st.session_state.get("patient_id", "")
    cached_result = st.session_state.get("rag_index_result")

    if (
        not force
        and cached_result
        and st.session_state.get("rag_index_signature") == signature
    ):
        try:
            if components["vector_store"].count(patient_id) > 0:
                return cached_result
        except Exception:
            # A stale cache must never block a fresh index attempt.
            pass

    result = ensure_documents_indexed(documents, components, force=force)
    result["warnings"].extend(document_errors)
    st.session_state["rag_index_result"] = result
    st.session_state["rag_index_signature"] = signature if result.get("success") else None
    return result


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


def _render_rag_answer(turn):
    """Render one assistant turn, including citations and evidence."""
    result = turn.get("result") or {}

    if not result.get("success"):
        st.error(result.get("error") or "The RAG pipeline could not produce an answer.")
        for warning in result.get("warnings", []):
            st.warning(warning)
        return

    st.markdown(result.get("answer", ""))

    if result.get("insufficient_evidence"):
        st.warning(
            "⚠️ Insufficient evidence: the retrieved documents do not fully answer "
            "this question. No answer was inferred from general medical knowledge."
        )

    contradictions = result.get("unresolved_contradictions") or []
    if contradictions:
        with st.expander("⚠️ Conflicting evidence", expanded=True):
            for contradiction in contradictions:
                st.markdown(f"- {contradiction}")

    sources = result.get("sources") or []
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})", expanded=True):
            for index, source in enumerate(sources, start=1):
                st.markdown(
                    f"**[{index}] {source.get('filename', 'Unknown document')} — "
                    f"{source.get('page_label', 'Page not available')}**"
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
        with st.expander(f"🔍 Retrieved evidence ({len(evidence)})"):
            for item in evidence:
                st.markdown(
                    f"**{item.get('evidence_id', '')} — "
                    f"{item.get('filename', 'Unknown document')} — "
                    f"{item.get('page_label', page_label(item.get('page_number')))}**"
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


def _render_rag_header():
    """Render the assistant title and short explanation."""
    st.divider()
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-top:1.5rem;">'
        '<span style="font-size:1.3rem;">💬</span>'
        '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">Clinical RAG Assistant</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem;">'
        "Ask evidence-grounded questions about the uploaded patient documents. "
        "Answers are generated only from retrieved passages and include source citations."
        "</p>",
        unsafe_allow_html=True,
    )


def _render_retrieval_settings(config, settings):
    """Render the retrieval controls and persist the chosen settings."""
    with st.expander("⚙️ Retrieval settings", expanded=False):
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


def _render_index_status(components, patient_id):
    """Show how many documents/chunks are indexed for this case."""
    try:
        indexed_documents = components["vector_store"].list_documents(patient_id)
        chunk_count = components["vector_store"].count(patient_id)
    except Exception as error:
        st.error(f"Could not read the RAG index: {error}")
        return False

    if indexed_documents:
        st.success(
            f"✅ {len(indexed_documents)} document(s) indexed · "
            f"{chunk_count} chunk(s) · case {patient_id}"
        )
        with st.expander("📚 Indexed documents"):
            for document in indexed_documents:
                st.markdown(
                    f"- **{document['filename']}** · "
                    f"{document['chunk_count']} chunk(s) · "
                    f"{document.get('document_date') or 'date unknown'}"
                )
    else:
        st.info(
            "No RAG index exists for this case yet. Click **Build / Update RAG index** "
            "or just ask a question; indexing happens automatically."
        )
    return True


def _render_index_controls(components, patient_id):
    """Render index build/update and delete controls."""
    index_col, force_col = st.columns([2, 1])
    with index_col:
        index_clicked = st.button(
            "📚 Build / Update RAG index",
            type="primary",
            use_container_width=True,
            key="rag_index_button",
        )
    with force_col:
        force_rebuild = st.checkbox(
            "Rebuild from scratch",
            value=False,
            key="rag_force_rebuild",
        )

    if index_clicked:
        with st.spinner("Chunking and embedding documents..."):
            index_result = _ensure_rag_index(components, force=force_rebuild)
        if index_result.get("success"):
            st.success(
                f"Index ready: {index_result.get('indexed', 0)} new, "
                f"{index_result.get('updated', 0)} updated, "
                f"{index_result.get('skipped', 0)} unchanged, "
                f"{index_result.get('chunks_indexed', 0)} chunks stored."
            )
        else:
            for error in index_result.get("errors", []):
                st.error(error)
        for warning in index_result.get("warnings", []):
            st.warning(warning)

    with st.expander("🗑 Remove this case's indexed data"):
        st.caption(
            "This deletes only the chunks indexed for the current case ID. "
            "It does not delete your uploaded files from this page."
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
            deleted = remove_patient_index(components["vector_store"], patient_id)
            components["hybrid_retriever"].invalidate(patient_id)
            st.session_state["rag_index_result"] = None
            st.session_state["rag_index_signature"] = None
            st.session_state["rag_chat"] = []
            st.success(f"Deleted {deleted} chunk(s) for case {patient_id}.")
            st.rerun()


def _render_chat_history():
    """Render previous chat turns and the clear-chat control."""
    st.markdown(
        '<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin:1.2rem 0 0.4rem 0;">'
        "Ask about the patient's records"
        "</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state["rag_chat"]:
        st.caption(
            "Example questions: What is the current cancer diagnosis? · "
            "What treatments were given previously? · "
            "What are the latest biomarker results? · "
            "Are there discrepancies in PD-L1 results? · "
            "Which document mentions EGFR mutation?"
        )

    for turn in st.session_state["rag_chat"]:
        with st.chat_message(turn.get("role", "assistant")):
            if turn.get("role") == "user":
                st.markdown(turn.get("content", ""))
            else:
                _render_rag_answer(turn)

    if st.session_state["rag_chat"]:
        if st.button("🧹 Clear chat", key="rag_clear_chat"):
            st.session_state["rag_chat"] = []
            st.rerun()


def _build_index_error_result(index_result):
    """Build the answer-shaped result used when indexing failed."""
    return {
        "success": False,
        "answer": "",
        "sources": [],
        "retrieved_evidence": [],
        "unresolved_contradictions": [],
        "insufficient_evidence": True,
        "warnings": index_result.get("warnings", []),
        "error": "The RAG index could not be built: "
        + "; ".join(index_result.get("errors", [])),
    }


def _handle_rag_question(components, patient_id, top_k, mode, use_rerank, threshold):
    """Handle one question: index if needed, retrieve, answer, and store."""
    question = st.chat_input(
        "Ask a question about the uploaded patient documents...",
        key="rag_question_input",
    )
    if not question:
        return

    with st.spinner("Retrieving evidence and generating a grounded answer..."):
        index_result = _ensure_rag_index(components)
        if not index_result.get("success"):
            answer_result = _build_index_error_result(index_result)
        else:
            # Reranking is chosen per question; the reranker object loads
            # lazily only when it is actually used.
            components["reranker"].enabled = bool(use_rerank)
            answer_result = components["rag_engine"].answer(
                patient_id=patient_id,
                question=question,
                top_k=top_k,
                mode=mode,
                rerank=use_rerank,
                score_threshold=None if threshold <= -1.0 else threshold,
            )

        st.session_state["rag_chat"].append({"role": "user", "content": question})
        st.session_state["rag_chat"].append(
            {
                "role": "assistant",
                "content": answer_result.get("answer", ""),
                "result": answer_result,
            }
        )
    st.rerun()


def create_rag_assistant_section():
    """
    Render the Clinical RAG Assistant.

    The assistant uses the same uploaded documents as the structured
    extraction pipeline, but retrieval is always filtered to the current
    session's patient/case ID.
    """
    if "processed_files" not in st.session_state:
        return

    _render_rag_header()

    # Retrieval works without an API key, but answer generation needs NVIDIA.
    if resolve_api_key() is None:
        st.warning(
            "⚠️ NVIDIA API key not configured. You can build and inspect the RAG "
            "index, but answer generation will fail until a key is entered in the sidebar."
        )

    components = _get_rag_components()
    if components is None:
        st.error(
            "The RAG knowledge base could not be initialised. "
            f"{st.session_state.get('rag_init_error', '')}"
        )
        return

    patient_id = st.session_state.get("patient_id", "")
    config = components["config"]
    settings = st.session_state.get("rag_settings", {})

    mode, top_k, use_rerank, threshold = _render_retrieval_settings(config, settings)

    if not _render_index_status(components, patient_id):
        return

    _render_index_controls(components, patient_id)
    _render_chat_history()
    _handle_rag_question(components, patient_id, top_k, mode, use_rerank, threshold)
