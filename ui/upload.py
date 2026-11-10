"""
upload.py - File upload, document processing, and extracted-text display.
"""

import streamlit as st

from backend.document_loader import process_uploaded_file


def create_file_uploader():
    """Render the multi-file uploader and return the uploaded files."""
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem;">'
        '<span style="font-size:1.3rem;">📄</span>'
        '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">Upload Documents</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem;">Upload oncology reports to generate a structured patient summary.</p>',
        unsafe_allow_html=True,
    )

    return st.file_uploader(
        label="Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="document_uploader",
    )


def display_uploaded_files(uploaded_files):
    """Show the uploaded files, or an empty-state message."""
    if not uploaded_files:
        st.markdown(
            '<div class="placeholder-box">'
            "📂 No files uploaded yet. Use the uploader above to add documents."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"**📎 Uploaded Files ({len(uploaded_files)} total)**")
    container = st.container()
    with container:
        for uploaded_file in uploaded_files:
            size_kb = uploaded_file.size / 1024
            st.markdown(
                f'<div class="file-item">📄 {uploaded_file.name} '
                f'<span style="color: #9E9E9E; font-size: 0.8rem;">'
                f"({size_kb:.1f} KB)</span></div>",
                unsafe_allow_html=True,
            )


def create_process_button(uploaded_files):
    """Render the process button and return whether documents are processed."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        process_clicked = st.button(
            "📄 Process Documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )
        if process_clicked:
            with st.spinner("Reading documents... Please wait."):
                results = [process_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]
                _store_processed_files(results)
            st.success(f"✅ Successfully processed {len(results)} document(s)!")

    return st.session_state.get("processing_done", False)


def _store_processed_files(results):
    """Persist processed files and invalidate RAG state that depends on them."""
    st.session_state["processed_files"] = results
    st.session_state["processing_done"] = True

    # New documents make the previous index and chat context unreliable.
    st.session_state["rag_index_result"] = None
    st.session_state["rag_index_signature"] = None
    st.session_state["rag_chat"] = []


def display_extracted_text():
    """Show each processed document's extracted text in an expander."""
    if "processed_files" not in st.session_state:
        return

    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-top:1.5rem;">'
        '<span style="font-size:1.3rem;">📖</span>'
        '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">Extracted Text</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem; margin-bottom:1rem;">'
        "Below is the raw text extracted from each document. Click to expand.</p>",
        unsafe_allow_html=True,
    )

    for result in st.session_state["processed_files"]:
        status_icon = "✅" if result["success"] else "❌"
        with st.expander(f"{status_icon} {result['filename']} ({result['file_type']})"):
            if not result["success"]:
                st.error(result["text"])
            elif result["text"].strip():
                st.text(result["text"])
            else:
                st.warning(
                    "This file contains no extractable text. It may be a scanned document."
                )

    st.info(
        "💡 Documents are ready. Use the 'Run AI Extraction' section below to extract "
        "structured clinical data."
    )
