"""
upload.py - Document intake, the Uploaded Documents view, and text display.

Reading files, building canonical documents and recording ingest status live
in ``application.ingestion`` / ``application.intelligence``.  This module only
renders the uploader, starts the ingestion workflow, and shows what was read.
"""

import streamlit as st

from ui.components import (
    esc,
    render_empty_state,
    render_file_row,
    render_pills,
    render_section,
    render_view_header,
)
from ui.session import get_intelligence_service


def create_file_uploader():
    """Render the multi-file uploader and return the uploaded files."""
    render_section(
        "Add documents",
        "PDF, DOCX, or TXT oncology records. Files are read once and shared by "
        "the structured record and the chat index.",
    )
    return st.file_uploader(
        label="Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="document_uploader",
        label_visibility="collapsed",
    )


def display_uploaded_files(uploaded_files):
    """Show the files selected in the uploader, or a short empty state."""
    if not uploaded_files:
        st.caption("No new files selected.")
        return

    st.markdown(f"**Selected files ({len(uploaded_files)})**")
    for uploaded_file in uploaded_files:
        size_kb = uploaded_file.size / 1024
        st.markdown(
            '<div class="ol-file-row">'
            f'<div class="ol-file-name">{esc(uploaded_file.name)}</div>'
            f'<div class="ol-file-meta">{size_kb:.1f} KB</div>'
            '<span class="ol-pill ol-pill--neutral">Ready</span>'
            "</div>",
            unsafe_allow_html=True,
        )


def create_process_button(uploaded_files):
    """Render the process button and return whether documents are processed."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        process_clicked = st.button(
            "Process documents",
            type="primary",
            key="process_documents_button",
            use_container_width=True,
            disabled=not uploaded_files,
        )
        if process_clicked:
            service = get_intelligence_service()
            with st.spinner("Reading documents... Please wait."):
                ingestion = service.ingest(uploaded_files)
                _store_processed_files(ingestion)

            st.session_state["processing_notice"] = {
                "documents": len(ingestion.documents),
                "errors": list(ingestion.errors),
                "warnings": list(ingestion.warnings),
            }
            st.rerun()

    return st.session_state.get("processing_done", False)


def render_processing_notice():
    """Show the result of the most recent ingestion exactly once."""
    notice = st.session_state.pop("processing_notice", None)
    if not notice:
        return

    documents = notice.get("documents", 0)
    if documents:
        st.success(f"Processed {documents} document(s) for this case.")
    else:
        st.error("No readable documents were found in this upload.")
    for error in notice.get("errors", []):
        st.error(error)
    for warning in notice.get("warnings", []):
        st.warning(warning)


def _store_processed_files(ingestion):
    """Persist display records and invalidate state that depends on them."""
    st.session_state["processed_files"] = list(ingestion.display_records)
    st.session_state["processing_done"] = True

    # New documents make the previous extraction, index and chat unreliable.
    for key in (
        "extraction_result",
        "per_doc_results",
        "extraction_done",
        "harmonized_data",
        "conflicts",
        "summary",
        "harmonization_done",
    ):
        st.session_state[key] = None
    st.session_state["rag_chat"] = []


def render_documents_intake():
    """Render the first-run document intake page."""
    render_view_header(
        "Uploaded Documents",
        "Start a patient case by adding its clinical documents. The workspace "
        "then builds a structured record and a patient-scoped evidence index "
        "from the same canonical document set.",
    )
    render_pills(
        [
            ("1 - Upload", "info"),
            ("2 - Process", "info"),
            ("3 - Patient Record / AI Chat", "neutral"),
        ]
    )

    uploaded_files = create_file_uploader()
    display_uploaded_files(uploaded_files)
    create_process_button(uploaded_files)

    render_empty_state(
        "Nothing processed yet",
        "After processing, this view lists each document, its extracted text, "
        "and the page boundaries used by citations.",
        "The document set is shared by both workspaces; switching views never "
        "re-uploads or re-parses a file.",
    )


def render_documents_view():
    """Render the Uploaded Documents workspace for an active case."""
    render_view_header(
        "Uploaded Documents",
        "The canonical documents for this patient case. The same files feed the "
        "structured record and the chat evidence index.",
    )

    records = list(st.session_state.get("processed_files") or [])
    if not records:
        render_empty_state(
            "No documents in this case",
            "Add and process documents to populate the workspace.",
        )
        return

    successful = [record for record in records if record.get("success")]
    failed = len(records) - len(successful)
    total_pages = sum(
        1
        for record in successful
        for page in record.get("pages") or []
        if page.get("page_number") is not None
    )
    pills = [
        (f"{len(successful)} readable document(s)", "ok"),
        (f"{total_pages} paginated page(s)", "neutral"),
    ]
    if failed:
        pills.append((f"{failed} failed file(s)", "bad"))
    render_pills(pills)

    render_section("Case documents")
    for record in records:
        page_count = sum(
            1
            for page in record.get("pages") or []
            if page.get("page_number") is not None
        )
        pages_label = (
            f"{page_count} page(s)" if page_count else "Page count unavailable"
        )
        render_file_row(
            filename=record.get("filename", "Unknown"),
            file_type=record.get("file_type", "Document"),
            pages=pages_label,
            status="Processed" if record.get("success") else "Failed",
            tone="ok" if record.get("success") else "bad",
        )

    with st.expander("Add more documents", expanded=False):
        uploaded_files = create_file_uploader()
        display_uploaded_files(uploaded_files)
        create_process_button(uploaded_files)

    display_extracted_text()


def display_extracted_text():
    """Show each processed document's extracted text and page boundaries."""
    if "processed_files" not in st.session_state:
        return

    render_section(
        "Extracted Text",
        "Raw text extracted from each document. PDF page numbers are preserved; "
        "DOCX and TXT report page availability honestly.",
    )

    for result in st.session_state["processed_files"]:
        status = "Read" if result["success"] else "Failed"
        with st.expander(
            f"{result['filename']} · {result['file_type']} · {status}",
            expanded=False,
        ):
            if not result["success"]:
                st.error(result["text"])
                continue

            pages = result.get("pages") or []
            paginated = [
                page for page in pages if page.get("page_number") is not None
            ]
            if paginated:
                st.caption(
                    "Pages: "
                    + ", ".join(str(page["page_number"]) for page in paginated)
                )
            else:
                st.caption("Page numbers are not available for this file type.")

            if result["text"].strip():
                st.text(result["text"])
            else:
                st.warning(
                    "This file contains no extractable text. It may be a scanned "
                    "document without a text layer."
                )

    st.caption(
        "Use Patient Record to extract structured clinical data from these "
        "documents, or AI Clinical Chat to ask questions grounded in them."
    )
