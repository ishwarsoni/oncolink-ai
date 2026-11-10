"""
document_loader.py - Detect an uploaded file's type and route it to a reader.

The Streamlit layer calls ``process_uploaded_file`` and receives a uniform
dictionary regardless of file format:

    {"filename": str, "text": str, "file_type": str, "success": bool,
     "pages": [{"page_number": int | None, "text": str}]}

Adding a new format only requires a new branch here and a reader module.
"""

from pathlib import Path
from typing import Any, Dict, List

from backend.docx_reader import read_docx
from backend.pdf_reader import read_pdf_pages
from backend.text_reader import read_txt

# Human-readable file type labels used by the UI.
_FILE_TYPES = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".txt": "TXT",
}


def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Read one uploaded file and return text plus page metadata."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        text, pages, success = _read_pdf(file_bytes, filename)
    elif extension in (".docx", ".txt"):
        text, pages, success = _read_plain_document(file_bytes, filename, extension)
    else:
        text = (
            f"Unsupported file format: '{extension}'. "
            "Please upload PDF, DOCX, or TXT files."
        )
        pages = []
        success = False

    # Readers report failures as text so the UI can display the reason.
    if text.startswith("Error"):
        success = False

    return {
        "filename": filename,
        "text": text,
        "file_type": _FILE_TYPES.get(extension, "UNKNOWN"),
        "success": success,
        "pages": pages,
    }


def _read_pdf(file_bytes: bytes, filename: str) -> tuple[str, List[Dict], bool]:
    """Read a PDF with page numbers, which RAG citations rely on."""
    pdf_result = read_pdf_pages(file_bytes, filename)
    return pdf_result["text"], pdf_result["pages"], bool(pdf_result["success"])


def _read_plain_document(
    file_bytes: bytes, filename: str, extension: str
) -> tuple[str, List[Dict], bool]:
    """
    Read a DOCX or TXT file as a single logical page.

    Neither format stores reliable page numbers, so ``page_number`` is
    ``None`` and citations display "Page not available".
    """
    reader = read_docx if extension == ".docx" else read_txt
    text = reader(file_bytes, filename)
    pages = [{"page_number": None, "text": text}] if text and not text.startswith("Error") else []
    return text, pages, True
