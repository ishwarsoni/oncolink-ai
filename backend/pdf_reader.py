"""
pdf_reader.py - Extract text and page numbers from PDF files.

PyMuPDF (imported as ``fitz``) reads directly from bytes, so uploads never
need a temporary file on disk.  Page numbers are preserved because RAG
citations must point at a real page.
"""

from typing import Any, Dict, List

import fitz


def read_pdf_pages(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract text from a PDF while preserving 1-based page numbers.

    Returns ``success``, ``text`` (all pages joined), ``pages``
    (``[{"page_number": int, "text": str}]``), and ``error``.
    """
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as error:
        # Scanned PDFs have no text layer, protected PDFs cannot be opened,
        # and corrupted files fail during parsing.
        return _error_result(filename, error)

    try:
        pages: List[Dict[str, Any]] = [
            {"page_number": index + 1, "text": page.get_text()}
            for index, page in enumerate(pdf_document)
        ]
        return {
            "success": True,
            "text": "".join(f"{page['text']}\n" for page in pages),
            "pages": pages,
            "error": None,
        }
    except Exception as error:
        return _error_result(filename, error)
    finally:
        pdf_document.close()


def _error_result(filename: str, error: Exception) -> Dict[str, Any]:
    """Return the standard failure shape for a PDF that could not be read."""
    return {
        "success": False,
        "text": f"Error reading PDF file '{filename}': {error}",
        "pages": [],
        "error": str(error),
    }
