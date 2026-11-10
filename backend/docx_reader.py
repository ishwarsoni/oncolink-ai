"""
docx_reader.py - Extract text from DOCX (Word) files.

A DOCX file is a ZIP archive of XML parts; ``python-docx`` parses it for us.
DOCX has no fixed pages (Word paginates at render time), so text is returned
as a single block.
"""

import io

import docx


def read_docx(file_bytes: bytes, filename: str) -> str:
    """
    Return all paragraph text from a DOCX file.

    Failures are returned as a message starting with "Error" so the document
    loader can mark the upload as unsuccessful.
    """
    try:
        # BytesIO makes the in-memory upload look like an open file to
        # python-docx, which otherwise expects a path.
        document = docx.Document(io.BytesIO(file_bytes))
        return "".join(f"{paragraph.text}\n" for paragraph in document.paragraphs)
    except Exception as error:
        return f"Error reading DOCX file '{filename}': {error}"
