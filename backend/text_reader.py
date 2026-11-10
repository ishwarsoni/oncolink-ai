"""
text_reader.py - Decode uploaded TXT files.

UTF-8 is tried first because it is the modern default.  ``latin-1`` is the
fallback: it maps every byte value, so it never fails, although characters
from other encodings may look wrong.
"""


def read_txt(file_bytes: bytes, filename: str) -> str:
    """
    Decode TXT bytes to text.

    Failures are returned as a message starting with "Error" so the document
    loader can mark the upload as unsuccessful.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except Exception as error:
            return f"Error reading TXT file '{filename}': {error}"
