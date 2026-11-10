"""export.py - Shared PDF/TXT export helpers."""

from typing import Union


def _pdf_output_bytes(pdf) -> bytes:
    """
    Convert an FPDF document to bytes.

    fpdf2 returns ``bytearray`` from ``output()`` in recent versions and a
    latin-1 string in older ones; both are normalized to ``bytes``.
    """
    data: Union[str, bytearray, bytes] = pdf.output()
    if isinstance(data, str):
        return data.encode("latin-1")
    return bytes(data)


def _text_to_pdf_bytes(text: str) -> bytes:
    """Render plain text into PDF bytes using a monospace font."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Courier", "", 9)
    for line in text.split("\n"):
        # Core FPDF fonts are latin-1 only, so replace unsupported glyphs
        # rather than failing the whole export.
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(0, 4.5, safe_line, new_x="LMARGIN", new_y="NEXT")
    return _pdf_output_bytes(pdf)
