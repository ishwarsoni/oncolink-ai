"""
=============================================================================
citation_manager.py - Evidence citations that always point at real chunks
=============================================================================

The LLM is allowed to say which evidence IDs support its answer, but it is
never trusted to invent filenames or page numbers.  Every citation is
rebuilt from the retrieved chunk metadata.  If the model cites an evidence
ID that was not retrieved, the citation is dropped and a warning is shown.

Page numbers:
    PDFs provide real 1-based page numbers.  DOCX and TXT documents do not
    contain reliable page information, so their chunks use -1 internally and
    are displayed as "Page not available".
=============================================================================
"""

import logging
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)


def page_label(page_number) -> str:
    """Return a human-readable page label without inventing a page number."""
    if page_number is None or page_number == "":
        return "Page not available"
    try:
        number = int(page_number)
    except (TypeError, ValueError):
        return "Page not available"
    if number < 0:
        return "Page not available"
    return f"Page {number}"


def build_evidence_index(results: Iterable[Dict]) -> Dict[str, Dict]:
    """
    Assign stable evidence IDs (E1, E2, ...) to retrieved chunks.

    The index is keyed by evidence ID so the LLM can reference [E1] etc.
    """
    index: Dict[str, Dict] = {}
    for number, item in enumerate(results or [], start=1):
        evidence_id = f"E{number}"
        entry = dict(item)
        entry["evidence_id"] = evidence_id
        entry["page_label"] = page_label(item.get("page_number"))
        index[evidence_id] = entry
    return index


def build_citations(sources, evidence_index: Dict[str, Dict]) -> Dict:
    """
    Convert the model's source references into validated citations.

    Parameters:
        sources: The "sources" list returned by the model.  Each item may
            contain evidence_id, chunk_id, filename, page_number and quote.
        evidence_index: The E1/E2/... map built from retrieved chunks.

    Returns:
        dict with citations (list), warnings (list), dropped (int).
    """
    citations: List[Dict] = []
    warnings: List[str] = []
    seen_chunk_ids = set()
    dropped = 0

    for source in sources or []:
        if not isinstance(source, dict):
            dropped += 1
            continue

        evidence_id = str(source.get("evidence_id", "") or "").strip()
        chunk_id = str(source.get("chunk_id", "") or "").strip()

        evidence = None
        if evidence_id and evidence_id in evidence_index:
            evidence = evidence_index[evidence_id]
        elif chunk_id:
            for candidate in evidence_index.values():
                if candidate.get("chunk_id") == chunk_id:
                    evidence = candidate
                    break

        if evidence is None:
            dropped += 1
            warnings.append(
                "The model referenced a source that was not retrieved; "
                "the citation was discarded."
            )
            continue

        real_chunk_id = evidence.get("chunk_id", "")
        if real_chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(real_chunk_id)

        quote = source.get("quote") or source.get("supporting_text") or ""
        if not isinstance(quote, str):
            quote = str(quote)
        quote = quote.strip()
        if not quote:
            quote = evidence.get("text", "").strip()

        citations.append(
            {
                "evidence_id": evidence.get("evidence_id", evidence_id),
                "chunk_id": real_chunk_id,
                "document_id": evidence.get("document_id", ""),
                # Filename and page come from the retrieved chunk, not the LLM.
                "filename": evidence.get("filename", "Unknown document"),
                "page_number": evidence.get("page_number"),
                "page_label": evidence.get("page_label", page_label(evidence.get("page_number"))),
                "section_heading": evidence.get("section_heading", ""),
                "quote": quote[:1200],
                "full_text": evidence.get("text", ""),
                "similarity": evidence.get("similarity"),
                "score": evidence.get("score"),
                "score_type": evidence.get("score_type"),
            }
        )

    return {"citations": citations, "warnings": warnings, "dropped": dropped}
