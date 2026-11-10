"""
chunker.py - Token-aware document chunking with clinical metadata.

An LLM cannot receive a whole patient chart for every question, so documents
are split into small overlapping passages ("chunks"), embedded, and retrieved
individually.

Design goals:
    1. Never split mid-sentence when a boundary is available.
    2. Prefer paragraph boundaries, then sentence boundaries.
    3. Keep the document name, page number, and nearest section heading.
    4. Produce stable chunk IDs so unchanged content is not indexed twice.
    5. Handle empty or malformed documents without raising.

Token counting uses tiktoken's cl100k_base encoding when installed.  It is a
*proxy* for the embedding model's tokenizer, so sizes are described as
approximate.  If tiktoken is unavailable, word/punctuation units are counted
and the method is reported by ``token_count_method()``.
"""

import hashlib
import re
from typing import Dict, Iterable, List, Optional

try:
    import tiktoken

    _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - exercised only when tiktoken is absent
    _TOKEN_ENCODING = None


_APPROX_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Common clinical section headings.  The heuristic also catches ALL-CAPS
# headings and short colon-terminated labels, so this list is a supplement.
_KNOWN_HEADINGS = {
    "patient information",
    "patient details",
    "demographics",
    "chief complaint",
    "history of presenting illness",
    "history of present illness",
    "past medical history",
    "medications",
    "current medications",
    "medications at discharge",
    "prescription",
    "diagnosis",
    "impression",
    "findings",
    "biomarkers",
    "molecular",
    "histopathology",
    "imaging",
    "biopsy",
    "management plan",
    "plan",
    "follow up",
    "follow-up",
    "follow-up plan",
    "next steps",
    "discharge instructions",
    "laboratory",
    "lab results",
    "treatment response",
    "adverse events",
    "recommendation",
    "clinical indication",
    "technique",
}

_MIN_CHUNK_SIZE = 50
_DEFAULT_CHUNK_SIZE = 700
_DEFAULT_CHUNK_OVERLAP = 120


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken, or approximate units when unavailable."""
    if not text:
        return 0
    if _TOKEN_ENCODING is not None:
        return len(_TOKEN_ENCODING.encode(text, disallowed_special=()))
    return len(_APPROX_TOKEN_PATTERN.findall(text))


def token_count_method() -> str:
    """Return a human-readable description of the token counting method."""
    if _TOKEN_ENCODING is not None:
        return "tiktoken:cl100k_base (proxy for the embedding tokenizer)"
    return "approximate word/punctuation units (tiktoken unavailable)"


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters without losing lines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_section_heading(line: str) -> bool:
    """
    Heuristically decide whether a line is a section heading.

    Clinical notes use ALL CAPS headings ("DIAGNOSIS:") and short
    colon-terminated labels.  The goal is to preserve context, not to parse
    every possible document layout perfectly.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False

    if stripped.rstrip(":").strip().lower() in _KNOWN_HEADINGS:
        return True

    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    word_count = len(stripped.split())

    if stripped.endswith(":") and word_count <= 8:
        return True
    return upper_ratio >= 0.8 and word_count <= 12


def _split_long_text(text: str, max_tokens: int) -> List[str]:
    """
    Split text that is too long for one chunk.

    Sentence boundaries are preferred.  A single oversized sentence falls back
    to word-level packing; text is never split by character count alone.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = [part.strip() for part in _SENTENCE_PATTERN.split(text) if part.strip()]
    pieces: List[str] = []
    buffer = ""

    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if count_tokens(candidate) <= max_tokens:
            buffer = candidate
            continue

        if buffer:
            pieces.append(buffer)
            buffer = ""

        if count_tokens(sentence) <= max_tokens:
            buffer = sentence
            continue

        # Hard fallback for run-on text: pack whole words up to the budget.
        word_buffer: List[str] = []
        for word in sentence.split():
            tentative = " ".join(word_buffer + [word])
            if word_buffer and count_tokens(tentative) > max_tokens:
                pieces.append(" ".join(word_buffer))
                word_buffer = [word]
            else:
                word_buffer.append(word)
        if word_buffer:
            buffer = " ".join(word_buffer)

    if buffer:
        pieces.append(buffer)

    return pieces or [text]


def _segment_page(text: str, max_tokens: int) -> List[Dict[str, object]]:
    """
    Convert a page into segments, marking which segments are headings.

    Paragraphs are separated by blank lines.  Consecutive non-blank lines stay
    together so clinical statements are not split arbitrarily.
    """
    segments: List[Dict[str, object]] = []
    paragraph_lines: List[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = "\n".join(paragraph_lines).strip()
        paragraph_lines = []
        if not paragraph:
            return
        for piece in _split_long_text(paragraph, max_tokens):
            segments.append({"text": piece, "is_heading": False})

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
        elif _is_section_heading(stripped):
            flush_paragraph()
            segments.append({"text": stripped, "is_heading": True})
        else:
            paragraph_lines.append(stripped)

    flush_paragraph()
    return segments


def _normalise_for_hash(text: str) -> str:
    """Normalize text so cosmetic differences do not create duplicate chunks."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _make_chunk_id(document_id: str, page_number, chunk_index: int, text: str) -> str:
    """Create a stable, non-identifying ID for a chunk."""
    payload = f"{document_id}|{page_number}|{chunk_index}|{_normalise_for_hash(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _overlap_text(text: str, overlap_tokens: int) -> str:
    """
    Return the trailing part of a chunk, measured in tokens.

    When a paragraph is longer than the overlap budget we still want a small
    tail of context, so this helper works at token granularity and drops the
    first (possibly partial) token.
    """
    if overlap_tokens <= 0 or not text:
        return ""

    if _TOKEN_ENCODING is not None:
        token_ids = _TOKEN_ENCODING.encode(text, disallowed_special=())
        if len(token_ids) <= overlap_tokens:
            return text
        tail = _TOKEN_ENCODING.decode(token_ids[-overlap_tokens:]).strip()
        parts = tail.split()
        if len(parts) > 1:
            tail = " ".join(parts[1:])
        return tail.strip()

    words = text.split()
    if len(words) <= overlap_tokens:
        return text
    return " ".join(words[-overlap_tokens:])


class _DocumentChunker:
    """Accumulate chunks for one document while walking its pages."""

    def __init__(self, document: Dict, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.patient_id = str(document.get("patient_id", "") or "")
        self.document_id = str(document.get("document_id", "") or "")
        self.filename = document.get("filename", "Unknown document")
        self.file_type = document.get("file_type", "")
        self.upload_timestamp = document.get("upload_timestamp")
        self.document_date = document.get("document_date")
        self.content_hash = document.get("content_hash", "")

        self.chunks: List[Dict] = []
        self._seen_text: set = set()
        self._chunk_index = 0

        self._buffer: List[str] = []
        self._buffer_tokens = 0
        self._current_heading: Optional[str] = None
        self._page_number = None

    def chunk_pages(self, pages: Iterable[Dict]) -> List[Dict]:
        """Chunk every page and return the accumulated chunk list."""
        for page in pages:
            self._chunk_page(page)
        return self.chunks

    def _chunk_page(self, page: Dict) -> None:
        """Split one page into segments and pack them into chunks."""
        self._page_number = page.get("page_number")
        page_text = clean_text(page.get("text", ""))
        if not page_text:
            return

        # Buffer state is per page so a heading never leaks across pages.
        self._buffer = []
        self._buffer_tokens = 0
        self._current_heading = None

        for segment in _segment_page(page_text, self.chunk_size):
            self._add_segment(segment)
        self._flush_buffer()

    def _add_segment(self, segment: Dict[str, object]) -> None:
        """Append one segment, flushing the buffer when the budget is reached."""
        segment_text = str(segment["text"]).strip()
        if not segment_text:
            return

        segment_tokens = count_tokens(segment_text)

        if segment["is_heading"]:
            # A heading starts a fresh chunk so it stays attached to the
            # content that follows it.
            if self._buffer:
                self._flush_buffer()
            self._current_heading = segment_text.rstrip(":").strip()
            self._buffer = [segment_text]
            self._buffer_tokens = segment_tokens
            return

        if self._buffer and self._buffer_tokens + segment_tokens > self.chunk_size:
            flushed_text = self._flush_buffer()

            # Carry a small overlap forward.  Heading-only chunks are not used
            # as overlap because that would detach the heading from its content.
            if (
                self.chunk_overlap > 0
                and flushed_text
                and not _is_section_heading(flushed_text)
            ):
                overlap = _overlap_text(flushed_text, self.chunk_overlap)
                if overlap:
                    self._buffer = [overlap]
                    self._buffer_tokens = count_tokens(overlap)

        self._buffer.append(segment_text)
        self._buffer_tokens += segment_tokens

    def _flush_buffer(self) -> str:
        """Emit the buffered text as a chunk and return it (empty when none)."""
        if not self._buffer:
            return ""

        chunk_text = "\n".join(self._buffer).strip()
        self._buffer = []
        self._buffer_tokens = 0
        if not chunk_text:
            return ""

        normalized = _normalise_for_hash(chunk_text)
        if normalized in self._seen_text:
            return chunk_text
        self._seen_text.add(normalized)

        chunk_id = _make_chunk_id(
            self.document_id, self._page_number, self._chunk_index, chunk_text
        )
        self.chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": self._chunk_index,
                "text": chunk_text,
                "patient_id": self.patient_id,
                "document_id": self.document_id,
                "filename": self.filename,
                "file_type": self.file_type,
                "page_number": self._page_number if self._page_number is not None else -1,
                "section_heading": self._current_heading or "",
                "upload_timestamp": self.upload_timestamp or "",
                "document_date": self.document_date or "",
                "content_hash": self.content_hash or "",
                "token_count": count_tokens(chunk_text),
                "char_count": len(chunk_text),
            }
        )
        self._chunk_index += 1
        return chunk_text


def chunk_document(
    document: Dict,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split one normalized document dict into chunks.

    Required keys: ``text``, ``filename``, ``document_id``, ``patient_id``.
    Optional keys: ``file_type``, ``pages``, ``upload_timestamp``,
    ``document_date``, ``content_hash``.
    """
    if not isinstance(document, dict):
        return []

    text = clean_text(document.get("text", ""))
    pages = document.get("pages") or []
    if not text and not pages:
        return []

    chunk_size = max(_MIN_CHUNK_SIZE, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))

    if not pages:
        pages = [{"page_number": None, "text": text}]

    return _DocumentChunker(document, chunk_size, chunk_overlap).chunk_pages(pages)


def chunk_documents(
    documents: Iterable[Dict],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """Chunk several documents, concatenating their chunk lists."""
    all_chunks: List[Dict] = []
    for document in documents or []:
        all_chunks.extend(
            chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return all_chunks
