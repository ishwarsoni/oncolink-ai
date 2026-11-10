"""Reusable, theme-aware presentation primitives for the OncoLink workspace.

These helpers render small pieces of the clinical UI with the project's CSS
classes.  They contain no business logic: values are passed in by the view
modules and every dynamic value is HTML-escaped before it is rendered.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st

MISSING = "Not documented"

_STATE_TONES = {
    "succeeded": "ok",
    "running": "info",
    "failed": "bad",
    "skipped": "neutral",
    "pending": "neutral",
}

_STAGE_LABELS = {
    "ingestion": "Ingestion",
    "clinical": "Clinical record",
    "rag": "Evidence index",
}


def esc(value: Any) -> str:
    """Escape a value for safe insertion into an HTML block."""
    return html.escape(str(value), quote=True)


def display_value(value: Any, missing: str = MISSING) -> tuple[str, bool]:
    """Return ``(text, is_missing)`` for a clinical value."""
    if value is None:
        return missing, True
    if isinstance(value, str) and not value.strip():
        return missing, True
    if isinstance(value, (list, tuple, dict)) and not value:
        return missing, True
    return str(value).strip(), False


def _basename(value: Any) -> str:
    """Return a display-safe document name without directory components."""
    text = str(value).replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def render_view_header(title: str, description: str = "") -> None:
    """Render the heading for one workspace view."""
    desc = f'<p class="ol-view-sub">{esc(description)}</p>' if description else ""
    st.markdown(
        f'<div class="ol-view-head"><h2 class="ol-view-title">{esc(title)}</h2>{desc}</div>',
        unsafe_allow_html=True,
    )


def render_product_header(
    title: str, subtitle: str, description: str, eyebrow: str = "OncoLink"
) -> None:
    """Render the compact product banner at the top of the workspace."""
    st.markdown(
        '<div class="ol-hero">'
        f'<div class="ol-eyebrow">{esc(eyebrow)}</div>'
        f'<div class="ol-hero-title">{esc(title)}</div>'
        f'<p class="ol-hero-sub"><strong>{esc(subtitle)}</strong> {esc(description)}</p>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_section(title: str, description: str = "") -> None:
    """Render a section heading with an optional one-line description."""
    body = f'<h3 class="ol-section-title">{esc(title)}</h3>'
    if description:
        body += f'<p class="ol-section-desc">{esc(description)}</p>'
    st.markdown(f'<div class="ol-section-head">{body}</div>', unsafe_allow_html=True)


def render_metric_grid(items: Sequence[Mapping[str, Any]]) -> None:
    """Render responsive label/value cards.

    Each item accepts ``label``, ``value`` and an optional ``missing`` text.
    Missing values are visually muted but stay explicit.
    """
    cards = []
    for item in items:
        label = esc(item.get("label", ""))
        text, missing = display_value(item.get("value"), item.get("missing", MISSING))
        value_class = "ol-value ol-missing" if missing else "ol-value"
        cards.append(
            '<div class="ol-metric">'
            f'<div class="ol-label">{label}</div>'
            f'<div class="{value_class}">{esc(text)}</div>'
            "</div>"
        )
    if cards:
        st.markdown(
            f'<div class="ol-grid">{"".join(cards)}</div>', unsafe_allow_html=True
        )


def render_text_blocks(items: Sequence[Mapping[str, Any]]) -> None:
    """Render full-width labelled text blocks (medication, plan, ...)."""
    blocks = []
    for item in items:
        label = esc(item.get("label", ""))
        text, missing = display_value(item.get("value"), item.get("missing", MISSING))
        value_class = "ol-value ol-missing" if missing else "ol-value"
        safe_text = esc(text).replace("\n", "<br>")
        blocks.append(
            '<div class="ol-block">'
            f'<div class="ol-label">{label}</div>'
            f'<div class="{value_class} ol-body">{safe_text}</div>'
            "</div>"
        )
    if blocks:
        st.markdown("".join(blocks), unsafe_allow_html=True)


def render_biomarkers(items: Sequence[Mapping[str, Any]]) -> None:
    """Render documented biomarkers without interpreting their values."""
    if not items:
        render_empty_state(
            "No biomarkers documented",
            "The uploaded documents did not yield a biomarker or test-result entry.",
        )
        return

    rows = [
        '<div class="ol-biomarker-row ol-biomarker-head">'
        "<div>Biomarker</div><div>Value</div><div>Status</div>"
        "</div>"
    ]
    for item in items:
        name, _ = display_value(item.get("name"))
        value, _ = display_value(item.get("value"))
        status, _ = display_value(item.get("status"))
        rows.append(
            '<div class="ol-biomarker-row">'
            f'<div class="ol-value">{esc(name)}</div>'
            f'<div>{esc(value)}</div>'
            f'<div><span class="ol-pill ol-pill--neutral">{esc(status)}</span></div>'
            "</div>"
        )
    st.markdown(
        f'<div class="ol-biomarker-table">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_pills(items: Iterable[tuple[str, str]], container=None) -> None:
    """Render inline status pills; tone is one of ok/info/warn/bad/neutral."""
    pills = "".join(
        f'<span class="ol-pill ol-pill--{esc(tone)}">{esc(text)}</span>'
        for text, tone in items
    )
    if pills:
        target = container if container is not None else st
        target.markdown(f'<div class="ol-pill-row">{pills}</div>', unsafe_allow_html=True)


def render_empty_state(title: str, body: str, hint: str = "") -> None:
    """Render a quiet empty state for a section or page."""
    hint_html = f'<p class="ol-empty-hint">{esc(hint)}</p>' if hint else ""
    st.markdown(
        '<div class="ol-empty">'
        f'<div class="ol-empty-title">{esc(title)}</div>'
        f'<p class="ol-empty-body">{esc(body)}</p>'
        f"{hint_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_provenance(rows: Sequence[Mapping[str, Any]]) -> None:
    """Render a field-to-source table for the harmonized record."""
    if not rows:
        render_empty_state(
            "No field-level provenance yet",
            "Provenance is recorded when per-document extractions are harmonized.",
        )
        return

    body = []
    for row in rows:
        label = esc(row.get("label", ""))
        sources = row.get("sources") or "Not recorded"
        body.append(
            '<div class="ol-provenance-row">'
            f'<div class="ol-provenance-field">{label}</div>'
            f'<div class="ol-provenance-source">{esc(sources)}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="ol-provenance">'
        '<div class="ol-provenance-head">'
        "<div>Field</div><div>Source document</div>"
        "</div>"
        f'{"".join(body)}'
        "</div>",
        unsafe_allow_html=True,
    )


def render_source_documents(records: Sequence[Mapping[str, Any]]) -> None:
    """Render the canonical documents that fed the structured record."""
    if not records:
        render_empty_state(
            "No source documents",
            "Process documents to populate the provenance list.",
        )
        return

    rows = []
    for record in records:
        pages = record.get("pages") or []
        page_count = sum(1 for page in pages if page.get("page_number") is not None)
        pages_label = f"{page_count} page(s)" if page_count else "page count unavailable"
        rows.append(
            '<div class="ol-file-row">'
            f'<div class="ol-file-name">{esc(record.get("filename", "Unknown"))}</div>'
            f'<div class="ol-file-meta">{esc(record.get("file_type", "Document"))}'
            f" - {esc(pages_label)}</div>"
            f'<span class="ol-pill ol-pill--{"ok" if record.get("success") else "bad"}">'
            f'{"Read" if record.get("success") else "Failed"}</span>'
            "</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_conflicts(conflicts: Sequence[Mapping[str, Any]]) -> None:
    """Render conflict cards with severity and every recorded source value."""
    if not conflicts:
        st.success("No conflicts were detected across the uploaded documents.")
        return

    cards = []
    for conflict in conflicts:
        severity = str(conflict.get("severity", "medium")).lower()
        tone = "bad" if severity == "high" else "warn"
        description = conflict.get("description") or conflict.get("field") or "Conflict"
        sources = conflict.get("sources") or []
        source_lines = "".join(
            '<div class="ol-conflict-source">'
            f"<span>{esc(_basename(pair[0]))}</span>"
            f"<span>{esc(pair[1])}</span>"
            "</div>"
            for pair in sources
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        )
        cards.append(
            f'<div class="ol-conflict ol-conflict--{tone}">'
            '<div class="ol-conflict-head">'
            f'<span class="ol-pill ol-pill--{tone}">{esc(severity.title())}</span>'
            f"<strong>{esc(description)}</strong>"
            "</div>"
            f"{source_lines}"
            "</div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_status_strip(stages: Iterable[Mapping[str, Any]]) -> None:
    """Render the independent ingestion/clinical/RAG stage statuses."""
    pills = []
    for stage in stages:
        state = str(stage.get("state", "pending")).lower()
        tone = _STATE_TONES.get(state, "neutral")
        stage_name = str(stage.get("stage", ""))
        label = stage.get("label", _STAGE_LABELS.get(stage_name, stage_name.title() or "Stage"))
        message = stage.get("message") or ""
        title = f' title="{esc(message)}"' if message else ""
        pills.append(
            f'<span class="ol-status-pill ol-pill--{tone}"{title}>'
            f'{esc(label)} - {esc(state.title())}'
            "</span>"
        )
    st.markdown(f'<div class="ol-pill-row">{"".join(pills)}</div>', unsafe_allow_html=True)


def render_file_row(
    filename: str,
    file_type: str,
    pages: str,
    status: str,
    tone: str = "neutral",
    size: str = "",
) -> None:
    """Render one document row in the Uploaded Documents view."""
    size_html = f'<div class="ol-file-meta">{esc(size)}</div>' if size else ""
    st.markdown(
        '<div class="ol-file-row">'
        f'<div class="ol-file-name">{esc(filename)}</div>'
        f'<div class="ol-file-meta">{esc(file_type)} - {esc(pages)}</div>'
        f"{size_html}"
        f'<span class="ol-pill ol-pill--{esc(tone)}">{esc(status)}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
