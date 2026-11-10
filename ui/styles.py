"""
styles.py - Page configuration, theme-aware CSS, and shared layout helpers.

The stylesheet deliberately leans on Streamlit's own theme variables
(``--background-color``, ``--secondary-background-color``, ``--text-color``)
and ``color-mix()`` rather than a second hard-coded palette.  That keeps the
workspace readable in both the light and dark themes without maintaining two
stylesheets, and it means a themed deployment never gets forced-light cards.
"""

import streamlit as st

_FONT_STACK = (
    '"Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, '
    '"Helvetica Neue", Arial, sans-serif'
)

_CSS = """
<style>
    /* ------------------------------------------------------------------
       Theme-derived design tokens
       ------------------------------------------------------------------ */
    .stApp {
        --ol-accent: color-mix(in srgb, #0f766e 72%, var(--text-color));
        --ol-accent-soft: color-mix(in srgb, var(--ol-accent) 12%, transparent);
        --ol-border: color-mix(in srgb, var(--text-color) 14%, transparent);
        --ol-border-strong: color-mix(in srgb, var(--text-color) 26%, transparent);
        --ol-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
        --ol-surface: color-mix(
            in srgb,
            var(--secondary-background-color) 72%,
            var(--background-color)
        );
        --ol-ok: color-mix(in srgb, #15803d 70%, var(--text-color));
        --ol-ok-soft: color-mix(in srgb, var(--ol-ok) 13%, transparent);
        --ol-warn: color-mix(in srgb, #b45309 76%, var(--text-color));
        --ol-warn-soft: color-mix(in srgb, var(--ol-warn) 14%, transparent);
        --ol-danger: color-mix(in srgb, #b91c1c 76%, var(--text-color));
        --ol-danger-soft: color-mix(in srgb, var(--ol-danger) 13%, transparent);
        --ol-info: color-mix(in srgb, #1d4ed8 66%, var(--text-color));
        --ol-info-soft: color-mix(in srgb, var(--ol-info) 12%, transparent);
    }

    html,
    body,
    [class*="st-"],
    .stApp {
        font-family: __FONT_STACK__;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    /* ------------------------------------------------------------------
       Product header
       ------------------------------------------------------------------ */
    .ol-hero {
        border: 1px solid var(--ol-border);
        background: var(--ol-surface);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .ol-eyebrow {
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ol-muted);
        font-weight: 700;
    }

    .ol-hero-title {
        font-size: 1.85rem;
        line-height: 1.15;
        font-weight: 750;
        letter-spacing: -0.025em;
        color: var(--text-color);
        margin: 0.2rem 0 0.15rem;
    }

    .ol-hero-sub {
        margin: 0;
        color: var(--ol-muted);
        font-size: 0.92rem;
        line-height: 1.55;
        max-width: 78ch;
    }

    .ol-view-head {
        margin: 0.35rem 0 1rem;
    }

    .ol-view-title {
        margin: 0;
        font-size: 1.35rem;
        line-height: 1.2;
        font-weight: 720;
        letter-spacing: -0.02em;
        color: var(--text-color);
    }

    .ol-view-sub {
        margin: 0.25rem 0 0;
        color: var(--ol-muted);
        font-size: 0.88rem;
        line-height: 1.55;
        max-width: 82ch;
    }

    /* ------------------------------------------------------------------
       Section headings
       ------------------------------------------------------------------ */
    .ol-section-head {
        margin: 1.4rem 0 0.7rem;
    }

    .ol-section-title {
        margin: 0;
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text-color);
    }

    .ol-section-desc {
        margin: 0.2rem 0 0;
        color: var(--ol-muted);
        font-size: 0.82rem;
        line-height: 1.5;
    }

    /* ------------------------------------------------------------------
       Clinical value blocks
       ------------------------------------------------------------------ */
    .ol-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.6rem;
        margin-bottom: 0.4rem;
    }

    .ol-metric,
    .ol-block {
        border: 1px solid var(--ol-border);
        background: var(--ol-surface);
        border-radius: 10px;
        padding: 0.72rem 0.9rem;
    }

    .ol-block + .ol-block {
        margin-top: 0.5rem;
    }

    .ol-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--ol-muted);
        margin-bottom: 0.28rem;
    }

    .ol-value {
        font-size: 0.94rem;
        font-weight: 600;
        color: var(--text-color);
        overflow-wrap: anywhere;
    }

    .ol-body {
        font-weight: 450;
        line-height: 1.55;
    }

    .ol-missing {
        color: var(--ol-warn);
        font-weight: 600;
    }

    .ol-muted-text {
        color: var(--ol-muted);
        font-size: 0.82rem;
        line-height: 1.5;
    }

    /* ------------------------------------------------------------------
       Pills and status
       ------------------------------------------------------------------ */
    .ol-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.35rem 0 0.25rem;
    }

    .ol-pill,
    .ol-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        border-radius: 999px;
        padding: 0.2rem 0.62rem;
        font-size: 0.71rem;
        font-weight: 650;
        line-height: 1.35;
        border: 1px solid transparent;
        white-space: nowrap;
    }

    .ol-pill--ok {
        color: var(--ol-ok);
        background: var(--ol-ok-soft);
        border-color: color-mix(in srgb, var(--ol-ok) 30%, transparent);
    }

    .ol-pill--info {
        color: var(--ol-info);
        background: var(--ol-info-soft);
        border-color: color-mix(in srgb, var(--ol-info) 30%, transparent);
    }

    .ol-pill--warn {
        color: var(--ol-warn);
        background: var(--ol-warn-soft);
        border-color: color-mix(in srgb, var(--ol-warn) 32%, transparent);
    }

    .ol-pill--bad {
        color: var(--ol-danger);
        background: var(--ol-danger-soft);
        border-color: color-mix(in srgb, var(--ol-danger) 32%, transparent);
    }

    .ol-pill--neutral {
        color: var(--ol-muted);
        background: color-mix(in srgb, var(--ol-muted) 10%, transparent);
        border-color: var(--ol-border);
    }

    /* ------------------------------------------------------------------
       Empty states and notices
       ------------------------------------------------------------------ */
    .ol-empty {
        border: 1px dashed var(--ol-border-strong);
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin: 0.6rem 0;
        background: transparent;
    }

    .ol-empty-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--text-color);
    }

    .ol-empty-body {
        margin: 0.3rem 0 0;
        color: var(--ol-muted);
        font-size: 0.86rem;
        line-height: 1.55;
        max-width: 72ch;
    }

    .ol-empty-hint {
        margin: 0.55rem 0 0;
        color: var(--ol-muted);
        font-size: 0.8rem;
        font-style: normal;
    }

    /* ------------------------------------------------------------------
       Provenance and conflicts
       ------------------------------------------------------------------ */
    .ol-provenance {
        border: 1px solid var(--ol-border);
        border-radius: 10px;
        overflow: hidden;
        background: transparent;
    }

    .ol-provenance-head,
    .ol-provenance-row {
        display: grid;
        grid-template-columns: minmax(140px, 1fr) minmax(0, 2fr);
        gap: 1rem;
        padding: 0.55rem 0.85rem;
        font-size: 0.84rem;
    }

    .ol-provenance-head {
        background: var(--ol-surface);
        color: var(--ol-muted);
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .ol-provenance-row + .ol-provenance-row {
        border-top: 1px solid var(--ol-border);
    }

    .ol-provenance-field {
        color: var(--ol-muted);
    }

    .ol-provenance-source {
        color: var(--text-color);
        overflow-wrap: anywhere;
    }

    .ol-conflict {
        border: 1px solid var(--ol-border);
        border-left: 3px solid var(--ol-warn);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        background: var(--ol-surface);
    }

    .ol-conflict--bad {
        border-left-color: var(--ol-danger);
    }

    .ol-conflict--warn {
        border-left-color: var(--ol-warn);
    }

    .ol-conflict-head {
        display: flex;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 0.45rem;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .ol-conflict-source {
        display: grid;
        grid-template-columns: minmax(120px, 1fr) minmax(0, 2fr);
        gap: 0.75rem;
        margin-top: 0.4rem;
        color: var(--ol-muted);
        font-size: 0.81rem;
        overflow-wrap: anywhere;
    }

    /* ------------------------------------------------------------------
       Biomarkers and documents
       ------------------------------------------------------------------ */
    .ol-biomarker-table {
        border: 1px solid var(--ol-border);
        border-radius: 10px;
        overflow: hidden;
    }

    .ol-biomarker-row {
        display: grid;
        grid-template-columns: minmax(110px, 1fr) minmax(90px, 1fr) minmax(90px, 1fr);
        gap: 0.75rem;
        padding: 0.55rem 0.85rem;
        font-size: 0.86rem;
        align-items: center;
    }

    .ol-biomarker-row + .ol-biomarker-row {
        border-top: 1px solid var(--ol-border);
    }

    .ol-biomarker-head {
        background: var(--ol-surface);
        color: var(--ol-muted);
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .ol-file-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        border: 1px solid var(--ol-border);
        border-radius: 9px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.45rem;
        background: var(--ol-surface);
    }

    .ol-file-name {
        flex: 1;
        min-width: 0;
        font-size: 0.89rem;
        font-weight: 600;
        color: var(--text-color);
        overflow-wrap: anywhere;
    }

    .ol-file-meta {
        color: var(--ol-muted);
        font-size: 0.77rem;
        white-space: nowrap;
    }

    /* ------------------------------------------------------------------
       Streamlit widget refinements (theme variables only)
       ------------------------------------------------------------------ */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px !important;
        border: 1px solid var(--ol-border-strong) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: background-color 120ms ease, border-color 120ms ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--ol-accent) !important;
        color: var(--ol-accent) !important;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: #0f5f59 !important;
        border-color: #0f5f59 !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        background: #0b4a45 !important;
        border-color: #0b4a45 !important;
        color: #ffffff !important;
    }

    .stButton > button:disabled,
    .stDownloadButton > button:disabled {
        opacity: 0.45 !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--ol-border) !important;
        border-radius: 10px !important;
        background: transparent !important;
    }

    [data-testid="stExpander"] summary {
        font-weight: 600;
    }

    [data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] section {
        background: var(--ol-surface) !important;
        border: 1px dashed var(--ol-border-strong) !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--ol-border);
        border-radius: 12px;
        background: var(--ol-surface);
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.55rem;
    }

    [data-testid="stAlert"] {
        border-radius: 9px;
    }

    [data-testid="stMetric"] {
        border: 1px solid var(--ol-border);
        background: var(--ol-surface);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
    }

    [data-testid="stRadio"] div[role="radiogroup"] {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.2rem;
        border: 1px solid var(--ol-border);
        background: var(--ol-surface);
        border-radius: 10px;
        padding: 0.25rem;
        margin-bottom: 0.3rem;
    }

    [data-testid="stRadio"] div[role="radiogroup"] > label {
        margin: 0 !important;
        padding: 0.3rem 0.72rem !important;
        border-radius: 8px;
        cursor: pointer;
    }

    [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: var(--ol-accent-soft);
        color: var(--ol-accent);
    }

    [data-testid="stRadio"] div[role="radiogroup"] > label:focus-within {
        outline: 2px solid var(--ol-accent);
        outline-offset: 1px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--ol-border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    .ol-brand {
        padding: 0.4rem 0 0.2rem;
    }

    .ol-brand-name {
        font-size: 1.45rem;
        font-weight: 760;
        letter-spacing: -0.02em;
        color: var(--text-color);
        line-height: 1.15;
    }

    .ol-brand-sub {
        color: var(--ol-muted);
        font-size: 0.76rem;
        line-height: 1.4;
        margin-top: 0.15rem;
    }

    .ol-sidebar-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ol-muted);
        margin: 0.9rem 0 0.35rem;
    }

    .ol-sidebar-note {
        color: var(--ol-muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }

    .ol-divider {
        height: 0;
        border-top: 1px solid var(--ol-border);
    }

    .ol-footer {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem 1rem;
        justify-content: space-between;
        color: var(--ol-muted);
        font-size: 0.76rem;
        padding-top: 0.4rem;
    }

    /* Legacy classes retained for integrations that render one-off markup. */
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }

    .main-subtitle,
    .main-description,
    .section-heading {
        color: var(--ol-muted);
    }

    .main-description {
        background: var(--ol-surface);
        border: 1px solid var(--ol-border);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .section-heading {
        font-size: 1rem;
        font-weight: 700;
        margin: 1.2rem 0 0.6rem;
    }

    .placeholder-box {
        background: transparent;
        border: 1px dashed var(--ol-border-strong);
        border-radius: 10px;
        padding: 1.4rem;
        text-align: center;
        color: var(--ol-muted);
        font-size: 0.9rem;
    }

    .file-item {
        background: var(--ol-surface);
        border: 1px solid var(--ol-border);
        border-left: 3px solid var(--ol-accent);
        border-radius: 8px;
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.45rem;
        font-size: 0.87rem;
    }

    .badge-success,
    .badge-pending {
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-success {
        color: var(--ol-ok);
        background: var(--ol-ok-soft);
    }

    .badge-pending {
        color: var(--ol-muted);
        background: color-mix(in srgb, var(--ol-muted) 10%, transparent);
    }

    /* ------------------------------------------------------------------
       Responsive layout
       ------------------------------------------------------------------ */
    @media (max-width: 780px) {
        .block-container {
            padding-top: 1.1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .ol-hero {
            padding: 1rem;
        }

        .ol-hero-title {
            font-size: 1.5rem;
        }

        .ol-view-title {
            font-size: 1.18rem;
        }

        .ol-provenance-head,
        .ol-provenance-row,
        .ol-conflict-source {
            grid-template-columns: 1fr;
            gap: 0.15rem;
        }

        .ol-file-row {
            flex-wrap: wrap;
        }

        .ol-file-meta {
            width: 100%;
            white-space: normal;
        }
    }

    @media (max-width: 520px) {
        .ol-grid {
            grid-template-columns: 1fr;
        }

        .ol-biomarker-row {
            grid-template-columns: 1fr;
            gap: 0.15rem;
        }
    }
</style>
""".replace("__FONT_STACK__", _FONT_STACK)


def configure_page():
    """Set the browser tab title, layout, and sidebar defaults."""
    st.set_page_config(
        page_title="OncoLink - Clinical Intelligence",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_custom_css():
    """Inject the theme-aware stylesheet into the page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def _gradient_divider(margin="1.5rem 0 1rem 0", container=None):
    """Render a hairline divider; the name is kept for backwards compatibility."""
    target = container if container is not None else st
    target.markdown(
        f'<div class="ol-divider" style="margin:{margin};"></div>',
        unsafe_allow_html=True,
    )


def create_footer():
    """Render the page footer and its educational-use disclaimer."""
    _gradient_divider("2rem 0 1rem 0")
    st.markdown(
        '<div class="ol-footer">'
        "<span>OncoLink v3.0 - AI Clinical Intelligence Platform</span>"
        "<span>Structured clinical intelligence and evidence-grounded chat</span>"
        "<span>Educational use only; not a medical device.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
