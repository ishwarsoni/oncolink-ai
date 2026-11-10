"""
styles.py - Page configuration, custom CSS, and shared layout helpers.
"""

import streamlit as st


def configure_page():
    """Set the browser tab title, layout, and sidebar defaults."""
    st.set_page_config(
        page_title="OncoLink",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def load_custom_css():
    """Inject the project's CSS overrides into the page."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main {
            padding: 1.5rem 2.5rem;
            background: #f8fafc;
        }
        
        .stApp {
            background: #f8fafc;
        }
        
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0f2b4a 0%, #1a6b9e 50%, #2d9cdb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
            letter-spacing: -0.5px;
        }
        
        .main-subtitle {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1rem;
            font-weight: 400;
        }
        
        .main-description {
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 1.5rem;
            line-height: 1.6;
            background: #ffffff;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        
        .section-heading {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0f2b4a;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .placeholder-box {
            background: #ffffff;
            border: 1.5px dashed #cbd5e1;
            border-radius: 12px;
            padding: 1.8rem;
            text-align: center;
            color: #94a3b8;
            font-size: 0.95rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            transition: all 0.2s ease;
        }
        
        .placeholder-box:hover {
            border-color: #94a3b8;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        .badge-success {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: #ffffff;
            padding: 2px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        
        .badge-pending {
            background: #f1f5f9;
            color: #94a3b8;
            padding: 2px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        
        .file-item {
            padding: 10px 14px;
            background: #ffffff;
            border-left: 3.5px solid #2d9cdb;
            margin-bottom: 8px;
            border-radius: 8px;
            font-size: 0.9rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        
        .file-item:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateX(2px);
        }
        
        .phase-item {
            padding: 6px 0;
            font-size: 0.88rem;
        }
        
        div[data-testid="stMetric"] {
            background: #ffffff;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: all 0.2s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-color: #cbd5e1;
        }
        
        div[data-testid="stMetric"] label {
            color: #64748b !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f2b4a !important;
            font-weight: 700 !important;
            font-size: 1.3rem !important;
        }
        
        .stButton button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        }
        
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #1a6b9e 0%, #2d9cdb 100%) !important;
            border: none !important;
            color: #ffffff !important;
        }
        
        .stButton button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(45, 156, 219, 0.35) !important;
        }
        
        .stButton button[kind="primary"]:disabled {
            background: #cbd5e1 !important;
            box-shadow: none !important;
        }
        
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 0.8rem;
        }
        
        div[data-testid="stExpander"] summary {
            font-weight: 600;
            color: #0f2b4a;
            padding: 0.5rem 0;
        }
        
        div[data-testid="stInfo"] {
            background: #eff6ff !important;
            border: 1px solid #bfdbfe !important;
            border-radius: 10px !important;
            color: #1e40af !important;
        }
        
        div[data-testid="stSuccess"] {
            background: #f0fdf4 !important;
            border: 1px solid #bbf7d0 !important;
            border-radius: 10px !important;
        }
        
        div[data-testid="stWarning"] {
            background: #fffbeb !important;
            border: 1px solid #fde68a !important;
            border-radius: 10px !important;
        }
        
        div[data-testid="stError"] {
            background: #fef2f2 !important;
            border: 1px solid #fecaca !important;
            border-radius: 10px !important;
        }
        
        .stSidebar {
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        
        .stSidebar .sidebar-content {
            background: #ffffff !important;
        }
        
        section[data-testid="stSidebar"] {
            background: #ffffff !important;
        }
        
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #0f2b4a;
            font-weight: 700;
        }
        
        div[data-testid="stFileUploader"] section {
            border: 2px dashed #cbd5e1 !important;
            border-radius: 12px !important;
            background: #ffffff !important;
            padding: 2rem !important;
            transition: all 0.2s ease !important;
        }
        
        div[data-testid="stFileUploader"] section:hover {
            border-color: #2d9cdb !important;
            background: #f8fafc !important;
        }
        
        div[data-testid="stSpinner"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        
        .stCode {
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.5rem !important;
        }
        
        hr {
            margin: 1.5rem 0 !important;
            border-color: #e2e8f0 !important;
        }
        
        .st-cb {
            color: #0f2b4a !important;
        }
        
        .st-b7 {
            color: #64748b !important;
        }
        
        .st-d5 {
            border-color: #e2e8f0 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def _gradient_divider(margin="1.5rem 0 1rem 0", container=None):
    """Render the subtle horizontal divider used between page sections."""
    target = container if container is not None else st
    target.markdown(
        f'<div style="height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:{margin};"></div>',
        unsafe_allow_html=True,
    )


def create_footer():
    """Render the page footer."""
    _gradient_divider("2rem 0 1rem 0")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<span style="font-size:0.8rem; color:#94a3b8;">🏥 OncoLink v1.0.0</span>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align:center; font-size:0.8rem; color:#94a3b8;">AI Clinical Intelligence Platform</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="text-align:right; font-size:0.8rem; color:#94a3b8;">Phase 5: Harmonization & Summary</div>', unsafe_allow_html=True)
