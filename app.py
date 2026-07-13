"""
=============================================================================
OncoLink - AI Clinical Intelligence Platform
=============================================================================
Phase 4: Data Validation and Normalization

This is the main entry point for the OncoLink application.
It creates a web-based dashboard where users can upload oncology documents.

Currently (Phase 4):
    - Upload multiple PDF, TXT, and DOCX files ✅
    - Read and extract text from PDF files ✅
    - Read and extract text from DOCX files ✅
    - Read and extract text from TXT files ✅
    - Display extracted text in expandable sections ✅
    - AI-powered clinical data extraction ✅
    - Pydantic schema validation (NEW) ✅
    - JSON cleaning and error recovery (NEW) ✅
    - Data normalization (NEW) ✅
    - Validation error display (NEW) ✅
    - Export extracted data as JSON ✅

Future phases will add:
    - Multi-document harmonization (Phase 5)
    - Conflict detection (Phase 5)
    - Summary generation (Phase 5)

How to run this file:
    Open terminal in this folder and type:
    streamlit run app.py
=============================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# What we're doing: Importing the libraries we need.
# Why: Python libraries give us pre-built tools so we don't write everything
#   from scratch. Streamlit handles all the web app complexity for us.

import streamlit as st
# streamlit: The web framework. Lets us create UI elements like buttons,
#   text boxes, file uploaders — all in pure Python.
# Documentation: https://docs.streamlit.io/

from pathlib import Path
# pathlib.Path: A modern way to work with file paths.
# Why: Better than os.path.join(). Handles Windows backslashes and
#   Linux/Mac forward slashes automatically.

import datetime
# datetime: Gives us current date/time for the "Last Updated" timestamp.

# Import our document processing module.
# This is a custom module we created in the backend/ folder.
# It detects file types and calls the correct reader (PDF, DOCX, or TXT).
from backend.document_loader import process_uploaded_file

# Import our AI extraction modules (Phase 3)
# These handle the LLM-powered clinical data extraction pipeline
from backend.extractor import extract_from_multiple_documents, extract_each_document
from backend.models import get_field_label

# Phase 5: Import harmonization, conflict detection, and summary modules
from backend.harmonizer import harmonize_extractions
from backend.conflict_detector import detect_conflicts
from backend.summarizer import generate_patient_summary

# Import python-dotenv to load API keys from .env file
# .env keeps secrets out of our code (and out of GitHub)
from dotenv import load_dotenv

# Import os for checking environment variables
import os
# os.getenv() reads environment variables like NVIDIA_API_KEY

# Import logging for tracking application events
# Logs help us debug issues and understand what the app is doing
import logging


# =============================================================================
# Logging Configuration
# =============================================================================
# Set up logging to write to a file in the logs/ directory.
# Logs capture:
# - When validation starts and finishes
# - When errors occur (with full error details)
# - When normalization is performed
#
# Why logs instead of print():
#   - print() only shows in the terminal
#   - Logs go to a file you can review later
#   - Logs have timestamps and severity levels
#   - You can search logs for specific events
# =============================================================================

def setup_logging():
    """
    Configure the logging system for the application.
    
    What this does:
        - Creates a log file at logs/application.log
        - Sets the format to include timestamps and severity levels
        - Logs all messages at INFO level and above
        - Clears the previous log file on each app restart
    
    Log levels (least to most severe):
        DEBUG: Detailed diagnostic info
        INFO: Confirmation that things are working
        WARNING: Something unexpected happened (but app still works)
        ERROR: Something failed (but app continues)
        CRITICAL: Something catastrophic (app may stop)
    
    Python concepts used:
        - logging.basicConfig() for one-time configuration
        - logging.getLogger() to create named loggers
    """
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure the root logger
    logging.basicConfig(
        # Log file location
        filename="logs/application.log",
        # Overwrite log on each restart (use 'a' to append)
        filemode="w",
        # Log INFO and above (ignore DEBUG)
        level=logging.INFO,
        # Format: timestamp | level | module | message
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        # Date format for timestamps
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Log that the application started
    logging.info("=" * 60)
    logging.info("ONCOLINK APPLICATION STARTED")
    logging.info("=" * 60)

# =============================================================================
# SECTION 2: PAGE CONFIGURATION
# =============================================================================

# This function sets up the browser tab appearance.
# It MUST be the first Streamlit command in the file.
def configure_page():
    """
    Configure the Streamlit page settings.
    
    What this does:
        - Sets the browser tab title to "OncoLink"
        - Makes the page use the full screen width (wide layout)
        - Adds a small icon (heart with pulse) in the browser tab
    
    Python concepts used:
        - Function definition (def)
        - Calling functions with named parameters (parameter=value)
        - st.set_page_config() is a Streamlit configuration function
    """
    st.set_page_config(
        page_title="OncoLink",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# =============================================================================
# SECTION 3: CUSTOM CSS
# =============================================================================

def load_custom_css():
    """
    Add custom CSS styling to make the app look professional.
    
    What this does:
        - Injects CSS code into the page to override Streamlit's default styles
        - Changes colors, fonts, spacing, and button appearance
    
    Why CSS:
        Streamlit looks functional but basic by default. A small amount of CSS
        makes it look like a real SaaS product.
    
    Python concepts used:
        - Multi-line string (triple quotes)
        - st.markdown() with unsafe_allow_html=True
          This tells Streamlit to render raw HTML/CSS in the page.
          "unsafe" means: don't filter out HTML tags. We trust our own code.
    """
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


# =============================================================================
# SECTION 4: SIDEBAR
# =============================================================================

def create_sidebar():
    """
    Build the sidebar navigation panel.
    
    What this does:
        - Shows project name and icon at the top
        - Lists all project phases with current status (done or pending)
        - Shows which phase we are currently in
    
    Why a sidebar:
        Gives users a quick overview of the project scope.
        Shows recruiters that we have planned the full roadmap.
    
    Python concepts used:
        - st.sidebar: Streamlit's sidebar container
        - st.markdown(): Render text with Markdown formatting
        - st.divider(): A horizontal line separator
        - f-strings: For embedding variables in strings (f"...{var}...")
        - Lists and loops: We use data structures to define phases
    """
    
    # --- Sidebar header ---
    st.sidebar.markdown(
        '<div style="text-align:center; padding: 1rem 0 0.5rem 0;">'
        '<span style="font-size:2.2rem; font-weight:800; background:linear-gradient(135deg,#0f2b4a,#2d9cdb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">🏥 OncoLink</span>'
        '<br><span style="font-size:0.8rem; color:#64748b; font-weight:400;">AI Clinical Intelligence Platform</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<div style="height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:0.5rem 0 1rem 0;"></div>',
        unsafe_allow_html=True
    )
    
    # --- API Key Status ---
    st.sidebar.markdown(
        '<div style="font-size:0.85rem; font-weight:700; color:#0f2b4a; text-transform:uppercase; letter-spacing:0.5px; margin:1rem 0 0.5rem 0;">🔑 API Status</div>',
        unsafe_allow_html=True
    )
    
    env_key = os.getenv("NVIDIA_API_KEY")
    if env_key == "nvapi-your-key-here":
        env_key = None
        
    if not env_key:
        api_key_input = st.sidebar.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Get a free key at: https://build.nvidia.com/",
            value=st.session_state.get("nvidia_api_key_input", "")
        )
        if api_key_input:
            st.session_state["nvidia_api_key_input"] = api_key_input
            os.environ["NVIDIA_API_KEY"] = api_key_input
            api_key = api_key_input
        else:
            api_key = None
    else:
        api_key = env_key
    
    if api_key:
        st.sidebar.markdown(
            '<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:0.5rem 0.8rem;">'
            '<span style="color:#16a34a; font-weight:600; font-size:0.85rem;">✅ NVIDIA API connected</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:0.5rem 0.8rem;">'
            '<span style="color:#dc2626; font-weight:600; font-size:0.85rem;">❌ API key not configured</span>'
            '<br><span style="color:#94a3b8; font-size:0.75rem;">Enter your key above to enable AI features.</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # --- Version Info at the bottom ---
    st.sidebar.markdown(
        '<div style="height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:1.5rem 0 0.5rem 0;"></div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<div style="text-align:center; font-size:0.75rem; color:#94a3b8;">v3.0.0 | Jul 2026</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# SECTION 5: FILE UPLOAD SECTION
# =============================================================================

def create_file_uploader():
    """
    Create the file upload area where users drop their documents.
    
    What this does:
        - Shows a drag-and-drop zone
        - Accepts PDF, TXT, and DOCX files
        - Lets users upload multiple files at once
        - Returns the list of uploaded files
    
    Python concepts used:
        - st.file_uploader(): Streamlit's built-in file upload widget
        - accept_multiple_files=True: Allows uploading several files at once
        - type=[...]: Restricts which file formats are accepted
        - Function return value: gives the list of files back to the caller
    """
    
    # Create a section header
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem;">'
        '<span style="font-size:1.3rem;">📄</span>'
        '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">Upload Documents</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem;">Upload oncology reports to generate a structured patient summary.</p>',
        unsafe_allow_html=True
    )
    
    # Streamlit's file uploader widget
    # Parameters explained:
    #   label: The text shown above the upload area
    #   type: List of allowed file extensions
    #   accept_multiple_files: True = upload many files at once
    #   key: Unique identifier for this widget (needed if you have multiple uploaders)
    uploaded_files = st.file_uploader(
        label="Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="document_uploader"
    )
    
    # Return the list of uploaded files to the caller
    # If no files uploaded, this returns an empty list
    return uploaded_files


# =============================================================================
# SECTION 6: DISPLAY UPLOADED FILES
# =============================================================================

def display_uploaded_files(uploaded_files):
    """
    Show the list of uploaded files in a clean format.
    
    What this does:
        - Checks if any files were uploaded
        - If yes: shows each filename with a document icon
        - If no: shows a helpful message asking the user to upload files
    
    Python concepts used:
        - if/else: Conditional logic to handle both cases
        - len(list): Count items in a list
        - for loop: Display each file one by one
        - f-strings: Embed variables in text
    """
    
    # Check if the uploaded_files list has any items
    # len() returns the number of items in the list
    if len(uploaded_files) > 0:
        # We have files - show a success message with count
        st.markdown(f"**📎 Uploaded Files ({len(uploaded_files)} total)**")
        
        # Create a container to group the file list visually
        file_list_container = st.container()
        
        # Loop through each uploaded file
        with file_list_container:
            for file in uploaded_files:
                # For each file, show:
                # - A document emoji
                # - The original filename (file.name)
                # - The file size in KB
                # file.size gives bytes; we divide by 1024 to get kilobytes
                file_size_kb = file.size / 1024
                
                # Display the file in a styled div
                st.markdown(
                    f'<div class="file-item">📄 {file.name} '
                    f'<span style="color: #9E9E9E; font-size: 0.8rem;">'
                    f'({file_size_kb:.1f} KB)</span></div>',
                    unsafe_allow_html=True
                )
    else:
        # No files uploaded yet - show the empty state message
        st.markdown(
            '<div class="placeholder-box">'
            '📂 No files uploaded yet. Use the uploader above to add documents.'
            '</div>',
            unsafe_allow_html=True
        )


# =============================================================================
# SECTION 7: PROCESS DOCUMENTS BUTTON
# =============================================================================

def create_process_button(uploaded_files):
    """
    Create the "Process Documents" button that reads all uploaded files.
    
    What this does:
        - Shows a large blue button labeled "Process Documents"
        - When clicked, processes each uploaded file through document_loader
        - The document_loader detects the file type and calls the correct reader
        - Results are stored in session_state so they survive page reruns
        - Shows a success message with the number of processed files
    
    Why this replaces the old "Analyze" placeholder:
        Phase 1 had a fake button that just showed a message.
        Phase 2 has a real button that actually reads documents.
    
    Parameters:
        uploaded_files (list): List of Streamlit UploadedFile objects.
            Empty list if no files uploaded.
    
    Returns:
        bool: True if processing was done, False otherwise.
    
    Python concepts used:
        - Function parameters
        - session_state for persisting data across reruns
        - len() to check if list has items
        - st.button() disabled parameter
        - For loop over uploaded files
        - st.spinner() for showing progress
    """
    
    # Create a centered column layout for the button
    # st.columns([1, 2, 1]) creates 3 columns with ratios 1:2:1
    # The button goes in the middle (largest) column, centering it
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Determine if the button should be disabled
        # If no files are uploaded, the button is grayed out and unclickable
        # len(uploaded_files) == 0 means "no files uploaded"
        has_files = len(uploaded_files) > 0
        
        # Create the button
        # type="primary" makes it blue (the main action button)
        # use_container_width=True makes it fill the column width
        # disabled=not has_files means: disable if no files
        process_clicked = st.button(
            "📄 Process Documents",
            type="primary",
            use_container_width=True,
            disabled=(not has_files)
        )
        
        # If the button was just clicked (process_clicked is True)
        if process_clicked:
            
            # Show a spinner while processing
            # st.spinner() shows a loading animation while the code inside
            # the "with" block runs. Once done, the spinner disappears.
            with st.spinner("Reading documents... Please wait."):
                
                # Initialize an empty list to store results
                results = []
                
                # Loop through each uploaded file
                # For each file, we call process_uploaded_file() which
                # detects the file type, calls the correct reader, and
                # returns a dictionary with the extracted text.
                for file in uploaded_files:
                    result = process_uploaded_file(file)
                    results.append(result)
                
                # Store the results in session_state so they survive
                # page reruns. Without this, the results would be lost
                # when Streamlit re-runs the script.
                st.session_state["processed_files"] = results
                st.session_state["processing_done"] = True
            
            # Show a success message
            st.success(f"✅ Successfully processed {len(results)} document(s)!")
    
    # Return whether processing has been done
    # If "processing_done" exists in session_state, return its value
    # Otherwise, return False (default value from .get())
    return st.session_state.get("processing_done", False)


# =============================================================================
# SECTION 8: DISPLAY EXTRACTED TEXT
# =============================================================================

def display_extracted_text():
    """
    Show the extracted text from processed documents.
    
    What this does:
        - Checks if processed files exist in session_state
        - If yes: shows each file's extracted text in an expandable section
        - If no: shows nothing (the section is hidden until processing is done)
        - Shows a green checkmark for successful reads
        - Shows a red X for failed reads with the error message
    
    Why expandable sections:
        - Some oncology reports are 5-10 pages long
        - Showing all text at once would make the page very long
        - Expandable sections let users click to see the text they care about
    
    Python concepts used:
        - Checking if a key exists in session_state
        - For loop over a list of dictionaries
        - Accessing dictionary values by key (result["filename"])
        - st.expander() for collapsible sections
        - st.text() for displaying pre-formatted text
        - st.warning() and st.error() for status messages
    """
    
    # Check if we have processed files stored in session_state
    # The "in" keyword checks if a key exists in a dictionary
    # We only show this section AFTER documents have been processed
    if "processed_files" in st.session_state:
        
        # Get the list of results from session_state
        results = st.session_state["processed_files"]
        
        st.markdown(
            '<div style="display:flex; align-items:center; gap:0.5rem; margin-top:1.5rem;">'
            '<span style="font-size:1.3rem;">📖</span>'
            '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">Extracted Text</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem; margin-bottom:1rem;">Below is the raw text extracted from each document. Click to expand.</p>',
            unsafe_allow_html=True
        )
        
        # Loop through each processed file's results
        for result in results:
            
            # Extract values from the result dictionary
            filename = result["filename"]     # The original file name
            file_type = result["file_type"]   # "PDF", "DOCX", "TXT", or "UNKNOWN"
            text = result["text"]             # The extracted text content
            success = result["success"]       # True if reading was successful
            
            # Choose an icon based on success status
            if success:
                status_icon = "✅"
            else:
                status_icon = "❌"
            
            # Create an expandable section for this file
            # The label shows: icon + filename + (file type)
            # Example: "✅ pathology_report.pdf (PDF)"
            with st.expander(f"{status_icon} {filename} ({file_type})"):
                
                if success:
                    # Check if the extracted text is empty or only whitespace
                    # .strip() removes leading/trailing spaces and newlines
                    # If nothing is left after stripping, the text is effectively empty
                    if len(text.strip()) > 0:
                        # Display the extracted text in a monospace font
                        # st.text() preserves whitespace and line breaks
                        st.text(text)
                    else:
                        # File was read but no text was found
                        st.warning("This file contains no extractable text. It may be a scanned document.")
                else:
                    # File could not be read - show the error message
                    # The "text" variable contains the error message from the reader
                    st.error(text)
        
        # Add a tip for the user
        st.info("💡 Documents are ready. Use the 'Run AI Extraction' section below to extract structured clinical data.")


# =============================================================================
# SECTION 9: AI CLINICAL EXTRACTION
# =============================================================================

def create_extraction_section():
    """
    Create the AI extraction button and results display.
    
    What this does:
        - Shows a "Run AI Extraction" button (only after documents are processed)
        - When clicked, combines all document text and sends it to the LLM
        - Calls extract_from_multiple_documents() which handles the full pipeline
        - Displays the extracted structured data in professional cards
        - Handles errors gracefully (missing API key, API failures, bad JSON)
    
    Why this is separate from the process button:
        Document processing (Phase 2) and AI extraction (Phase 3) are
        different operations. Keeping them separate means:
        1. User can process documents without using API credits
        2. User can re-run AI extraction without re-processing documents
        3. Each button has a single, clear responsibility
    
    Python concepts used:
        - Checking session_state for prerequisites
        - st.button() with conditional enabling
        - Error handling with try/except
        - Calling functions from other modules
    """
    
    # Only show this section after documents have been processed
    if "processed_files" in st.session_state:
        
        # Add a visual separator
        st.divider()
        
        # Section heading
        st.markdown(
            '<div style="display:flex; align-items:center; gap:0.5rem; margin-top:1.5rem;">'
            '<span style="font-size:1.3rem;">🤖</span>'
            '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">AI Clinical Extraction</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem;">'
            'Extract structured clinical information from all documents using NVIDIA\'s Nemotron-3 Ultra AI model.'
            '</p>',
            unsafe_allow_html=True
        )
        
        # Check if API key is configured before showing the button
        api_key = os.getenv("NVIDIA_API_KEY")
        api_ready = api_key is not None and api_key != "nvapi-your-key-here" and len(api_key.strip()) > 0
        
        if not api_ready:
            # API key is not configured
            st.warning(
                "⚠️ NVIDIA API key not found.\n\n"
                "To enable AI extraction, please **enter your NVIDIA API Key in the sidebar input field**.\n\n"
                "Get a free API key at: https://build.nvidia.com/"
            )
        
        # Create a centered column layout for the button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # The extraction button is only enabled if:
            # 1. Documents are processed (we have text to extract from)
            # 2. API key is configured
            extraction_clicked = st.button(
                "🧠 Run AI Extraction",
                type="primary",
                use_container_width=True,
                disabled=(not api_ready)
            )
            
            if extraction_clicked:
                # Run the extraction with a spinner
                with st.spinner("Calling NVIDIA AI... This may take 10-30 seconds."):
                    
                    # Get the processed files from session_state
                    processed_files = st.session_state["processed_files"]
                    
                    # Run extraction on all documents combined
                    result = extract_from_multiple_documents(processed_files)
                    
                    # Also run per-document extraction for harmonization
                    per_doc_results = extract_each_document(processed_files)
                    
                    # Store the extraction results in session_state
                    st.session_state["extraction_result"] = result
                    st.session_state["per_doc_results"] = per_doc_results
                    st.session_state["extraction_done"] = True
                    st.session_state["harmonization_done"] = False
                
                # Show feedback based on result
                if result["success"]:
                    st.success("✅ AI extraction completed successfully!")
                    
                    # Show validation warnings if any
                    warnings = result.get("warnings", [])
                    if len(warnings) > 0:
                        for warning in warnings:
                            st.warning(f"⚠️ {warning}")
                else:
                    # Check for validation-specific errors
                    errors = result.get("errors", [])
                    if len(errors) > 0:
                        error_message = "\n".join(errors)
                        st.error(f"❌ Extraction failed:\n{error_message}")
                    else:
                        st.error("❌ Extraction failed with an unknown error.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # =====================================================================
        # Display extraction results (if available)
        # =====================================================================
        if "extraction_result" in st.session_state:
            result = st.session_state["extraction_result"]
            
            if result["success"]:
                # Display a validation status badge
                validation_info = result.get("validation", {})
                if validation_info and validation_info.get("valid"):
                    st.success("✅ Data validated against Pydantic schema")
                
                # Display normalization info
                normalization_info = result.get("normalization", {})
                if normalization_info and normalization_info.get("success"):
                    st.info("🔄 Values normalized for consistency")
                
                # Show the structured clinical data
                display_structured_clinical_data(result["data"])
            else:
                # Show errors with details
                errors = result.get("errors", [])
                for error in errors:
                    st.error(f"❌ {error}")
                
                # Show raw response for debugging if available
                if result.get("raw_response"):
                    with st.expander("🔧 Debug: Raw LLM Response"):
                        st.text(result["raw_response"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # =====================================================================
        # SECTION 9b: HARMONIZATION & SUMMARY (Phase 5)
        # =====================================================================
        if "extraction_done" in st.session_state and st.session_state["extraction_done"]:
            show_harmonization_section()


def show_harmonization_section():
    """
    Phase 5: Harmonization, conflict detection, and summary generation.
    """
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-top:2rem;">'
        '<span style="font-size:1.3rem;">🔗</span>'
        '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">Harmonization & Summary</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem;">Merge data from all documents and generate a unified patient summary.</p>',
        unsafe_allow_html=True
    )
    
    harmonization_done = st.session_state.get("harmonization_done", False)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        harmonize_clicked = st.button(
            "🔄 Run Harmonization",
            type="primary",
            use_container_width=True,
            disabled=harmonization_done
        )
    
    if harmonize_clicked:
        with st.spinner("Harmonizing data across documents..."):
            per_doc_results = st.session_state.get("per_doc_results", [])
            
            harmonized = harmonize_extractions(per_doc_results)
            st.session_state["harmonized_data"] = harmonized
            
            conflicts = detect_conflicts(per_doc_results)
            st.session_state["conflicts"] = conflicts
            
            if harmonized["success"]:
                summary = generate_patient_summary(
                    harmonized["data"],
                    harmonized.get("field_sources"),
                    conflicts.get("conflicts", [])
                )
                st.session_state["summary"] = summary
            
            st.session_state["harmonization_done"] = True
        
        if harmonized["success"]:
            st.success(f"✅ Harmonization complete! {harmonized['documents_merged']} document(s) merged.")
        else:
            st.error("❌ Harmonization failed.")
    
    # Display results
    if harmonization_done:
        harmonized = st.session_state.get("harmonized_data", {})
        conflicts = st.session_state.get("conflicts", {})
        summary = st.session_state.get("summary", {})
        
        # Harmonized data
        if harmonized.get("success") and harmonized.get("data"):
            with st.expander("📊 Harmonized Patient Record", expanded=True):
                display_structured_clinical_data(harmonized["data"])
        
        # Conflicts — full detail view
        conflict_list = conflicts.get("conflicts", [])
        if conflict_list:
            high_count = sum(1 for c in conflict_list if c["severity"] == "high")
            st.markdown(
                f'<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin:1rem 0 0.5rem 0;">⚠️ Conflicts Detected ({len(conflict_list)})</div>',
                unsafe_allow_html=True
            )
            for c in conflict_list:
                severity_icon = "🔴" if c["severity"] == "high" else "🟡"
                st.markdown(
                    f'<div style="background:#ffffff; border:1px solid #e2e8f0; border-left:3.5px solid {"#dc2626" if c["severity"] == "high" else "#eab308"}; border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.6rem;">'
                    f'<div style="font-weight:600; color:#0f2b4a; font-size:0.9rem; margin-bottom:0.4rem;">{severity_icon} {c["description"]}</div>',
                    unsafe_allow_html=True
                )
                for doc, val in c["sources"]:
                    st.markdown(
                        f'<div style="display:flex; gap:0.5rem; font-size:0.85rem; padding:0.2rem 0; color:#475569;">'
                        f'<span style="font-weight:500; color:#64748b; min-width:120px;">📄 {doc}</span>'
                        f'<span style="color:#94a3b8;">→</span>'
                        f'<span style="color:#0f2b4a;">{val}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:1.2rem; text-align:center; margin:1rem 0;">'
                '<span style="color:#16a34a; font-weight:500;">All documents agree on shared fields. No conflicts detected.</span>'
                '</div>',
                unsafe_allow_html=True
            )

        # Patient summary
        if summary.get("success") and summary.get("summary"):
            st.markdown(
                '<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin:1.5rem 0 0.5rem 0;">📋 Patient Summary</div>',
                unsafe_allow_html=True
            )
            st.code(summary["summary"], language="text")
            summary_text = summary["summary"]
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇ Download .txt",
                    data=summary_text,
                    file_name="patient_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col_b:
                try:
                    from fpdf import FPDF
                    pdf = FPDF(orientation="P", unit="mm", format="A4")
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Courier", "", 9)
                    for line in summary_text.split("\n"):
                        clean = line.encode("latin-1", errors="replace").decode("latin-1")
                        pdf.cell(0, 4.5, clean, new_x="LMARGIN", new_y="NEXT")
                    pdf_bytes = pdf.output(dest="S").encode("latin-1")
                    st.download_button(
                        "⬇ Download .pdf",
                        data=pdf_bytes,
                        file_name="patient_summary.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception:
                    st.download_button(
                        "⬇ Download .pdf",
                        data=summary_text.encode("utf-8"),
                        file_name="patient_summary.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
        
        # Export
        if harmonized.get("success") and harmonized.get("data"):
            st.markdown(
                '<div style="height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:1.5rem 0 1rem 0;"></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin-bottom:0.5rem;">📥 Export Harmonized Data</div>',
                unsafe_allow_html=True
            )
            with st.expander("View Harmonized JSON"):
                import json
                formatted = json.dumps(harmonized["data"], indent=2)
                st.code(formatted, language="json")


def display_structured_clinical_data(data):
    """
    Display the extracted clinical data in professional cards.
    
    What this does:
        - Takes the extracted JSON data dictionary
        - Displays each field in a clean card layout
        - Groups related fields together (patient info, cancer info, etc.)
        - Handles null values gracefully (shows "Not found")
        - Handles the biomarkers list specially (shows each as a sub-card)
    
    Parameters:
        data (dict): The extracted clinical data from the LLM.
            Keys are field names like "patient_name", "diagnosis", etc.
            Values are strings, numbers, lists, or None.
    
    Python concepts used:
        - Dictionary iteration with .items()
        - Type checking with isinstance()
        - Nested loops (for biomarkers which is a list of dicts)
        - Conditional formatting based on value presence
        - st.metric() for key-value display
        - st.container() for grouping
    """
    
    # Check if data is None or empty
    if data is None:
        st.warning("No clinical data was extracted.")
        return
    
    # =========================================================================
    # Section 1: Patient Information
    # =========================================================================
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">'
        '<span style="font-size:1.1rem;">👤</span>'
        '<span style="font-size:1.05rem; font-weight:700; color:#0f2b4a;">Patient Information</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Create 3 columns for patient info (name, age, gender)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Display patient name
        value = data.get("patient_name")
        if value:
            st.metric(label="Patient Name", value=value)
        else:
            st.metric(label="Patient Name", value="Not found")
    
    with col2:
        # Display age
        value = data.get("age")
        if value is not None:
            st.metric(label="Age", value=f"{value} years")
        else:
            st.metric(label="Age", value="Not found")
    
    with col3:
        # Display gender
        value = data.get("gender")
        if value:
            st.metric(label="Gender", value=value)
        else:
            st.metric(label="Gender", value="Not found")
    
    st.markdown("---")
    
    # =========================================================================
    # Section 2: Cancer Information
    # =========================================================================
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">'
        '<span style="font-size:1.1rem;">🧬</span>'
        '<span style="font-size:1.05rem; font-weight:700; color:#0f2b4a;">Cancer Information</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Create 3 columns for cancer info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Display diagnosis
        value = data.get("diagnosis")
        if value:
            # Use a container for longer text (diagnosis can be multi-line)
            st.markdown("**Diagnosis**")
            st.info(value)
        else:
            st.metric(label="Diagnosis", value="Not found")
    
    with col2:
        # Display cancer type
        value = data.get("cancer_type")
        if value:
            st.metric(label="Cancer Type", value=value)
        else:
            st.metric(label="Cancer Type", value="Not found")
        
        # Display cancer stage below cancer type
        value = data.get("cancer_stage")
        if value:
            st.markdown("**Cancer Stage**")
            st.info(value)
        else:
            st.metric(label="Cancer Stage", value="Not found")
    
    with col3:
        # Display ECOG score
        value = data.get("ecog_score")
        if value is not None:
            st.metric(label="ECOG Score", value=str(value))
        else:
            st.metric(label="ECOG Score", value="Not found")
    
    st.markdown("---")
    
    # =========================================================================
    # Section 3: Biomarkers
    # =========================================================================
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">'
        '<span style="font-size:1.1rem;">🔬</span>'
        '<span style="font-size:1.05rem; font-weight:700; color:#0f2b4a;">Biomarkers</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    biomarkers = data.get("biomarkers")
    
    if biomarkers and len(biomarkers) > 0:
        # Display each biomarker in a row
        # Create columns dynamically based on number of biomarkers
        biomarker_cols = st.columns(len(biomarkers))
        
        for index, biomarker in enumerate(biomarkers):
            with biomarker_cols[index]:
                # Each biomarker has: name, value, status
                name = biomarker.get("name", "Unknown")
                value = biomarker.get("value", "N/A")
                status = biomarker.get("status", "N/A")
                
                st.markdown(f"**{name}**")
                st.markdown(f"Value: {value}")
                st.markdown(f"Status: {status}")
    else:
        st.info("No biomarker information found in the documents.")
    
    st.markdown("---")
    
    # =========================================================================
    # Section 4: Treatment Information (Two columns)
    # =========================================================================
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">'
        '<span style="font-size:1.1rem;">💊</span>'
        '<span style="font-size:1.05rem; font-weight:700; color:#0f2b4a;">Treatment Information</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current medication
        value = data.get("current_medication")
        st.markdown("**Current Medication**")
        if value:
            st.info(value)
        else:
            st.info("Not found")
    
    with col2:
        # Previous treatment
        value = data.get("previous_treatment")
        st.markdown("**Previous Treatment**")
        if value:
            st.info(value)
        else:
            st.info("Not found")
    
    st.markdown("---")
    
    # =========================================================================
    # Section 5: Plan (Two columns)
    # =========================================================================
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">'
        '<span style="font-size:1.1rem;">📋</span>'
        '<span style="font-size:1.05rem; font-weight:700; color:#0f2b4a;">Plan</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Follow-up plan
        value = data.get("follow_up_plan")
        st.markdown("**Follow-up Plan**")
        if value:
            st.info(value)
        else:
            st.info("Not found")
    
    with col2:
        # Next steps
        value = data.get("next_steps")
        st.markdown("**Next Steps**")
        if value:
            st.info(value)
        else:
            st.info("Not found")
    
    # =========================================================================
    # Raw JSON Export
    # =========================================================================
    st.markdown(
        '<div style="height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:1.5rem 0 1rem 0;"></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin-bottom:0.5rem;">📥 Export Data</div>',
        unsafe_allow_html=True
    )
    
    with st.expander("View Raw JSON"):
        import json
        formatted_json = json.dumps(data, indent=2)
        st.code(formatted_json, language="json")


# =============================================================================
# SECTION 10: PLACEHOLDER SECTIONS
# =============================================================================

def create_placeholder_sections(processing_done):
    extraction_done = st.session_state.get("extraction_done", False)
    harmonization_done = st.session_state.get("harmonization_done", False)
    harmonized = st.session_state.get("harmonized_data", {})
    conflicts = st.session_state.get("conflicts", {})
    summary = st.session_state.get("summary", {})
    
    st.markdown('<div class="section-heading">📋 Patient Summary</div>',
                unsafe_allow_html=True)
    
    if harmonization_done and summary.get("success") and summary.get("summary"):
        st.code(summary["summary"], language="text")
    elif extraction_done:
        st.markdown(
            '<div class="placeholder-box">'
            '📊 Extraction complete. Click "Run Harmonization" above to generate the patient summary.'
            '</div>',
            unsafe_allow_html=True
        )
    elif processing_done:
        st.markdown(
            '<div class="placeholder-box">'
            '📊 Documents processed. Click "Run AI Extraction" above.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="placeholder-box">'
            '📊 No data yet. Upload documents and click "Process Documents".'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-heading">⚠️ Warnings & Conflicts</div>',
                    unsafe_allow_html=True)
        
        if harmonization_done:
            conflict_list = conflicts.get("conflicts", [])
            if conflict_list:
                for c in conflict_list:
                    severity_icon = "🔴" if c["severity"] == "high" else "🟡"
                    st.markdown(
                        f'<div style="background:#ffffff; border:1px solid #e2e8f0; border-left:3.5px solid {"#dc2626" if c["severity"] == "high" else "#eab308"}; border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:0.5rem;">'
                        f'<div style="font-weight:600; color:#0f2b4a; font-size:0.85rem;">{severity_icon} {c["description"]}</div>',
                        unsafe_allow_html=True
                    )
                    for doc, val in c["sources"]:
                        st.markdown(
                            f'<div style="font-size:0.8rem; padding:0.1rem 0; color:#475569; display:flex; gap:0.5rem;">'
                            f'<span style="color:#64748b; min-width:100px;">{doc}</span>'
                            f'<span style="color:#94a3b8;">→</span>'
                            f'<span style="color:#0f2b4a;">{val}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="placeholder-box">'
                    '✅ No conflicts detected across documents.'
                    '</div>',
                    unsafe_allow_html=True
                )
        elif extraction_done:
            st.markdown(
                '<div class="placeholder-box">'
                '🔍 Run "Harmonization" to detect data conflicts across documents.'
                '</div>',
                unsafe_allow_html=True
            )
        elif processing_done:
            st.markdown(
                '<div class="placeholder-box">'
                '🔄 Run AI extraction to identify potential data issues.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="placeholder-box">'
                '✅ No warnings.'
                '</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown('<div class="section-heading">📊 Structured Data</div>',
                    unsafe_allow_html=True)
        
        if harmonization_done and harmonized.get("success"):
            st.markdown(
                f'<div class="placeholder-box">'
                f'✅ {harmonized["documents_merged"]} document(s) harmonized. '
                f'View the unified record above.'
                f'</div>',
                unsafe_allow_html=True
            )
        elif extraction_done:
            st.markdown(
                '<div class="placeholder-box">'
                '✅ Structured data extracted. Run "Harmonization" to merge all documents.'
                '</div>',
                unsafe_allow_html=True
            )
        elif processing_done:
            st.markdown(
                '<div class="placeholder-box">'
                '⏳ Click "Run AI Extraction" to extract structured data.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="placeholder-box">'
                '⏸️ Waiting for analysis.'
                '</div>',
                unsafe_allow_html=True
            )


# =============================================================================
# SECTION 9: FOOTER
# =============================================================================

def create_footer():
    """
    Add a simple footer at the bottom of the page.
    
    What this does:
        - Shows a horizontal line
        - Displays project info and copyright
    
    Python concepts used:
        - st.divider(): Horizontal line
        - st.caption(): Small, muted text (good for footers)
        - Multi-line string
    """
    st.markdown(
        '<div style="height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:2rem 0 1rem 0;"></div>',
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<span style="font-size:0.8rem; color:#94a3b8;">🏥 OncoLink v1.0.0</span>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align:center; font-size:0.8rem; color:#94a3b8;">AI Clinical Intelligence Platform</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="text-align:right; font-size:0.8rem; color:#94a3b8;">Phase 5: Harmonization & Summary</div>', unsafe_allow_html=True)


# =============================================================================
# SECTION 10: MAIN FUNCTION (Application Entry Point)
# =============================================================================

def main():
    """
    Main function that runs the entire application.
    
    This is the entry point — Python starts executing here when you run:
        streamlit run app.py
    
    What this does:
        - Loads environment variables from .env file
        - Calls all the helper functions in the right order
        - Orchestrates the whole page layout
    
    Why a main() function:
        - Keeps the top-level code clean
        - Makes it easy to understand the app flow at a glance
        - Good practice for organizing Python programs
    
    Python concepts used:
        - Function calls: We call all our helper functions here
        - if __name__ == "__main__": Standard Python pattern that ensures
          main() only runs when this file is executed directly (not imported)
    """
    
    # Step 0: Load environment variables from .env file
    # This must happen early so all API keys are available
    load_dotenv()
    
    # Step 0.5: Set up logging
    setup_logging()
    
    # Step 1: Configure the page (must be first)
    configure_page()
    
    # Step 2: Load custom CSS styles
    load_custom_css()
    
    # Step 3: Build the sidebar
    create_sidebar()
    
    # Step 4: Main page header
    col_logo, col_text = st.columns([0.08, 0.92])
    with col_logo:
        st.markdown(
            '<div style="font-size:2.5rem; line-height:1;">🏥</div>',
            unsafe_allow_html=True
        )
    with col_text:
        st.markdown('<div class="main-title">OncoLink</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">AI Clinical Intelligence Platform</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="main-description">Upload oncology reports — referral letters, pathology reports, lab reports, and EMR notes — and generate structured, unified patient summaries.</div>', unsafe_allow_html=True)
    
    # Add a visual separator
    st.divider()
    
    # Step 5: Create the file uploader and get uploaded files
    uploaded_files = create_file_uploader()
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 6: Display the uploaded files
    display_uploaded_files(uploaded_files)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 7: Create the Process Documents button
    # We pass uploaded_files so the button knows if there are files to process
    # The button is disabled (grayed out) when no files are uploaded
    processing_done = create_process_button(uploaded_files)
    
    # Step 8: Display extracted text (only shows after processing)
    display_extracted_text()
    
    # Step 9: AI Clinical Extraction (Phase 3)
    # This section appears after document processing is done
    # It contains the "Run AI Extraction" button and results display
    create_extraction_section()
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    # Step 10: Show the placeholder sections
    create_placeholder_sections(processing_done)
    
    # Step 11: Show footer
    create_footer()


# =============================================================================
# SECTION 11: SCRIPT ENTRY POINT
# =============================================================================

# This is the standard Python pattern for a runnable script.
# 
# What it does:
#   - When you run "streamlit run app.py", Python sets __name__ to "__main__"
#   - This means: "If this file is the one being executed, call main()"
#   - If this file were imported by another file, __name__ would be "app"
#     and main() would NOT run automatically
#
# Why this matters:
#   - It lets us reuse functions from this file in other files without
#     accidentally running the whole app
#   - This is a standard best practice in Python

if __name__ == "__main__":
    main()
