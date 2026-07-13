"""
=============================================================================
document_loader.py - Document Type Detector and Router
=============================================================================

What this module does:
    Acts as a "controller" or "orchestrator" for document reading.
    
    When given an uploaded file, it:
    1. Looks at the file extension (.pdf, .docx, .txt)
    2. Calls the correct reader module
    3. Returns the extracted text

Why this design (Controller Pattern):
    Imagine if app.py had to check file types and call different readers
    directly. Every time we add a new file type (like .csv or .dcm for
    medical images), we'd need to change app.py. That's bad because:
    
    - app.py is about the USER INTERFACE, not file processing logic
    - Mixing UI code and business logic makes the code hard to maintain
    - Testing becomes harder (you can't test file reading without the UI)
    
    With document_loader.py as a controller:
    - app.py just calls ONE function: process_uploaded_file()
    - To add a new file type, we edit ONLY this file
    - To fix PDF reading, we edit ONLY pdf_reader.py
    - Each file has ONE responsibility

How it works:
    uploaded_file (from Streamlit)
        ↓
    document_loader.py checks .name → gets extension
        ↓
    .pdf → calls pdf_reader.read_pdf()
    .docx → calls docx_reader.read_docx()
    .txt → calls text_reader.read_txt()
        ↓
    Returns dictionary with filename, text, file_type, success
=============================================================================
"""

# Import the operating system module for working with file paths
# os.path.splitext() splits a filename into (name, extension)
import os

# Import our three reader modules
# Each handles one file type
from backend.pdf_reader import read_pdf
from backend.docx_reader import read_docx
from backend.text_reader import read_txt


def process_uploaded_file(uploaded_file):
    """
    Process a single uploaded file by detecting its type and reading it.
    
    Parameters:
        uploaded_file: A Streamlit UploadedFile object.
            This object has:
            - .name: The original filename (e.g., "report.pdf")
            - .size: File size in bytes
            - .read(): Method that returns the file content as bytes
    
    Returns:
        dict: A dictionary with these keys:
            - "filename": Original filename (e.g., "pathology_report.pdf")
            - "text": Extracted text content (string)
            - "file_type": Detected type ("PDF", "DOCX", "TXT", or "UNKNOWN")
            - "success": True if reading succeeded, False if error
    
    Python concepts used:
        - Importing from other modules in our project
        - Dictionary as a return value (groups related data together)
        - os.path.splitext() for file extension extraction
        - if/elif/else for routing logic
        - Calling functions and using their return values
    """
    
    # =========================================================================
    # Step 1: Get file information
    # =========================================================================
    # uploaded_file.name contains the original filename from the user's computer
    # Example: "Meera_Sharma_Pathology_Report.pdf"
    filename = uploaded_file.name
    
    # Read the file content as bytes
    # uploaded_file.read() returns the entire file as a bytes object
    # Bytes are raw data - 0s and 1s grouped into bytes
    file_bytes = uploaded_file.read()
    
    # =========================================================================
    # Step 2: Detect file type from extension
    # =========================================================================
    # os.path.splitext() splits a filename into two parts:
    #   - The base name (everything before the last dot)
    #   - The extension (everything after the last dot, including the dot)
    #
    # Example: os.path.splitext("report.pdf") → ("report", ".pdf")
    #
    # We take [1] to get the extension part, then convert to lowercase
    # so that ".PDF" and ".pdf" are treated the same.
    file_extension = os.path.splitext(filename)[1].lower()
    
    # =========================================================================
    # Step 3: Route to the correct reader
    # =========================================================================
    # We check the file extension and call the appropriate reader function.
    # Each reader returns the extracted text as a string.
    # 
    # If the extension doesn't match any known type, we return an error.
    
    # These variables will store our results
    text = ""
    file_type = ""
    success = True
    
    # Check extension and route to the right reader
    if file_extension == ".pdf":
        # Call the PDF reader with the file bytes
        text = read_pdf(file_bytes, filename)
        file_type = "PDF"
        
    elif file_extension == ".docx":
        # Call the DOCX reader
        text = read_docx(file_bytes, filename)
        file_type = "DOCX"
        
    elif file_extension == ".txt":
        # Call the text reader
        text = read_txt(file_bytes, filename)
        file_type = "TXT"
        
    else:
        # Unknown file type - return an error message
        text = f"Unsupported file format: '{file_extension}'. Please upload PDF, DOCX, or TXT files."
        file_type = "UNKNOWN"
        success = False
    
    # =========================================================================
    # Step 4: Check if the reader encountered an error
    # =========================================================================
    # Our reader functions return error messages (strings starting with "Error")
    # when something goes wrong. We check if the text starts with "Error" to
    # determine if the reading was successful.
    #
    # This is a simple approach. A more advanced approach would be to raise
    # exceptions, but checking string prefixes is easier for beginners.
    if text.startswith("Error"):
        success = False
    
    # =========================================================================
    # Step 5: Return results as a dictionary
    # =========================================================================
    # A dictionary lets us return multiple values with clear labels.
    # The caller can access each value by name (result["text"], result["filename"])
    # instead of remembering the order of values.
    result = {
        "filename": filename,
        "text": text,
        "file_type": file_type,
        "success": success
    }
    
    return result
