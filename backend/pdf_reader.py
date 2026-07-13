"""
=============================================================================
pdf_reader.py - Extract text from PDF files
=============================================================================

What this module does:
    Takes a PDF file (as raw bytes) and extracts all text from it.
    Uses PyMuPDF (imported as "fitz") - a fast, beginner-friendly PDF library.

Why a separate module:
    PDF reading is complex. Keeping it in its own file means:
    1. If we need to change the PDF library later, we only change one file.
    2. Other files don't need to know HOW we read PDFs, only THAT we can.
    3. It's easy to test PDF reading in isolation.

Library used:
    PyMuPDF (imported as "fitz") - Reads PDFs from bytes, handles pages, 
    extracts text. Better than PyPDF2 because:
    - Handles more PDF types (scanned-with-text-layer, forms, etc.)
    - Faster
    - Simpler API (one function call per page)
=============================================================================
"""

# Import the PyMuPDF library
# We import it as "fitz" because that's the name of the underlying C library.
# This is a convention - all PyMuPDF documentation uses "import fitz".
import fitz


def read_pdf(file_bytes, filename):
    """
    Extract all text from a PDF file.
    
    Parameters:
        file_bytes (bytes): The raw bytes of the uploaded PDF file.
            When a user uploads a file in Streamlit, we get the file content
            as bytes. Bytes are just raw data - 0s and 1s.
            PyMuPDF knows how to interpret these bytes as a PDF.
        
        filename (str): The name of the file (e.g., "pathology_report.pdf").
            We only use this for error messages so the user knows which file
            had a problem.
    
    Returns:
        str: All the text found in the PDF, with pages separated by newlines.
            If an error occurs, returns an error message string.
    
    How it works:
        1. Open the PDF from bytes using fitz.open()
        2. Loop through each page in the PDF
        3. Extract text from each page using page.get_text()
        4. Combine all pages into one string
        5. Close the PDF and return the text
    
    Python concepts used:
        - Function with parameters
        - For loop (iterating over pages)
        - try/except for error handling
        - String concatenation (+=)
        - Calling methods on objects (pdf_document.get_text())
    """
    
    try:
        # =========================================================================
        # Step 1: Open the PDF from bytes
        # =========================================================================
        # fitz.open() can open PDFs from a file path OR from bytes in memory.
        # We use "stream=file_bytes" to pass the bytes, and "filetype='pdf'"
        # to tell fitz what kind of file this is.
        # 
        # Why open from bytes instead of saving to disk first?
        #   When a user uploads a file in Streamlit, the file stays in memory
        #   (RAM). Writing it to disk would be slower and create temporary files
        #   we'd need to clean up. Reading from bytes is faster and cleaner.
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Initialize an empty string to hold all the text.
        # We'll add each page's text to this string.
        all_text = ""
        
        # Get the total number of pages in the PDF.
        # len(pdf_document) returns the page count.
        total_pages = len(pdf_document)
        
        # =========================================================================
        # Step 2: Loop through each page and extract text
        # =========================================================================
        # We use range(total_pages) to get numbers from 0 to (total_pages - 1).
        # Page 0 is the first page, page 1 is the second, etc.
        # This is called "zero-indexing" - Python starts counting at 0.
        for page_number in range(total_pages):
            
            # Get the page object for the current page number
            page = pdf_document[page_number]
            
            # Extract text from this page.
            # page.get_text() returns all the text on the page as a string.
            # PyMuPDF reads the text in reading order (left-to-right, top-to-bottom).
            page_text = page.get_text()
            
            # Add this page's text to our accumulated string.
            # We add a newline character (\n) between pages to keep them separate.
            all_text = all_text + page_text + "\n"
        
        # =========================================================================
        # Step 3: Close the PDF document
        # =========================================================================
        # Always close files when done to free up memory.
        # This is like putting a book back on the shelf after reading it.
        pdf_document.close()
        
        # Return the combined text from all pages
        return all_text
    
    except Exception as error:
        # =========================================================================
        # Error handling
        # =========================================================================
        # If ANYTHING goes wrong in the try block above, we jump here.
        # Possible errors:
        #   - The file is corrupted and can't be opened as a PDF
        #   - The file is password-protected
        #   - The PDF has no text layer (scanned document)
        #
        # Instead of crashing the app, we return a helpful error message.
        # The filename is included so the user knows which file caused the problem.
        error_message = f"Error reading PDF file '{filename}': {error}"
        return error_message
