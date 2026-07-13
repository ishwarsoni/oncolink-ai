"""
=============================================================================
text_reader.py - Extract text from TXT files
=============================================================================

What this module does:
    Takes a TXT file as raw bytes and converts it to a Python string.

Why a separate module for something so simple?
    Even though reading a text file is just one line of Python, having it
    in its own module means:
    1. If we need to add encoding detection later, we change one file.
    2. The document_loader can treat ALL file types the same way.
    3. It keeps the code consistent (every file type has its own reader).

What is encoding?
    Computers store text as numbers (bytes). Encoding is the system that
    maps those numbers to letters. "A" might be stored as the number 65.
    
    UTF-8 is the most common encoding on the web and modern systems.
    It can represent almost every character from every language.
    
    Sometimes files are saved in older encodings like "latin-1" (Windows
    Western European) or "cp1252". If we use the wrong encoding, we'll
    see garbled text like "Ã©" instead of "é".

Why we try UTF-8 first:
    UTF-8 is the standard for modern text files. If the file was created
    by any modern application (VS Code, Word, Google Docs → download as txt),
    it will be UTF-8.
=============================================================================
"""


def read_txt(file_bytes, filename):
    """
    Extract text from a TXT file.
    
    Parameters:
        file_bytes (bytes): The raw bytes of the uploaded TXT file.
        
        filename (str): The name of the file (for error messages).
    
    Returns:
        str: The text content of the file.
            If the file can't be decoded, returns an error message.
    
    How bytes become text:
        Bytes are raw numbers (e.g., b'Hello' is [72, 101, 108, 108, 111]).
        The .decode() method converts these numbers to a Python string
        using a specific encoding (a mapping from numbers to characters).
    
    Python concepts used:
        - bytes.decode() method
        - try/except with specific exception type (UnicodeDecodeError)
        - Exception chaining (try UTF-8 first, fall back to latin-1)
    """
    
    try:
        # =========================================================================
        # Step 1: Try UTF-8 encoding first
        # =========================================================================
        # .decode("utf-8") converts the bytes to a string using UTF-8 rules.
        # This works for >95% of modern text files.
        # 
        # If the file contains bytes that are NOT valid UTF-8, Python raises
        # a UnicodeDecodeError. We catch this and try a different encoding.
        text = file_bytes.decode("utf-8")
        return text
        
    except UnicodeDecodeError:
        # =========================================================================
        # Step 2: Fallback to latin-1 encoding
        # =========================================================================
        # latin-1 (also called ISO 8859-1) is an older encoding that can
        # decode ANY byte value (0-255). It won't crash, but characters
        # might look wrong if the file was actually in a different encoding.
        # 
        # This is our "safe fallback" - it always works but might not be
        # perfectly accurate.
        try:
            text = file_bytes.decode("latin-1")
            return text
            
        except Exception as error:
            # If even latin-1 fails (extremely rare), return an error
            error_message = f"Error reading TXT file '{filename}': {error}"
            return error_message
