"""
Utility functions for file handling and processing.
"""
from pypdf import PdfReader

def read_pdf(path):
    """
    Reads a PDF file from the specified path and extracts all its text content.

    Args:
        path (str): The local file path to the PDF.

    Returns:
        str: The extracted text from all pages of the PDF.
    """
    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text