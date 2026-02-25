"""
pdf_utils.py - PDF text extraction and chunking utilities.
"""

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf_text(pdf_path: str) -> str:
    """Extract raw text from every page of a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)
