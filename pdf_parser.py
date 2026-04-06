"""
pdf_parser.py
Extracts clean text from uploaded PDF resumes using PyPDF2.
Skills used: Python file handling, string processing, NLP preprocessing
"""

import re
import io
import PyPDF2


class PDFParser:

    def extract_text(self, file_bytes: bytes) -> str:
        """
        Accepts raw PDF bytes (from Streamlit file_uploader),
        extracts and cleans text from all pages.
        """
        text_pages = []
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        except Exception as e:
            return f"[PDF Error: {str(e)}]"

        raw = "\n".join(text_pages)
        return self._clean(raw)

    def _clean(self, text: str) -> str:
        """
        Cleans extracted PDF text:
        - Removes garbled characters
        - Normalises whitespace and newlines
        - Keeps structure readable
        """
        # Fix broken hyphenated words (PDF line-breaks)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Collapse multiple blank lines to single
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove non-printable / control characters
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
        # Normalise spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    def get_page_count(self, file_bytes: bytes) -> int:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            return len(reader.pages)
        except Exception:
            return 0
