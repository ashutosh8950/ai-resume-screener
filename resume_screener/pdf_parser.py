"""Validated text extraction from uploaded PDFs."""

import io
import re

import PyPDF2

from resume_screener.config import MAX_PDF_PAGES, MAX_UPLOAD_MB
from resume_screener.logger import get_logger


logger = get_logger("pdf_parser")


class PDFParseError(ValueError):
    """Raised when an uploaded PDF cannot be safely used as resume text."""


class PDFParser:
    def __init__(self, max_pages: int = MAX_PDF_PAGES, max_upload_mb: int = MAX_UPLOAD_MB):
        self.max_pages = max_pages
        self.max_upload_bytes = max_upload_mb * 1024 * 1024

    def parse(self, file_bytes: bytes) -> tuple[str, int]:
        if not file_bytes:
            raise PDFParseError("The uploaded PDF is empty. Please choose a non-empty PDF.")
        if len(file_bytes) > self.max_upload_bytes:
            raise PDFParseError(
                f"The PDF exceeds the {self.max_upload_bytes // (1024 * 1024)} MB upload limit."
            )

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
            if page_count == 0:
                raise PDFParseError("The PDF has no pages. Please choose a valid resume PDF.")
            if page_count > self.max_pages:
                raise PDFParseError(
                    f"The PDF has {page_count} pages; the maximum allowed is {self.max_pages}."
                )
            text_pages = [page.extract_text() or "" for page in reader.pages]
        except PDFParseError:
            raise
        except Exception as error:
            logger.exception("Unable to parse uploaded PDF")
            raise PDFParseError(
                "We couldn't read this PDF. Please check that it is a valid, unencrypted PDF."
            ) from error

        text = self._clean("\n".join(text_pages))
        if not text:
            raise PDFParseError(
                "No selectable text was found. Scanned/image-only PDFs are not supported; "
                "please upload a text-based PDF or paste the resume text."
            )
        return text, page_count

    def extract_text(self, file_bytes: bytes) -> str:
        text, _ = self.parse(file_bytes)
        return text

    def get_page_count(self, file_bytes: bytes) -> int:
        _, page_count = self.parse(file_bytes)
        return page_count

    def _clean(self, text: str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()