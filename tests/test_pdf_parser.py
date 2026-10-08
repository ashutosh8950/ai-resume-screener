from types import SimpleNamespace

import pytest

from resume_screener import pdf_parser
from resume_screener.pdf_parser import PDFParseError, PDFParser


def test_parse_rejects_empty_pdf():
    with pytest.raises(PDFParseError, match="empty"):
        PDFParser().parse(b"")


def test_parse_rejects_corrupt_pdf():
    with pytest.raises(PDFParseError, match="couldn't read"):
        PDFParser().parse(b"not a pdf")


def test_parse_rejects_scanned_pdf(monkeypatch):
    monkeypatch.setattr(
        pdf_parser.PyPDF2,
        "PdfReader",
        lambda _stream: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "")]),
    )

    with pytest.raises(PDFParseError, match="No selectable text"):
        PDFParser().parse(b"valid-looking-bytes")


def test_parse_rejects_page_limit(monkeypatch):
    monkeypatch.setattr(
        pdf_parser.PyPDF2,
        "PdfReader",
        lambda _stream: SimpleNamespace(pages=[object(), object()]),
    )

    with pytest.raises(PDFParseError, match="maximum allowed is 1"):
        PDFParser(max_pages=1).parse(b"valid-looking-bytes")


def test_parse_rejects_upload_size_limit():
    with pytest.raises(PDFParseError, match="upload limit"):
        PDFParser(max_upload_mb=1).parse(b"x" * (1024 * 1024 + 1))


def test_parse_returns_text_and_page_count_from_reader(monkeypatch):
    pages = [SimpleNamespace(extract_text=lambda: "Python developer")]
    monkeypatch.setattr(
        pdf_parser.PyPDF2,
        "PdfReader",
        lambda _stream: SimpleNamespace(pages=pages),
    )

    text, page_count = PDFParser().parse(b"valid-looking-bytes")

    assert text == "Python developer"
    assert page_count == 1