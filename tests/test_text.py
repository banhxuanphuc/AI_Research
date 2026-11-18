import pymupdf

from research_scientist.ingestion.pdf_fallback import parse_pdf_pages
from research_scientist.ingestion.text import clean_text


def test_clean_text_removes_nul_and_controls_keeps_layout():
    raw = "Class\x00-Balanced\x07 Loss\r\nline two\tTab\x9f"
    assert clean_text(raw) == "Class-Balanced Loss\nline two\tTab"


def test_clean_text_normalizes_unicode():
    # "e" + combining acute -> single precomposed character
    assert clean_text("résumé") == "résumé"


def test_pymupdf_fallback_output_has_no_nul():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Hello world")
    parsed = parse_pdf_pages(doc.tobytes())
    assert parsed.sections
    assert all("\x00" not in p.text for s in parsed.sections for p in s.paragraphs)
