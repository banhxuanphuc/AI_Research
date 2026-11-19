"""Fallback khi GROBID không khả dụng: tách text theo trang bằng PyMuPDF.

Không có cấu trúc section, mỗi trang là một "section" -> chất lượng retrieval thấp hơn,
nên luôn ghi parser="pymupdf" để eval phân biệt được.
"""

import pymupdf

from research_scientist.ingestion.grobid import Paragraph, ParsedDocument, ParsedSection
from research_scientist.ingestion.text import clean_text


def parse_pdf_pages(pdf_bytes: bytes) -> ParsedDocument:
    sections: list[ParsedSection] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        title = clean_text((doc.metadata or {}).get("title") or "") or None
        for page_no, page in enumerate(doc, start=1):
            blocks = [t for b in page.get_text("blocks") if (t := clean_text(b[4]))]
            if blocks:
                sections.append(
                    ParsedSection(f"Page {page_no}", [Paragraph(b, page_no) for b in blocks])
                )
    return ParsedDocument(
        title=title,
        abstract=None,
        sections=sections,
        reference_count=0,
        parser="pymupdf",
        parser_version=pymupdf.VersionBind,
    )
