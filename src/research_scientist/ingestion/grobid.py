"""Parse PDF khoa học bằng GROBID -> TEI XML -> cấu trúc section/paragraph có số trang.

Fallback PyMuPDF (pdf_fallback.py) khi GROBID lỗi hoặc chưa bật profile "ai".
"""

from dataclasses import dataclass, field

import httpx
from lxml import etree

from research_scientist.ingestion.text import clean_text

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
PARSER_NAME = "grobid"


@dataclass
class Paragraph:
    text: str
    page: int | None = None


@dataclass
class ParsedSection:
    path: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


@dataclass
class ParsedDocument:
    title: str | None
    abstract: str | None
    sections: list[ParsedSection]
    reference_count: int
    parser: str
    parser_version: str | None = None


class GrobidClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def version(self) -> str | None:
        try:
            resp = await self._client.get("/api/version", timeout=3.0)
            return resp.text.strip() if resp.status_code == 200 else None
        except httpx.HTTPError:
            return None

    async def process_fulltext(self, pdf_bytes: bytes) -> bytes:
        resp = await self._client.post(
            "/api/processFulltextDocument",
            files={"input": ("paper.pdf", pdf_bytes, "application/pdf")},
            data={"teiCoordinates": "p", "consolidateHeader": "0"},
        )
        resp.raise_for_status()
        return resp.content


def _text(el: etree._Element) -> str:
    return " ".join(clean_text("".join(el.itertext())).split())


def _first_page(el: etree._Element) -> int | None:
    coords = el.get("coords")
    if not coords:
        return None
    head = coords.split(";", 1)[0].split(",", 1)[0]
    return int(head) if head.isdigit() else None


def parse_tei(xml: bytes, parser_version: str | None = None) -> ParsedDocument:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    root = etree.fromstring(xml, parser=parser)

    title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)

    sections: list[ParsedSection] = []
    if abstract_el is not None and _text(abstract_el):
        sections.append(ParsedSection("Abstract", [Paragraph(_text(abstract_el), 1)]))

    for div in root.findall(".//tei:text/tei:body/tei:div", TEI_NS):
        head = div.find("tei:head", TEI_NS)
        number = head.get("n") if head is not None else None
        heading = _text(head) if head is not None else "Untitled"
        path = f"{number} {heading}".strip() if number else heading
        paras = [
            Paragraph(_text(p), _first_page(p)) for p in div.findall("tei:p", TEI_NS) if _text(p)
        ]
        # Caption của hình/bảng đi cùng section chứa nó
        for fig in div.findall(".//tei:figure", TEI_NS):
            desc = fig.find("tei:figDesc", TEI_NS)
            if desc is not None and _text(desc):
                label = "Table" if fig.get("type") == "table" else "Figure"
                paras.append(Paragraph(f"[{label}] {_text(desc)}", _first_page(fig)))
        if paras:
            sections.append(ParsedSection(path, paras))

    refs = root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS)
    return ParsedDocument(
        title=_text(title_el) if title_el is not None else None,
        abstract=_text(abstract_el) if abstract_el is not None else None,
        sections=sections,
        reference_count=len(refs),
        parser=PARSER_NAME,
        parser_version=parser_version,
    )
