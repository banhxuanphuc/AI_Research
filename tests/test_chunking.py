from research_scientist.ingestion.chunking import chunk_sections
from research_scientist.ingestion.grobid import Paragraph, ParsedSection


def _section(path: str, *paras: str, page: int = 1) -> ParsedSection:
    return ParsedSection(path, [Paragraph(p, page) for p in paras])


def test_chunks_never_cross_sections():
    sections = [
        _section("1 Introduction", "a" * 100, "b" * 100),
        _section("2 Method", "c" * 100, page=3),
    ]
    chunks = chunk_sections(sections, max_chars=2000)
    assert [c.section_path for c in chunks] == ["1 Introduction", "2 Method"]
    assert chunks[1].page == 3
    assert chunks[1].section_index == 1


def test_offsets_point_back_into_section_text():
    sec = _section("3 Results", "First paragraph.", "Second paragraph.", "Third one here.")
    chunks = chunk_sections([sec], max_chars=40)
    assert len(chunks) >= 2
    for c in chunks:
        assert sec.text[c.char_start : c.char_end] == c.text


def test_long_paragraph_is_split_by_sentence_under_limit():
    long_para = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_sections([_section("4 Discussion", long_para)], max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert chunks[0].text.startswith("Sentence number 0")


def test_embed_text_carries_heading():
    chunks = chunk_sections([_section("2 Method > 2.1 Loss", "We use focal loss.")])
    assert chunks[0].embed_text.startswith("2 Method > 2.1 Loss\n")
