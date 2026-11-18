"""Structure-aware chunking: cắt theo section/paragraph thay vì theo số token cố định.

- Không bao giờ ghép paragraph của hai section khác nhau vào một chunk.
- Gom paragraph liên tiếp cho đến khi gần `max_chars`; paragraph quá dài thì cắt theo câu.
- Mỗi chunk giữ section_path, trang, offset ký tự trong section -> làm evidence span.
- Văn bản đưa đi embed có tiền tố "section_path" để giữ ngữ cảnh heading.

~4 ký tự ≈ 1 token tiếng Anh, nên max_chars=2000 ≈ 500 token.
"""

import re
from dataclasses import dataclass

from research_scientist.ingestion.grobid import Paragraph, ParsedSection

CHUNKER_VERSION = "struct-v1"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
SEPARATOR = "\n\n"


@dataclass
class ChunkDraft:
    ordinal: int
    section_index: int
    section_path: str
    page: int | None
    char_start: int  # offset trong ParsedSection.text
    char_end: int
    text: str

    @property
    def embed_text(self) -> str:
        return f"{self.section_path}\n{self.text}"


def _split_long(p: Paragraph, max_chars: int) -> list[Paragraph]:
    if len(p.text) <= max_chars:
        return [p]
    pieces: list[Paragraph] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(p.text):
        if buf and len(buf) + 1 + len(sentence) > max_chars:
            pieces.append(Paragraph(buf, p.page))
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
        while len(buf) > max_chars:  # câu dài bất thường: cắt cứng
            pieces.append(Paragraph(buf[:max_chars], p.page))
            buf = buf[max_chars:]
    if buf:
        pieces.append(Paragraph(buf, p.page))
    return pieces


def chunk_sections(sections: list[ParsedSection], max_chars: int = 2000) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    for s_idx, section in enumerate(sections):
        # offset của từng paragraph gốc trong section.text
        cursor = 0
        units: list[tuple[int, Paragraph]] = []
        for para in section.paragraphs:
            start = cursor
            for piece in _split_long(para, max_chars):
                found = section.text.find(piece.text, start)
                pos = found if found >= 0 else start
                units.append((pos, piece))
                start = pos + len(piece.text)
            cursor += len(para.text) + len(SEPARATOR)

        buf: list[tuple[int, Paragraph]] = []

        def flush() -> None:
            if not buf:
                return
            text = SEPARATOR.join(p.text for _, p in buf)
            start = buf[0][0]
            end = buf[-1][0] + len(buf[-1][1].text)
            chunks.append(
                ChunkDraft(
                    ordinal=len(chunks),
                    section_index=s_idx,
                    section_path=section.path,
                    page=buf[0][1].page,
                    char_start=start,
                    char_end=end,
                    text=text,
                )
            )
            buf.clear()

        size = 0
        for pos, piece in units:
            extra = len(piece.text) + (len(SEPARATOR) if buf else 0)
            if buf and size + extra > max_chars:
                flush()
                size = 0
                extra = len(piece.text)
            buf.append((pos, piece))
            size += extra
        flush()
    return chunks
