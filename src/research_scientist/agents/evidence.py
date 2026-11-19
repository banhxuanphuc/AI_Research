"""Semantic validation sau LLM: schema hợp lệ chưa đủ, quote phải thật sự nằm trong chunk.

Đây là kiểm tra xác định (deterministic) - chạy trước mọi LLM reviewer.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from pydantic import BaseModel
from rapidfuzz import fuzz

from research_scientist.schemas.paper_card import KEY_FIELDS, Evidence, PaperCard


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(?=[a-z])")  # "con- tains" -> "contains" (PDF line break)
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ligatures: "ﬁ" -> "fi"
    text = _HYPHEN_BREAK.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _segment_in_chunk(q: str, c: str, threshold: int) -> bool:
    if q in c:
        return True
    # Số liệu phải khớp tuyệt đối: "52.4%" không được coi là gần đúng với "52.3%"
    if any(num not in c for num in _NUMBER.findall(q)):
        return False
    # cho phép sai khác nhỏ (ký tự PDF, dấu câu) bằng partial ratio
    return fuzz.partial_ratio(q, c) >= threshold


def quote_in_chunk(quote: str, chunk_text: str, threshold: int = 90) -> bool:
    c = _norm(chunk_text)
    segments = [s for s in (_norm(p) for p in _ELLIPSIS.split(quote)) if s]
    if not segments:
        return False
    if len(segments) == 1:
        return _segment_in_chunk(segments[0], c, threshold)
    # Quote nối bằng "...": mọi đoạn phải có trong chunk, theo đúng thứ tự
    pos = 0
    for seg in segments:
        idx = c.find(seg, pos)
        if idx < 0:
            if not _segment_in_chunk(seg, c[pos:], threshold):
                return False
            continue
        pos = idx + len(seg)
    return True


@dataclass
class EvidenceReport:
    total: int = 0
    verified: int = 0
    unknown_chunk: list[str] = field(default_factory=list)
    quote_mismatch: list[str] = field(default_factory=list)
    key_fields_with_evidence: int = 0

    @property
    def precision(self) -> float:
        return self.verified / self.total if self.total else 0.0

    @property
    def key_field_coverage(self) -> float:
        return self.key_fields_with_evidence / len(KEY_FIELDS)


def _collect(obj: object, path: str, out: list[tuple[str, Evidence]]) -> None:
    if isinstance(obj, Evidence):
        out.append((path, obj))
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            _collect(getattr(obj, name), f"{path}.{name}" if path else name, out)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _collect(item, f"{path}[{i}]", out)


def _has_verified_evidence(ok_paths: set[str], field_name: str) -> bool:
    return any(p == field_name or p.startswith((f"{field_name}.", f"{field_name}[")) for p in ok_paths)


def verify_card(card: PaperCard, chunks: dict[str, str]) -> EvidenceReport:
    """`chunks`: chunk_id -> text của các chunk đã đưa vào context."""
    items: list[tuple[str, Evidence]] = []
    _collect(card, "", items)
    report = EvidenceReport(total=len(items))
    ok_paths: set[str] = set()
    for path, ev in items:
        text = chunks.get(ev.chunk_id)
        if text is None:
            report.unknown_chunk.append(path)
        elif not quote_in_chunk(ev.quote, text):
            report.quote_mismatch.append(path)
        else:
            report.verified += 1
            ok_paths.add(path)
    report.key_fields_with_evidence = sum(
        _has_verified_evidence(ok_paths, f) for f in KEY_FIELDS
    )
    return report
