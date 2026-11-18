"""Semantic validation sau LLM: schema hợp lệ chưa đủ, quote phải thật sự nằm trong chunk.

Đây là kiểm tra xác định (deterministic) - chạy trước mọi LLM reviewer.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel
from rapidfuzz import fuzz

from research_scientist.schemas.paper_card import KEY_FIELDS, Evidence, PaperCard


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_chunk(quote: str, chunk_text: str, threshold: int = 90) -> bool:
    q, c = _norm(quote), _norm(chunk_text)
    if not q:
        return False
    if q in c:
        return True
    # Số liệu phải khớp tuyệt đối: "52.4%" không được coi là gần đúng với "52.3%"
    if any(num not in c for num in _NUMBER.findall(q)):
        return False
    # cho phép sai khác nhỏ (dấu nối, ký tự PDF) bằng partial ratio
    return fuzz.partial_ratio(q, c) >= threshold


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
