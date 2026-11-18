"""Chuẩn hoá định danh và loại trùng paper từ nhiều nguồn."""

import re
import unicodedata

from rapidfuzz import fuzz

from research_scientist.schemas.research import PaperCandidate

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def normalize_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    doi = raw.strip()
    lower = doi.lower()
    for prefix in _DOI_PREFIXES:
        if lower.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    doi = doi.strip().lower()
    return doi if _DOI_RE.match(doi) else None


def normalize_arxiv_id(raw: str | None) -> str | None:
    """'http://arxiv.org/abs/2401.01234v2' -> '2401.01234' (bỏ version)."""
    if not raw:
        return None
    ident = raw.strip().rsplit("/abs/", 1)[-1]
    ident = ident.removeprefix("arXiv:").removeprefix("arxiv:")
    return re.sub(r"v\d+$", "", ident) or None


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", title)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _same_paper(a: PaperCandidate, b: PaperCandidate, title_threshold: int) -> bool:
    if a.doi and b.doi:
        return a.doi == b.doi
    if a.arxiv_id and b.arxiv_id and a.arxiv_id == b.arxiv_id:
        return True
    if a.s2_id and b.s2_id and a.s2_id == b.s2_id:
        return True
    if a.year and b.year and abs(a.year - b.year) > 1:
        return False
    score = fuzz.token_sort_ratio(normalize_title(a.title), normalize_title(b.title))
    return score >= title_threshold


def _merge(a: PaperCandidate, b: PaperCandidate) -> PaperCandidate:
    """Ưu tiên bản có peer review (không phải preprint) làm gốc, bổ sung trường còn thiếu."""
    base, other = (b, a) if a.is_preprint and not b.is_preprint else (a, b)
    data = base.model_dump()
    for key, value in other.model_dump().items():
        if key == "sources":
            continue
        if data.get(key) in (None, "", []) and value not in (None, "", []):
            data[key] = value
    data["sources"] = sorted(set(a.sources) | set(b.sources))
    data["is_preprint"] = a.is_preprint and b.is_preprint
    return PaperCandidate(**data)


def dedup_candidates(
    candidates: list[PaperCandidate], title_threshold: int = 95
) -> list[PaperCandidate]:
    merged: list[PaperCandidate] = []
    for cand in candidates:
        for i, existing in enumerate(merged):
            if _same_paper(existing, cand, title_threshold):
                merged[i] = _merge(existing, cand)
                break
        else:
            merged.append(cand)
    return merged
