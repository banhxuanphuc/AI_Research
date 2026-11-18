"""Reciprocal Rank Fusion: gộp nhiều danh sách xếp hạng (dense, sparse, full-text).

score(d) = sum_i 1 / (k + rank_i(d)),  rank bắt đầu từ 1.
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, limit: int | None = None
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:limit] if limit else fused
