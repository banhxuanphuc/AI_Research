"""Metric retrieval & grounding, tự viết để hiểu rõ công thức (Ragas/DeepEval thêm ở tuần 8)."""

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """`relevance`: doc_id -> độ liên quan (graded, vd 0/1/2)."""

    def dcg(gains: Sequence[float]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    gains = [relevance.get(doc, 0.0) for doc in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def citation_precision(cited: Sequence[str], supporting: set[str]) -> float:
    """Tỉ lệ citation trỏ đúng vào đoạn thật sự hỗ trợ claim (mục tiêu MVP >= 0.95)."""
    if not cited:
        return 0.0
    return sum(1 for c in cited if c in supporting) / len(cited)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
