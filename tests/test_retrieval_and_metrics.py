import math

import pytest

from research_scientist.evals.metrics import (
    citation_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from research_scientist.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement_between_rankers():
    dense = ["a", "b", "c"]
    lexical = ["c", "a", "d"]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    ids = [doc for doc, _ in fused]
    assert ids[0] == "a"  # hạng 1 + hạng 2
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_limit_and_duplicate_ids():
    fused = reciprocal_rank_fusion([["x", "x", "y"]], limit=1)
    assert fused == [("x", pytest.approx(1 / 61))]


def test_ranking_metrics():
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = {"d2", "d4", "d9"}
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)
    assert reciprocal_rank(["x"], relevant) == 0.0


def test_ndcg_perfect_and_imperfect():
    rel = {"a": 2.0, "b": 1.0}
    assert ndcg_at_k(["a", "b"], rel, 2) == pytest.approx(1.0)
    worse = ndcg_at_k(["b", "a"], rel, 2)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert worse == pytest.approx(expected)


def test_citation_precision():
    assert citation_precision(["c1", "c2", "c3", "c4"], {"c1", "c2", "c3"}) == 0.75
    assert citation_precision([], {"c1"}) == 0.0
