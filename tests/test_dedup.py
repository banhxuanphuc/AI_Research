from research_scientist.ingestion.dedup import (
    dedup_candidates,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
)
from research_scientist.schemas.research import PaperCandidate


def test_normalize_doi_strips_prefix_and_lowercases():
    assert normalize_doi("https://doi.org/10.1109/CVPR.2016.90") == "10.1109/cvpr.2016.90"
    assert normalize_doi("doi:10.48550/arXiv.1706.03762") == "10.48550/arxiv.1706.03762"
    assert normalize_doi("not a doi") is None
    assert normalize_doi(None) is None


def test_normalize_arxiv_id_drops_version():
    assert normalize_arxiv_id("http://arxiv.org/abs/1706.03762v7") == "1706.03762"
    assert normalize_arxiv_id("arXiv:2401.01234v2") == "2401.01234"


def test_normalize_title():
    assert normalize_title("Attention Is All You Need!") == "attention is all you need"
    assert normalize_title("Café—Net: résumé") == "cafe net resume"


def test_dedup_merges_preprint_and_published_version():
    preprint = PaperCandidate(
        title="Deep Residual Learning for Image Recognition",
        arxiv_id="1512.03385",
        year=2015,
        is_preprint=True,
        open_access_pdf="https://arxiv.org/pdf/1512.03385",
        sources=["arxiv"],
    )
    published = PaperCandidate(
        title="Deep residual learning for image recognition",
        doi="10.1109/cvpr.2016.90",
        year=2016,
        venue="CVPR",
        sources=["semantic_scholar"],
    )
    other = PaperCandidate(title="Densely Connected Convolutional Networks", year=2017)

    merged = dedup_candidates([preprint, published, other])

    assert len(merged) == 2
    resnet = merged[0]
    assert resnet.doi == "10.1109/cvpr.2016.90"  # bản peer-reviewed làm gốc
    assert resnet.open_access_pdf == "https://arxiv.org/pdf/1512.03385"  # bổ sung từ preprint
    assert resnet.is_preprint is False
    assert resnet.sources == ["arxiv", "semantic_scholar"]


def test_dedup_different_dois_never_merge():
    a = PaperCandidate(title="Same Title", doi="10.1/a")
    b = PaperCandidate(title="Same Title", doi="10.1/b")
    assert len(dedup_candidates([a, b])) == 2
