from research_scientist.agents.evidence import quote_in_chunk, verify_card
from research_scientist.schemas.paper_card import (
    DatasetUse,
    Evidence,
    ListItem,
    PaperCard,
    ResultEntry,
    TextField,
)

CHUNKS = {
    "c1": "We study long-tailed image classification where rare classes have few samples.",
    "c2": "Our method re-weights the loss by effective number of samples per class.",
    "c3": "On ImageNet-LT, top-1 accuracy improves to 52.3%.",
}


def _empty() -> TextField:
    return TextField(value=None, missing_reason="not stated")


def test_quote_matching_tolerates_whitespace_but_not_fabrication():
    assert quote_in_chunk("re-weights  the loss", CHUNKS["c2"])
    assert not quote_in_chunk("uses a transformer backbone", CHUNKS["c2"])
    # gần giống về chữ nhưng sai số -> phải bị loại
    assert not quote_in_chunk("top-1 accuracy improves to 52.4%", CHUNKS["c3"])


def test_quote_matching_handles_pdf_artifacts_from_real_papers():
    chunk = (
        "Intuitively, a better estimation of the ef- fective number of samples could be "
        "obtained. This makes our loss generally applicable. We deﬁne the weighting factor."
    )
    # line-break hyphenation + ligature
    assert quote_in_chunk("a better estimation of the effective number of samples", chunk)
    assert quote_in_chunk("We define the weighting factor", chunk)
    # ellipsis joining two real segments, in order
    assert quote_in_chunk("Intuitively, a better estimation ... generally applicable", chunk)
    # segments out of order or invented segment must fail
    assert not quote_in_chunk("generally applicable ... Intuitively, a better estimation", chunk)
    assert not quote_in_chunk("Intuitively ... we use a transformer backbone", chunk)


def test_verify_card_counts_verified_unknown_and_mismatch():
    card = PaperCard(
        research_problem=TextField(
            value="Long-tailed classification",
            evidence=[Evidence(chunk_id="c1", quote="long-tailed image classification")],
        ),
        hypothesis=_empty(),
        contributions=[],
        method=TextField(
            value="Class-balanced loss",
            evidence=[Evidence(chunk_id="c9", quote="re-weights the loss")],  # chunk không tồn tại
        ),
        datasets=[
            DatasetUse(name="ImageNet-LT", evidence=[Evidence(chunk_id="c3", quote="ImageNet-LT")])
        ],
        baselines=[],
        metrics=[
            ListItem(value="top-1", evidence=[Evidence(chunk_id="c3", quote="top-1 accuracy")])
        ],
        results=[
            ResultEntry(
                metric="top-1",
                value="55.0%",
                evidence=[Evidence(chunk_id="c3", quote="improves to 55.0%")],  # bịa số
            )
        ],
        limitations=[],
        threats_to_validity=[],
        future_work=[],
    )
    report = verify_card(card, CHUNKS)
    assert report.total == 5
    assert report.verified == 3
    assert report.unknown_chunk == ["method.evidence[0]"]
    assert report.quote_mismatch == ["results[0].evidence[0]"]
    # research_problem, datasets, metrics có evidence hợp lệ; method/results/limitations thì không
    assert report.key_fields_with_evidence == 3
    assert report.key_field_coverage == 0.5
