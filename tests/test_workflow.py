"""Workflow chạy end-to-end với LLM giả + checkpointer in-memory, kiểm tra 2 điểm human approval."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research_scientist.providers.gateway import LLMGateway
from research_scientist.schemas.research import PaperCandidate, ResearchBrief
from research_scientist.workflows.research_graph import Deps, build_research_graph
from tests.fakes import FakeProvider

BRIEF = ResearchBrief(
    scope="Long-tailed image classification, 2019-2024.",
    research_questions=["Which re-weighting methods work best on ImageNet-LT?"],
    inclusion_criteria=["image classification"],
    exclusion_criteria=["object detection"],
    assumptions=["Focus on supervised learning"],
    search_queries=["long-tailed image classification", "class-balanced loss"],
)


async def _noop(record) -> None:
    pass


async def fake_discover(brief, inp):
    assert brief.search_queries == BRIEF.search_queries
    return [
        PaperCandidate(title="Class-Balanced Loss Based on Effective Number of Samples", doi="10.1/cb"),
        PaperCandidate(title="Class-balanced loss based on effective number of samples", doi="10.1/cb"),
        PaperCandidate(title="Decoupling Representation and Classifier", arxiv_id="1910.09217"),
    ]


def _graph():
    gw = LLMGateway(
        providers={"anthropic": FakeProvider(lambda m, s, u: BRIEF, cost=0.02)}, recorder=_noop
    )
    return build_research_graph(Deps(gateway=gw, discover=fake_discover)).compile(
        checkpointer=InMemorySaver()
    )


async def test_workflow_pauses_for_brief_then_selection():
    graph = _graph()
    config = {"configurable": {"thread_id": "t1"}}

    await graph.ainvoke({"brief_input": {"topic": "long-tail"}, "cost_usd": 0.0}, config)
    snap = await graph.aget_state(config)
    assert snap.next == ("approve_brief",)
    assert snap.tasks[0].interrupts[0].value["type"] == "approve_brief"
    assert snap.values["cost_usd"] == 0.02

    await graph.ainvoke(Command(resume={"approved": True}), config)
    snap = await graph.aget_state(config)
    assert snap.next == ("select_papers",)
    assert len(snap.values["candidates"]) == 2  # trùng DOI đã được gộp

    await graph.ainvoke(Command(resume={"selected_indices": [1, 99]}), config)
    snap = await graph.aget_state(config)
    assert snap.next == ()
    assert [c["title"] for c in snap.values["selected"]] == ["Decoupling Representation and Classifier"]


async def test_rejected_brief_ends_without_discovery():
    graph = _graph()
    config = {"configurable": {"thread_id": "t2"}}
    await graph.ainvoke({"brief_input": {"topic": "x"}, "cost_usd": 0.0}, config)
    await graph.ainvoke(Command(resume={"approved": False}), config)
    snap = await graph.aget_state(config)
    assert snap.next == ()
    assert snap.values["status"] == "brief_rejected"
    assert "candidates" not in snap.values
