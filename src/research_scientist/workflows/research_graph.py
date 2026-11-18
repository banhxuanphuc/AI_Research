"""Workflow chính (LangGraph) - phiên bản skeleton.

    START -> plan_brief -> [HUMAN: duyệt brief] -> discover -> [HUMAN: chọn paper] -> END

Các node tiếp theo (ingest, extract_cards, synthesize, review, write_report) sẽ thêm dần
theo lộ trình (tuần 6, 9, 10). Mỗi điểm [HUMAN] là một `interrupt()`: graph dừng, state được
checkpoint vào Postgres, API trả payload cho UI; khi người dùng duyệt, API gọi lại graph với
`Command(resume=...)`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from research_scientist.agents.brief_planner import plan_brief
from research_scientist.ingestion.dedup import dedup_candidates
from research_scientist.providers.gateway import LLMGateway
from research_scientist.schemas.research import PaperCandidate, ResearchBrief, ResearchBriefInput

GRAPH_NAME = "research_v0"

DiscoverFn = Callable[[ResearchBrief, ResearchBriefInput], Awaitable[list[PaperCandidate]]]


class ResearchState(TypedDict, total=False):
    project_id: str
    brief_input: dict[str, Any]
    brief: dict[str, Any]
    brief_approved: bool
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    cost_usd: float
    status: str


@dataclass
class Deps:
    gateway: LLMGateway
    discover: DiscoverFn
    max_candidates: int = 60


def build_research_graph(deps: Deps) -> StateGraph:
    async def plan_brief_node(state: ResearchState) -> dict[str, Any]:
        inp = ResearchBriefInput.model_validate(state["brief_input"])
        result = await plan_brief(deps.gateway, inp)
        return {
            "brief": result.output.model_dump(),
            "cost_usd": state.get("cost_usd", 0.0) + result.cost_usd,
            "status": "awaiting_brief_approval",
        }

    def approve_brief_node(state: ResearchState) -> dict[str, Any]:
        decision = interrupt({"type": "approve_brief", "brief": state["brief"]})
        # decision: {"approved": bool, "brief": <brief đã chỉnh sửa, tuỳ chọn>}
        if not decision.get("approved"):
            return {"brief_approved": False, "status": "brief_rejected"}
        edited = decision.get("brief")
        brief = ResearchBrief.model_validate(edited) if edited else None
        return {
            "brief_approved": True,
            "brief": brief.model_dump() if brief else state["brief"],
            "status": "discovering",
        }

    async def discover_node(state: ResearchState) -> dict[str, Any]:
        brief = ResearchBrief.model_validate(state["brief"])
        inp = ResearchBriefInput.model_validate(state["brief_input"])
        found = dedup_candidates(await deps.discover(brief, inp))[: deps.max_candidates]
        return {"candidates": [c.model_dump() for c in found], "status": "awaiting_selection"}

    def select_papers_node(state: ResearchState) -> dict[str, Any]:
        decision = interrupt({"type": "select_papers", "candidates": state["candidates"]})
        # decision: {"selected_indices": [int, ...]}
        idx = [i for i in decision.get("selected_indices", []) if 0 <= i < len(state["candidates"])]
        return {"selected": [state["candidates"][i] for i in idx], "status": "papers_selected"}

    def after_approval(state: ResearchState) -> str:
        return "discover" if state.get("brief_approved") else END

    g = StateGraph(ResearchState)
    g.add_node("plan_brief", plan_brief_node)
    g.add_node("approve_brief", approve_brief_node)
    g.add_node("discover", discover_node)
    g.add_node("select_papers", select_papers_node)
    g.add_edge(START, "plan_brief")
    g.add_edge("plan_brief", "approve_brief")
    g.add_conditional_edges("approve_brief", after_approval, ["discover", END])
    g.add_edge("discover", "select_papers")
    g.add_edge("select_papers", END)
    return g
