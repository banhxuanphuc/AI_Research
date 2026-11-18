"""Chạy / tiếp tục workflow LangGraph có human-in-the-loop."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.api.deps import current_workspace_id
from research_scientist.api.routers.discovery import scholarly_clients
from research_scientist.api.routers.projects import get_owned_project
from research_scientist.core.config import get_settings
from research_scientist.core.db import get_session
from research_scientist.domain.models import WorkflowRun
from research_scientist.ingestion.scholarly import discover
from research_scientist.providers.gateway import build_gateway
from research_scientist.schemas.research import (
    PaperCandidate,
    ResearchBrief,
    ResearchBriefInput,
)
from research_scientist.workflows.research_graph import GRAPH_NAME, Deps, build_research_graph

router = APIRouter(tags=["runs"])


class RunState(BaseModel):
    run_id: uuid.UUID
    thread_id: str
    status: str | None
    next: list[str]
    interrupts: list[Any]
    values: dict[str, Any]


async def _discover(brief: ResearchBrief, inp: ResearchBriefInput) -> list[PaperCandidate]:
    s2, arxiv = scholarly_clients()
    result = await discover(brief.search_queries, s2, arxiv, 10, inp.year_from, inp.year_to)
    # TODO(tuần 5): đưa result.errors vào state để UI hiển thị nguồn nào bị lỗi
    return result.candidates


def _graph(request: Request):
    deps = Deps(gateway=build_gateway(get_settings().run_budget_usd), discover=_discover)
    return build_research_graph(deps).compile(checkpointer=request.app.state.checkpointer)


async def _snapshot(graph, run: WorkflowRun) -> RunState:
    snap = await graph.aget_state({"configurable": {"thread_id": run.thread_id}})
    interrupts = [i.value for task in snap.tasks for i in task.interrupts]
    return RunState(
        run_id=run.id,
        thread_id=run.thread_id,
        status=snap.values.get("status"),
        next=list(snap.next),
        interrupts=interrupts,
        values=snap.values,
    )


async def _invoke(session: AsyncSession, run: WorkflowRun, graph, payload: Any, config: dict) -> None:
    """Chạy graph tới interrupt tiếp theo; lỗi thì đánh dấu run failed (checkpoint vẫn còn,
    có thể resume sau khi sửa nguyên nhân, vd thiếu ANTHROPIC_API_KEY)."""
    try:
        await graph.ainvoke(payload, config)
    except Exception as exc:
        run.status = "failed"
        await session.commit()
        raise HTTPException(502, f"Workflow lỗi ở bước LLM/tool: {type(exc).__name__}: {exc}") from exc


async def _sync_run(session: AsyncSession, run: WorkflowRun, state: RunState) -> None:
    run.status = "done" if not state.next else (state.status or "running")
    run.cost_usd = float(state.values.get("cost_usd", 0.0))
    await session.commit()


@router.post("/projects/{project_id}/runs", response_model=RunState, status_code=201)
async def start_run(
    project_id: uuid.UUID,
    body: ResearchBriefInput,
    request: Request,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> RunState:
    await get_owned_project(project_id, session, workspace_id)
    run = WorkflowRun(project_id=project_id, graph=GRAPH_NAME, thread_id=str(uuid.uuid4()))
    session.add(run)
    await session.commit()

    graph = _graph(request)
    config = {"configurable": {"thread_id": run.thread_id}}
    initial = {"project_id": str(project_id), "brief_input": body.model_dump(), "cost_usd": 0.0}
    await _invoke(session, run, graph, initial, config)
    state = await _snapshot(graph, run)
    await _sync_run(session, run, state)
    return state


async def _get_run(session: AsyncSession, run_id: uuid.UUID, workspace_id: uuid.UUID) -> WorkflowRun:
    run = await session.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    await get_owned_project(run.project_id, session, workspace_id)
    return run


@router.get("/runs/{run_id}", response_model=RunState)
async def get_run(
    run_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> RunState:
    run = await _get_run(session, run_id, workspace_id)
    return await _snapshot(_graph(request), run)


@router.post("/runs/{run_id}/resume", response_model=RunState)
async def resume_run(
    run_id: uuid.UUID,
    decision: dict[str, Any],
    request: Request,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> RunState:
    """`decision` tuỳ theo interrupt đang chờ:
    - approve_brief: {"approved": true, "brief": {...tuỳ chọn}}
    - select_papers: {"selected_indices": [0, 2, 5]}
    """
    run = await _get_run(session, run_id, workspace_id)
    graph = _graph(request)
    config = {"configurable": {"thread_id": run.thread_id}}
    if not (await graph.aget_state(config)).next:
        raise HTTPException(409, "Run đã kết thúc")
    await _invoke(session, run, graph, Command(resume=decision), config)
    state = await _snapshot(graph, run)
    await _sync_run(session, run, state)
    return state
