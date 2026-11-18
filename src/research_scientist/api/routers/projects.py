import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.api.deps import current_workspace_id
from research_scientist.core.db import get_session
from research_scientist.domain.models import ResearchProject

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    title: str
    topic: str


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    topic: str
    brief: dict | None
    created_at: datetime


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> ResearchProject:
    project = ResearchProject(workspace_id=workspace_id, title=body.title, topic=body.topic)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> list[ResearchProject]:
    stmt = (
        select(ResearchProject)
        .where(ResearchProject.workspace_id == workspace_id)
        .order_by(ResearchProject.created_at.desc())
    )
    return list((await session.scalars(stmt)).all())


async def get_owned_project(
    project_id: uuid.UUID, session: AsyncSession, workspace_id: uuid.UUID
) -> ResearchProject:
    project = await session.get(ResearchProject, project_id)
    if project is None or project.workspace_id != workspace_id:
        raise HTTPException(404, "Project not found")
    return project


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> ResearchProject:
    return await get_owned_project(project_id, session, workspace_id)
