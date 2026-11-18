"""Upload PDF hợp pháp -> MinIO -> enqueue job ingest; xem sections/chunks đã parse."""

import asyncio
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.api.deps import current_workspace_id
from research_scientist.api.routers.projects import get_owned_project
from research_scientist.core.db import get_session
from research_scientist.domain.models import (
    AccessStatus,
    Chunk,
    Paper,
    PaperVersion,
    ProjectPaper,
    ProjectPaperStatus,
)
from research_scientist.ingestion.dedup import normalize_doi, normalize_title
from research_scientist.ingestion.storage import ObjectStore, sha256

router = APIRouter(tags=["papers"])

MAX_PDF_BYTES = 50 * 1024 * 1024


class UploadOut(BaseModel):
    paper_id: uuid.UUID
    version_id: uuid.UUID
    job_id: str | None


class PaperOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    doi: str | None
    year: int | None
    venue: str | None


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    ordinal: int
    section_path: str
    page: int | None
    text: str


@router.post("/projects/{project_id}/papers/upload", response_model=UploadOut, status_code=202)
async def upload_pdf(
    project_id: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    doi: str | None = Form(None),
    license: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> UploadOut:
    await get_owned_project(project_id, session, workspace_id)
    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF quá lớn (tối đa 50 MB)")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "File không phải PDF")

    title = file.filename or "uploaded.pdf"
    paper = Paper(
        title=title,
        title_normalized=normalize_title(title),
        doi=normalize_doi(doi),
        source="user_upload",
    )
    session.add(paper)
    await session.flush()

    checksum = sha256(data)
    key = f"{workspace_id}/{paper.id}/{checksum}.pdf"
    uri = await asyncio.to_thread(ObjectStore().put, key, data, "application/pdf")
    version = PaperVersion(
        paper_id=paper.id,
        file_uri=uri,
        checksum_sha256=checksum,
        license=license,
        access_status=AccessStatus.user_uploaded,
    )
    session.add(version)
    session.add(
        ProjectPaper(project_id=project_id, paper_id=paper.id, status=ProjectPaperStatus.included)
    )
    await session.commit()

    job = await request.app.state.arq.enqueue_job("ingest_pdf", str(version.id), str(workspace_id))
    return UploadOut(paper_id=paper.id, version_id=version.id, job_id=job.job_id if job else None)


@router.get("/projects/{project_id}/papers", response_model=list[PaperOut])
async def list_project_papers(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> list[Paper]:
    await get_owned_project(project_id, session, workspace_id)
    stmt = (
        select(Paper)
        .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
        .where(ProjectPaper.project_id == project_id)
    )
    return list((await session.scalars(stmt)).all())


@router.get("/papers/{paper_id}/chunks", response_model=list[ChunkOut])
async def list_chunks(
    paper_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .where(Chunk.paper_id == paper_id, Chunk.workspace_id == workspace_id)
        .order_by(Chunk.ordinal)
    )
    return list((await session.scalars(stmt)).all())
