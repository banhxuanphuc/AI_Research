"""Đưa paper vào project (upload PDF hoặc lấy open-access từ arXiv), xem chunk và PaperCard."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.api.deps import current_workspace_id
from research_scientist.api.routers.projects import get_owned_project
from research_scientist.core.db import get_session
from research_scientist.domain.models import (
    AccessStatus,
    Chunk,
    Paper,
    PaperCardRecord,
    ProjectPaper,
)
from research_scientist.ingestion.dedup import normalize_arxiv_id
from research_scientist.ingestion.scholarly import ArxivClient
from research_scientist.services.papers import Registered, register_pdf

router = APIRouter(tags=["papers"])

MAX_PDF_BYTES = 50 * 1024 * 1024


class IngestOut(BaseModel):
    paper_id: uuid.UUID
    version_id: uuid.UUID
    title: str
    is_new_version: bool
    job_id: str | None


class ArxivIn(BaseModel):
    arxiv_id: str = Field(examples=["1901.05555"])
    extract_card: bool = True


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


class CardOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    paper_id: uuid.UUID
    model: str
    prompt_version: str
    schema_version: str
    evidence_coverage: float | None
    evidence_report: dict
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    created_at: datetime
    card: dict


class JobOut(BaseModel):
    job_id: str | None


async def _enqueue_ingest(
    request: Request, reg: Registered, workspace_id: uuid.UUID, card: bool
) -> IngestOut:
    job = None
    # parser is set only after a successful ingest -> re-ingest versions that failed before
    if reg.is_new_version or reg.version.parser is None:
        job = await request.app.state.arq.enqueue_job(
            "ingest_pdf", str(reg.version.id), str(workspace_id), card
        )
    elif card:
        job = await request.app.state.arq.enqueue_job(
            "extract_card", str(reg.paper.id), str(workspace_id)
        )
    return IngestOut(
        paper_id=reg.paper.id,
        version_id=reg.version.id,
        title=reg.paper.title,
        is_new_version=reg.is_new_version,
        job_id=job.job_id if job else None,
    )


@router.post("/projects/{project_id}/papers/upload", response_model=IngestOut, status_code=202)
async def upload_pdf(
    project_id: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    doi: str | None = Form(None),
    license: str | None = Form(None),
    extract_card: bool = Form(False),
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> IngestOut:
    await get_owned_project(project_id, session, workspace_id)
    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF quá lớn (tối đa 50 MB)")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "File không phải PDF")
    reg = await register_pdf(
        session,
        project_id=project_id,
        workspace_id=workspace_id,
        pdf=data,
        access_status=AccessStatus.user_uploaded,
        title=file.filename or "uploaded.pdf",
        doi=doi,
        license=license,
    )
    return await _enqueue_ingest(request, reg, workspace_id, extract_card)


@router.post("/projects/{project_id}/papers/from-arxiv", response_model=IngestOut, status_code=202)
async def add_from_arxiv(
    project_id: uuid.UUID,
    body: ArxivIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> IngestOut:
    """Lấy metadata + PDF open-access từ arXiv, rồi ingest (và trích PaperCard nếu bật)."""
    await get_owned_project(project_id, session, workspace_id)
    arxiv_id = normalize_arxiv_id(body.arxiv_id)
    if not arxiv_id:
        raise HTTPException(422, "arXiv id không hợp lệ")
    client = ArxivClient()
    meta = await client.get_by_id(arxiv_id)
    if meta is None:
        raise HTTPException(404, f"Không tìm thấy arXiv:{arxiv_id}")
    try:
        pdf = await client.download_pdf(arxiv_id)
    except ValueError as exc:
        raise HTTPException(502, f"Không tải được PDF: {exc}") from exc
    reg = await register_pdf(
        session,
        project_id=project_id,
        workspace_id=workspace_id,
        pdf=pdf,
        access_status=AccessStatus.open_access,
        title=meta.title,
        meta=meta,
        license="arXiv",
    )
    return await _enqueue_ingest(request, reg, workspace_id, body.extract_card)


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


@router.post("/papers/{paper_id}/card", response_model=JobOut, status_code=202)
async def request_card(
    paper_id: uuid.UUID,
    request: Request,
    model: str | None = None,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> JobOut:
    """Xếp hàng trích PaperCard (chạy lại được, vd để so sánh model qua `?model=`)."""
    has_chunks = await session.scalar(
        select(Chunk.id)
        .where(Chunk.paper_id == paper_id, Chunk.workspace_id == workspace_id)
        .limit(1)
    )
    if has_chunks is None:
        raise HTTPException(409, "Paper chưa được ingest (chưa có chunk)")
    job = await request.app.state.arq.enqueue_job(
        "extract_card", str(paper_id), str(workspace_id), model
    )
    return JobOut(job_id=job.job_id if job else None)


@router.get("/papers/{paper_id}/card", response_model=CardOut)
async def get_latest_card(
    paper_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    workspace_id: uuid.UUID = Depends(current_workspace_id),
) -> PaperCardRecord:
    owned = await session.scalar(
        select(Chunk.id)
        .where(Chunk.paper_id == paper_id, Chunk.workspace_id == workspace_id)
        .limit(1)
    )
    card = await session.scalar(
        select(PaperCardRecord)
        .where(PaperCardRecord.paper_id == paper_id)
        .order_by(PaperCardRecord.created_at.desc())
        .limit(1)
    )
    if owned is None or card is None:
        raise HTTPException(404, "Chưa có PaperCard cho paper này")
    return card
