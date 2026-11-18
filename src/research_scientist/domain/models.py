"""Mô hình dữ liệu PostgreSQL (theo mục 6 tài liệu kiến trúc + provenance).

Nguyên tắc:
- Mọi dữ liệu thuộc về một workspace -> lọc ACL theo workspace_id.
- Mọi thứ do LLM sinh ra đều lưu model + prompt_version để tái lập.
- Bảng hypotheses / experiments / reports sẽ thêm ở phase sau.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --- Users & workspaces ---------------------------------------------------------------


class WorkspaceRole(enum.StrEnum):
    owner = "owner"
    editor = "editor"
    viewer = "viewer"


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(200))
    preferences: Mapped[dict] = mapped_column(JSONB, default=dict)


class Workspace(TimestampMixin, Base):
    __tablename__ = "workspaces"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[WorkspaceRole] = mapped_column(Enum(WorkspaceRole, name="workspace_role"))


# --- Research projects ----------------------------------------------------------------


class ResearchProject(TimestampMixin, Base):
    __tablename__ = "research_projects"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    topic: Mapped[str] = mapped_column(Text)
    brief: Mapped[dict | None] = mapped_column(JSONB)  # ResearchBrief đã duyệt
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))


# --- Papers ---------------------------------------------------------------------------


class AccessStatus(enum.StrEnum):
    metadata_only = "metadata_only"
    open_access = "open_access"
    user_uploaded = "user_uploaded"


class Paper(TimestampMixin, Base):
    """Bản ghi canonical. Định danh cụ thể (DOI, S2, arXiv...) nằm ở PaperIdentifier."""

    __tablename__ = "papers"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    doi: Mapped[str | None] = mapped_column(String(255), unique=True)
    title: Mapped[str] = mapped_column(Text)
    title_normalized: Mapped[str] = mapped_column(Text, index=True)
    authors: Mapped[list] = mapped_column(JSONB, default=list)
    year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(String(500))
    abstract: Mapped[str | None] = mapped_column(Text)
    is_peer_reviewed: Mapped[bool | None] = mapped_column()
    source: Mapped[str] = mapped_column(String(50))  # nguồn đầu tiên tìm thấy

    identifiers: Mapped[list["PaperIdentifier"]] = relationship(back_populates="paper")


class PaperIdentifier(Base):
    """Không đồng nhất paperId của Semantic Scholar với DOI -> lưu riêng từng scheme."""

    __tablename__ = "paper_identifiers"
    __table_args__ = (UniqueConstraint("scheme", "value"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    scheme: Mapped[str] = mapped_column(String(30))  # doi | s2 | arxiv | openalex
    value: Mapped[str] = mapped_column(String(255))

    paper: Mapped[Paper] = relationship(back_populates="identifiers")


class ProjectPaperStatus(enum.StrEnum):
    candidate = "candidate"
    included = "included"
    excluded = "excluded"


class ProjectPaper(TimestampMixin, Base):
    __tablename__ = "project_papers"
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("research_projects.id", ondelete="CASCADE"), primary_key=True
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[ProjectPaperStatus] = mapped_column(
        Enum(ProjectPaperStatus, name="project_paper_status"),
        default=ProjectPaperStatus.candidate,
    )
    reason: Mapped[str | None] = mapped_column(Text)  # lý do relevance / loại trừ


class PaperVersion(TimestampMixin, Base):
    __tablename__ = "paper_versions"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[str] = mapped_column(String(50), default="v1")
    file_uri: Mapped[str] = mapped_column(Text)  # s3://bucket/key
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    license: Mapped[str | None] = mapped_column(String(200))
    access_status: Mapped[AccessStatus] = mapped_column(Enum(AccessStatus, name="access_status"))
    parser: Mapped[str | None] = mapped_column(String(50))
    parser_version: Mapped[str | None] = mapped_column(String(50))


class PaperSection(Base):
    __tablename__ = "paper_sections"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    paper_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paper_versions.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[str] = mapped_column(Text)  # vd "3 Method > 3.2 Loss"
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class Chunk(Base):
    """Đơn vị retrieval có provenance. Vector nằm ở Qdrant (point id = chunk id)."""

    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    section_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("paper_sections.id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    chunker_version: Mapped[str] = mapped_column(String(20))


class PaperCardRecord(TimestampMixin, Base):
    __tablename__ = "paper_cards"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    schema_version: Mapped[str] = mapped_column(String(20))
    prompt_version: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    card: Mapped[dict] = mapped_column(JSONB)
    evidence_coverage: Mapped[float | None] = mapped_column(Float)


# --- Claims & evidence ----------------------------------------------------------------


class Stance(enum.StrEnum):
    supports = "supports"
    refutes = "refutes"
    partial = "partial"


class Claim(TimestampMixin, Base):
    __tablename__ = "claims"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("research_projects.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_by_node: Mapped[str] = mapped_column(String(100))


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"
    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    stance: Mapped[Stance] = mapped_column(Enum(Stance, name="stance"))
    quote: Mapped[str | None] = mapped_column(Text)


# --- Ops: workflow runs, LLM calls, evals ---------------------------------------------


class WorkflowRun(TimestampMixin, Base):
    __tablename__ = "workflow_runs"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("research_projects.id", ondelete="CASCADE"), index=True
    )
    graph: Mapped[str] = mapped_column(String(100))
    thread_id: Mapped[str] = mapped_column(String(100), unique=True)  # LangGraph thread
    status: Mapped[str] = mapped_column(String(30), default="running")
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class LLMCall(TimestampMixin, Base):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("workflow_runs.id"), index=True)
    task: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str | None] = mapped_column(String(50))
    request_id: Mapped[str | None] = mapped_column(String(100))


class EvalRun(TimestampMixin, Base):
    __tablename__ = "eval_runs"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid)
    suite: Mapped[str] = mapped_column(String(100))
    git_sha: Mapped[str | None] = mapped_column(String(40))
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    scores: Mapped[dict] = mapped_column(JSONB, default=dict)
