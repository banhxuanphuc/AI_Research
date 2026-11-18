"""Workspace mặc định cho giai đoạn chưa có auth (Phase 3 sẽ thay bằng user/JWT thật)."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.domain.models import Workspace

DEFAULT_WORKSPACE_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


async def ensure_default_workspace(session: AsyncSession) -> uuid.UUID:
    existing = await session.scalar(select(Workspace.id).where(Workspace.id == DEFAULT_WORKSPACE_ID))
    if existing is None:
        session.add(Workspace(id=DEFAULT_WORKSPACE_ID, name="Default"))
        await session.commit()
    return DEFAULT_WORKSPACE_ID
