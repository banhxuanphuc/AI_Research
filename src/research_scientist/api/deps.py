"""FastAPI dependencies dùng chung."""

import uuid

from research_scientist.services.workspace import DEFAULT_WORKSPACE_ID


async def current_workspace_id() -> uuid.UUID:
    """Chưa có auth: mọi request thuộc workspace mặc định.

    Phase 3: đọc JWT -> user -> kiểm tra WorkspaceMember. Mọi query đều đã nhận workspace_id
    qua dependency này nên chỉ cần đổi ở một chỗ.
    """
    return DEFAULT_WORKSPACE_ID
