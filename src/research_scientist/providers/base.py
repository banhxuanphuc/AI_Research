"""Giao diện chung cho mọi LLM provider (adapter pattern)."""

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from research_scientist.providers.models import Effort

T = TypeVar("T", bound=BaseModel)

# Nội dung user có thể là chuỗi hoặc danh sách content block (text, document...)
UserContent = str | list[dict[str, Any]]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMResult[R]:
    output: R
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    stop_reason: str | None = None
    request_id: str | None = None


class LLMError(Exception):
    """Lỗi chung của tầng LLM."""


class LLMRefusalError(LLMError):
    def __init__(self, model: str, category: str | None, explanation: str | None):
        super().__init__(f"{model} refused (category={category}): {explanation}")
        self.model = model
        self.category = category


class LLMTruncatedError(LLMError):
    """Hết max_tokens trước khi xong - đầu ra không dùng được."""


class LLMOutputInvalidError(LLMError):
    """Đầu ra không parse/validate được theo schema."""


class LLMProvider(Protocol):
    name: str

    async def structured(
        self,
        *,
        model: str,
        system: str,
        user: UserContent,
        schema: type[T],
        effort: Effort,
        max_tokens: int,
    ) -> LLMResult[T]: ...

    async def text(
        self,
        *,
        model: str,
        system: str,
        user: UserContent,
        effort: Effort,
        max_tokens: int,
    ) -> LLMResult[str]: ...
