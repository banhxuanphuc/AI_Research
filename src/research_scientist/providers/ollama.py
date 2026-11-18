"""Adapter provider #2: Ollama chạy local (profile docker "ollama").

Dùng để so sánh chất lượng/chi phí với Claude trên cùng bộ eval. Model id trong routing có
dạng "ollama:<tên model>", ví dụ "ollama:qwen2.5:3b".

Lưu ý: máy 8 GB RAM / GTX 1650 4 GB VRAM chỉ chạy tốt model ~3B tham số (quantized).
"""

import json
import time
from typing import Any

import httpx
from pydantic import ValidationError

from research_scientist.providers.base import LLMOutputInvalidError, LLMResult, T, Usage, UserContent
from research_scientist.providers.models import Effort

PREFIX = "ollama:"


def _flatten(user: UserContent) -> str:
    if isinstance(user, str):
        return user
    parts: list[str] = []
    for block in user:
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "document":
            src = block.get("source", {})
            parts.append(f"<document title={block.get('title', '')!r}>\n{src.get('data', '')}\n</document>")
    return "\n\n".join(parts)


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, timeout: float = 300.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def _chat(self, model: str, system: str, user: UserContent, fmt: Any) -> tuple[dict, int]:
        started = time.perf_counter()
        body: dict[str, Any] = {
            "model": model.removeprefix(PREFIX),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _flatten(user)},
            ],
            "stream": False,
        }
        if fmt is not None:
            body["format"] = fmt
        resp = await self._client.post("/api/chat", json=body)
        resp.raise_for_status()
        return resp.json(), int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _usage(data: dict) -> Usage:
        return Usage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def structured(
        self,
        *,
        model: str,
        system: str,
        user: UserContent,
        schema: type[T],
        effort: Effort,
        max_tokens: int,
    ) -> LLMResult[T]:
        data, latency = await self._chat(model, system, user, schema.model_json_schema())
        content = data.get("message", {}).get("content", "")
        try:
            parsed = schema.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMOutputInvalidError(f"{model}: {exc}") from exc
        return LLMResult(
            output=parsed,
            provider=self.name,
            model=model,
            usage=self._usage(data),
            latency_ms=latency,
            stop_reason=data.get("done_reason"),
        )

    async def text(
        self,
        *,
        model: str,
        system: str,
        user: UserContent,
        effort: Effort,
        max_tokens: int,
    ) -> LLMResult[str]:
        data, latency = await self._chat(model, system, user, None)
        return LLMResult(
            output=data.get("message", {}).get("content", ""),
            provider=self.name,
            model=model,
            usage=self._usage(data),
            latency_ms=latency,
            stop_reason=data.get("done_reason"),
        )
