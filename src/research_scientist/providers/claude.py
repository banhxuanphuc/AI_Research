"""Adapter cho Claude qua Anthropic Python SDK (AsyncAnthropic).

- Structured output: `messages.parse(output_format=PydanticModel)` -> `parsed_output`.
- Effort: `output_config={"effort": ...}` (SDK tự gộp với format khi dùng parse).
- Thinking: Claude Opus 5.5 luôn bật adaptive thinking, nên không gửi tham số `thinking`.
- Không gửi temperature/top_p: các model mới đã bỏ các tham số sampling này.
- Prompt caching: system prompt gắn cache_control; prefix ngắn hơn ngưỡng thì đơn giản là
  không được cache (vô hại).
"""

import time
from typing import Any

import anthropic

from research_scientist.providers.base import (
    LLMOutputInvalidError,
    LLMRefusalError,
    LLMResult,
    LLMTruncatedError,
    T,
    Usage,
    UserContent,
)
from research_scientist.providers.models import Effort, estimate_cost


class ClaudeProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, max_retries: int = 2, timeout: float = 300.0):
        kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key  # nếu không truyền, SDK tự đọc ANTHROPIC_API_KEY
        self._client = anthropic.AsyncAnthropic(**kwargs)

    @staticmethod
    def _system_blocks(system: str) -> list[dict[str, Any]]:
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    @staticmethod
    def _check_stop(response: Any, model: str) -> None:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(
                model,
                getattr(details, "category", None),
                getattr(details, "explanation", None),
            )
        if response.stop_reason == "max_tokens":
            raise LLMTruncatedError(f"{model} hit max_tokens")

    @staticmethod
    def _usage(response: Any) -> Usage:
        u = response.usage
        return Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
        )

    def _result[R](self, output: R, response: Any, model: str, started: float) -> LLMResult[R]:
        usage = self._usage(response)
        return LLMResult(
            output=output,
            provider=self.name,
            model=response.model or model,
            usage=usage,
            cost_usd=estimate_cost(
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
            stop_reason=response.stop_reason,
            request_id=getattr(response, "_request_id", None),
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
        started = time.perf_counter()
        response = await self._client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},
        )
        self._check_stop(response, model)
        parsed = response.parsed_output
        if parsed is None:
            raise LLMOutputInvalidError(
                f"{model} returned no parsable output for {schema.__name__}"
            )
        return self._result(parsed, response, model, started)

    async def text(
        self,
        *,
        model: str,
        system: str,
        user: UserContent,
        effort: Effort,
        max_tokens: int,
    ) -> LLMResult[str]:
        started = time.perf_counter()
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
        )
        self._check_stop(response, model)
        text = "".join(b.text for b in response.content if b.type == "text")
        return self._result(text, response, model, started)
