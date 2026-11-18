"""Adapter for Google Gemini via the official `google-genai` SDK (async client).

- Structured output: `response_mime_type="application/json"` + `response_json_schema` built from
  the Pydantic model; the response text is then validated with the same Pydantic model, so the
  rest of the system gets exactly the same objects as with Claude.
- Effort -> `thinking_config.thinking_level` (Gemini 3+). Gemini 2.5 uses thinking budgets
  instead, so no thinking config is sent for those models.
- Free tier quotas are low: 429 and 5xx are retried with exponential backoff.
- Safety blocks (finish_reason SAFETY / PROHIBITED_CONTENT / ..., or prompt_feedback.block_reason)
  are mapped to LLMRefusalError so the gateway treats them like Claude refusals.
"""

import json
import time
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

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

PREFIXES = ("gemini-", "gemma-")

_THINKING_LEVEL: dict[str, str] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "xhigh": "XHIGH",
    "max": "XHIGH",
}
_REFUSAL_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "MODEL_ARMOR"}


def is_gemini_model(model: str) -> bool:
    return model.startswith(PREFIXES)


def supports_thinking_level(model: str) -> bool:
    """thinking_level exists from Gemini 3 onward; 2.x and Gemma use other controls."""
    if not model.startswith("gemini-"):
        return False
    major = model.removeprefix("gemini-").split("-", 1)[0].split(".", 1)[0]
    return major.isdigit() and int(major) >= 3


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in (429, 500, 502, 503, 504)


_retry = retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)


def to_gemini_contents(user: UserContent) -> str:
    """Gemini takes plain text here; Claude-style document blocks are flattened to tagged text."""
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


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None = None, timeout_s: float = 180.0):
        # Without api_key the SDK falls back to GOOGLE_API_KEY / GEMINI_API_KEY env vars.
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )

    def _config(
        self, model: str, system: str, effort: Effort, max_tokens: int, json_schema: dict | None
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {"system_instruction": system, "max_output_tokens": max_tokens}
        if supports_thinking_level(model):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=_THINKING_LEVEL[effort])
        if json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = json_schema
        return types.GenerateContentConfig(**kwargs)

    @_retry
    async def _generate(self, model: str, contents: str, config: types.GenerateContentConfig) -> Any:
        return await self._client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )

    @staticmethod
    def _check(response: Any, model: str) -> str:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and feedback.block_reason:
            raise LLMRefusalError(model, str(feedback.block_reason), "prompt blocked")
        candidates = response.candidates or []
        if not candidates:
            raise LLMOutputInvalidError(f"{model} returned no candidates")
        finish = candidates[0].finish_reason
        finish_name = getattr(finish, "value", str(finish)) if finish else None
        if finish_name in _REFUSAL_FINISH:
            raise LLMRefusalError(model, finish_name, "response blocked")
        if finish_name == "MAX_TOKENS":
            raise LLMTruncatedError(f"{model} hit max_output_tokens")
        return finish_name or "STOP"

    @staticmethod
    def _usage(response: Any) -> Usage:
        u = response.usage_metadata
        if u is None:
            return Usage()
        # Thinking tokens are billed as output tokens.
        return Usage(
            input_tokens=(u.prompt_token_count or 0) - (u.cached_content_token_count or 0),
            output_tokens=(u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
            cache_read_tokens=u.cached_content_token_count or 0,
        )

    def _result[R](self, output: R, response: Any, model: str, finish: str, started: float) -> LLMResult[R]:
        usage = self._usage(response)
        return LLMResult(
            output=output,
            provider=self.name,
            model=getattr(response, "model_version", None) or model,
            usage=usage,
            cost_usd=estimate_cost(
                model, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
            stop_reason=finish,
            request_id=getattr(response, "response_id", None),
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
        config = self._config(model, system, effort, max_tokens, schema.model_json_schema())
        response = await self._generate(model, to_gemini_contents(user), config)
        finish = self._check(response, model)
        try:
            parsed = schema.model_validate(json.loads(response.text or ""))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMOutputInvalidError(f"{model}: {exc}") from exc
        return self._result(parsed, response, model, finish, started)

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
        config = self._config(model, system, effort, max_tokens, None)
        response = await self._generate(model, to_gemini_contents(user), config)
        finish = self._check(response, model)
        return self._result(response.text or "", response, model, finish, started)
