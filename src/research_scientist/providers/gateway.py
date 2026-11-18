"""LLM Gateway: điểm duy nhất mà agent/workflow gọi model.

Trách nhiệm:
- Chọn provider + model + effort theo task (TASK_ROUTES), cho phép override khi benchmark.
- Giữ ngân sách mỗi run (token/tiền) -> chặn runaway cost.
- Refusal -> thử lại một lần bằng model dự phòng.
- Ghi lại mọi call (task, model, prompt_version, token, cost, latency) qua `recorder`.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from research_scientist.providers.base import (
    LLMProvider,
    LLMRefusalError,
    LLMResult,
    T,
    UserContent,
)
from research_scientist.providers.gemini import is_gemini_model
from research_scientist.providers.models import DEFAULT_MODEL, REFUSAL_FALLBACK, TASK_ROUTES, Route
from research_scientist.providers.ollama import PREFIX as OLLAMA_PREFIX

log = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    pass


@dataclass
class RunBudget:
    limit_usd: float
    spent_usd: float = 0.0

    def check(self) -> None:
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceededError(
                f"Run budget exhausted: spent ${self.spent_usd:.4f} / ${self.limit_usd:.2f}"
            )

    def add(self, cost: float) -> None:
        self.spent_usd += cost


@dataclass
class CallRecord:
    task: str
    prompt_version: str | None
    result: LLMResult
    fallback_from: str | None = None


Recorder = Callable[[CallRecord], Awaitable[None]]


async def _log_recorder(record: CallRecord) -> None:
    r = record.result
    log.info(
        "llm_call",
        extra={
            "extra_fields": {
                "task": record.task,
                "provider": r.provider,
                "model": r.model,
                "prompt_version": record.prompt_version,
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "cache_read_tokens": r.usage.cache_read_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "latency_ms": r.latency_ms,
                "fallback_from": record.fallback_from,
            }
        },
    )


@dataclass
class LLMGateway:
    providers: dict[str, LLMProvider]
    routes: dict[str, Route] = field(default_factory=lambda: dict(TASK_ROUTES))
    budget: RunBudget | None = None
    recorder: Recorder = _log_recorder
    default_model: str = DEFAULT_MODEL

    @staticmethod
    def provider_key(model: str) -> str:
        if model.startswith(OLLAMA_PREFIX):
            return "ollama"
        if is_gemini_model(model):
            return "gemini"
        return "anthropic"

    def _provider_for(self, model: str) -> LLMProvider:
        key = self.provider_key(model)
        try:
            return self.providers[key]
        except KeyError as exc:
            raise RuntimeError(f"Provider '{key}' chưa được cấu hình cho model {model}") from exc

    def route(self, task: str, model_override: str | None = None) -> Route:
        base = self.routes.get(task)
        if base is None:
            raise KeyError(f"Không có route cho task '{task}'")
        model = model_override or base.model or self.default_model
        return Route(base.effort, base.max_tokens, model)

    async def structured(
        self,
        task: str,
        *,
        system: str,
        user: UserContent,
        schema: type[T],
        prompt_version: str | None = None,
        model_override: str | None = None,
    ) -> LLMResult[T]:
        route = self.route(task, model_override)

        async def call(model: str) -> LLMResult[T]:
            return await self._provider_for(model).structured(
                model=model,
                system=system,
                user=user,
                schema=schema,
                effort=route.effort,
                max_tokens=route.max_tokens,
            )

        return await self._run(task, prompt_version, route.model, call)

    async def text(
        self,
        task: str,
        *,
        system: str,
        user: UserContent,
        prompt_version: str | None = None,
        model_override: str | None = None,
    ) -> LLMResult[str]:
        route = self.route(task, model_override)

        async def call(model: str) -> LLMResult[str]:
            return await self._provider_for(model).text(
                model=model,
                system=system,
                user=user,
                effort=route.effort,
                max_tokens=route.max_tokens,
            )

        return await self._run(task, prompt_version, route.model, call)

    async def _run[R](
        self,
        task: str,
        prompt_version: str | None,
        model: str,
        call: Callable[[str], Awaitable[LLMResult[R]]],
    ) -> LLMResult[R]:
        if self.budget:
            self.budget.check()
        fallback_from: str | None = None
        try:
            result = await call(model)
        except LLMRefusalError as exc:
            fallback = REFUSAL_FALLBACK.get(model)
            if fallback is None:
                raise
            log.warning("refusal, retrying on fallback model", extra={"extra_fields": {
                "task": task, "model": model, "fallback": fallback, "category": exc.category}})
            fallback_from = model
            result = await call(fallback)
        if self.budget:
            self.budget.add(result.cost_usd)
        await self.recorder(CallRecord(task, prompt_version, result, fallback_from))
        return result


def build_gateway(budget_usd: float | None = None) -> LLMGateway:
    """Tạo gateway từ Settings."""
    from research_scientist.core.config import get_settings
    from research_scientist.providers.claude import ClaudeProvider
    from research_scientist.providers.gemini import GeminiProvider
    from research_scientist.providers.ollama import OllamaProvider

    s = get_settings()
    providers: dict[str, LLMProvider] = {"ollama": OllamaProvider(s.ollama_url)}
    # Chỉ tạo provider khi có key, để lỗi cấu hình hiện rõ ("Provider ... chưa được cấu hình")
    if s.anthropic_key():
        providers["anthropic"] = ClaudeProvider(api_key=s.anthropic_key())
    if s.gemini_key():
        providers["gemini"] = GeminiProvider(api_key=s.gemini_key())
    budget = RunBudget(budget_usd) if budget_usd is not None else None
    return LLMGateway(providers=providers, budget=budget, default_model=s.llm_default_model)
