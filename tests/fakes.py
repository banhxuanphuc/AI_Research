"""Provider giả để test gateway/agent/workflow mà không gọi API thật."""

from collections.abc import Callable
from typing import Any

from research_scientist.providers.base import LLMRefusalError, LLMResult, Usage


class FakeProvider:
    name = "anthropic"

    def __init__(self, respond: Callable[[str, type, Any], Any], cost: float = 0.01):
        self.respond = respond
        self.cost = cost
        self.calls: list[dict[str, Any]] = []

    async def structured(self, *, model, system, user, schema, effort, max_tokens):
        self.calls.append({"model": model, "effort": effort, "schema": schema.__name__})
        out = self.respond(model, schema, user)
        if isinstance(out, Exception):
            raise out
        return LLMResult(
            output=out, provider=self.name, model=model, usage=Usage(100, 50), cost_usd=self.cost
        )

    async def text(self, *, model, system, user, effort, max_tokens):
        self.calls.append({"model": model, "effort": effort, "schema": None})
        return LLMResult(output="ok", provider=self.name, model=model, cost_usd=self.cost)


def refuse_on(model_to_refuse: str, then: Callable[[], Any]):
    def respond(model, schema, user):
        if model == model_to_refuse:
            return LLMRefusalError(model, "test", "refused in test")
        return then()

    return respond
