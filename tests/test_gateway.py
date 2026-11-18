import pytest
from pydantic import BaseModel

from research_scientist.providers.gateway import BudgetExceededError, LLMGateway, RunBudget
from research_scientist.providers.models import estimate_cost
from tests.fakes import FakeProvider, refuse_on


class Echo(BaseModel):
    value: str


async def _noop_recorder(record) -> None:
    pass


def test_estimate_cost_opus():
    # 12k input + 2k output trên Opus 5.5: 12000*4/1e6 + 2000*20/1e6 = 0.048 + 0.04
    assert estimate_cost("claude-opus-5-5", 12_000, 2_000) == pytest.approx(0.088)
    assert estimate_cost("ollama:qwen2.5:3b", 10_000, 10_000) == 0.0


async def test_gateway_uses_route_model_and_effort():
    provider = FakeProvider(lambda m, s, u: Echo(value="hi"))
    gw = LLMGateway(providers={"anthropic": provider}, recorder=_noop_recorder)
    result = await gw.structured("review", system="s", user="u", schema=Echo)
    assert result.output.value == "hi"
    assert provider.calls[0] == {"model": "claude-opus-5-5", "effort": "high", "schema": "Echo"}


async def test_gateway_falls_back_on_refusal():
    provider = FakeProvider(refuse_on("claude-opus-5-5", lambda: Echo(value="from fallback")))
    records = []

    async def recorder(rec):
        records.append(rec)

    gw = LLMGateway(providers={"anthropic": provider}, recorder=recorder)
    result = await gw.structured("paper_card", system="s", user="u", schema=Echo)
    assert result.model == "claude-opus-4-8"
    assert records[0].fallback_from == "claude-opus-5-5"


async def test_budget_stops_runaway_cost():
    provider = FakeProvider(lambda m, s, u: Echo(value="x"), cost=0.6)
    gw = LLMGateway(
        providers={"anthropic": provider}, budget=RunBudget(limit_usd=1.0), recorder=_noop_recorder
    )
    await gw.structured("review", system="s", user="u", schema=Echo)
    await gw.structured("review", system="s", user="u", schema=Echo)  # spent 1.2
    with pytest.raises(BudgetExceededError):
        await gw.structured("review", system="s", user="u", schema=Echo)


async def test_model_override_routes_to_ollama_provider():
    claude = FakeProvider(lambda m, s, u: Echo(value="claude"))
    ollama = FakeProvider(lambda m, s, u: Echo(value="ollama"))
    gw = LLMGateway(providers={"anthropic": claude, "ollama": ollama}, recorder=_noop_recorder)
    result = await gw.structured(
        "paper_card", system="s", user="u", schema=Echo, model_override="ollama:qwen2.5:3b"
    )
    assert result.output.value == "ollama"
    assert not claude.calls
