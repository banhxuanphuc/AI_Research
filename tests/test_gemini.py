"""GeminiProvider tested with fake SDK responses (no network, no quota)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from research_scientist.providers.base import (
    LLMOutputInvalidError,
    LLMRefusalError,
    LLMTruncatedError,
)
from research_scientist.providers.gateway import LLMGateway
from research_scientist.providers.gemini import (
    GeminiProvider,
    supports_thinking_level,
    to_gemini_contents,
)
from tests.fakes import FakeProvider


class Brief(BaseModel):
    scope: str
    queries: list[str]


def fake_response(text: str, finish: str = "STOP", block_reason=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value=finish))],
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            cached_content_token_count=20,
            candidates_token_count=30,
            thoughts_token_count=50,
        ),
        model_version="gemini-4-hydrogen",
        response_id="resp-1",
    )


@pytest.fixture
def provider(monkeypatch):
    p = GeminiProvider(api_key="test-key")
    p.sent = []

    def install(response):
        async def _generate(model, contents, config):
            p.sent.append({"model": model, "contents": contents, "config": config})
            return response

        monkeypatch.setattr(p, "_generate", _generate)

    p.install = install
    return p


def test_thinking_level_only_for_gemini_3_plus():
    assert supports_thinking_level("gemini-4-hydrogen")
    assert supports_thinking_level("gemini-3.5-flash-lite")
    assert not supports_thinking_level("gemini-2.5-flash")
    assert not supports_thinking_level("gemma-4-31b-it")


def test_document_blocks_are_flattened():
    text = to_gemini_contents(
        [
            {"type": "text", "text": "Q?"},
            {"type": "document", "title": "P", "source": {"data": "body"}},
        ]
    )
    assert text == "Q?\n\n<document title='P'>\nbody\n</document>"


async def test_structured_output_is_validated_and_usage_counted(provider):
    provider.install(fake_response('{"scope": "long-tail", "queries": ["a", "b"]}'))
    result = await provider.structured(
        model="gemini-4-hydrogen",
        system="sys",
        user="u",
        schema=Brief,
        effort="high",
        max_tokens=1000,
    )
    assert result.output == Brief(scope="long-tail", queries=["a", "b"])
    assert result.provider == "gemini"
    assert result.usage.input_tokens == 100  # 120 prompt - 20 cached
    assert result.usage.cache_read_tokens == 20
    assert result.usage.output_tokens == 80  # 30 answer + 50 thinking
    cfg = provider.sent[0]["config"]
    assert cfg.system_instruction == "sys"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema["required"] == ["scope", "queries"]
    assert cfg.thinking_config.thinking_level.value == "HIGH"


async def test_no_thinking_config_for_gemini_2_5(provider):
    provider.install(fake_response("hello"))
    await provider.text(model="gemini-2.5-flash", system="s", user="u", effort="low", max_tokens=10)
    assert provider.sent[0]["config"].thinking_config is None


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (fake_response("{}", finish="SAFETY"), LLMRefusalError),
        (fake_response("", block_reason="PROHIBITED_CONTENT"), LLMRefusalError),
        (fake_response('{"scope": "x"', finish="MAX_TOKENS"), LLMTruncatedError),
        (fake_response('{"scope": 1}'), LLMOutputInvalidError),
    ],
)
async def test_failures_map_to_gateway_errors(provider, response, error):
    provider.install(response)
    with pytest.raises(error):
        await provider.structured(
            model="gemini-4-hydrogen",
            system="s",
            user="u",
            schema=Brief,
            effort="low",
            max_tokens=10,
        )


async def _noop(record) -> None:
    pass


async def test_gateway_routes_by_model_name_and_default_model():
    gemini = FakeProvider(lambda m, s, u: Brief(scope="g", queries=[]))
    gemini.name = "gemini"
    claude = FakeProvider(lambda m, s, u: Brief(scope="c", queries=[]))
    gw = LLMGateway(
        providers={"gemini": gemini, "anthropic": claude},
        recorder=_noop,
        default_model="gemini-4-hydrogen",
    )
    result = await gw.structured("research_brief", system="s", user="u", schema=Brief)
    assert result.output.scope == "g"
    assert gemini.calls[0]["model"] == "gemini-4-hydrogen"
    # override per call still reaches Claude
    result = await gw.structured(
        "research_brief", system="s", user="u", schema=Brief, model_override="claude-opus-5-5"
    )
    assert result.output.scope == "c"
    assert LLMGateway.provider_key("gemma-4-31b-it") == "gemini"
    assert LLMGateway.provider_key("ollama:qwen2.5:3b") == "ollama"


async def test_missing_provider_gives_clear_error():
    gw = LLMGateway(providers={}, recorder=_noop, default_model="gemini-4-hydrogen")
    with pytest.raises(RuntimeError, match="Provider 'gemini'"):
        await gw.structured("research_brief", system="s", user="u", schema=Brief)
