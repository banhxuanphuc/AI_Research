"""Bảng giá model và routing theo task.

Giá: USD / 1 triệu token (Anthropic API, cập nhật 11/2026). Kiểm tra lại trang pricing khi
dùng thật - đây là số để ước tính cost/run, không phải hoá đơn.

Routing: mọi task dùng model mặc định (LLM_DEFAULT_MODEL trong .env), chỉ khác nhau ở `effort`.
Provider được chọn theo tên model: "claude-*" -> Anthropic, "gemini-*"/"gemma-*" -> Google,
"ollama:*" -> Ollama local. Tuần 2 của roadmap sẽ benchmark các model trên cùng bộ test rồi mới
gán model riêng cho từng task (điền `model=` trong TASK_ROUTES) dựa trên quality-latency-cost.

Gemini chưa có giá trong PRICING -> cost ghi nhận là 0 (gói free). Khi bật billing, thêm giá
từ trang pricing của Google để budget mỗi run hoạt động đúng.
"""

from dataclasses import dataclass
from typing import Literal

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    cache_write_multiplier: float = 1.25  # cache write (TTL 5 phút) ~1.25x giá input


PRICING: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.10),
    # Haiku 5.5: giá cho prompt <= 100K token; cache_read ước tính 10% input
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_read=0.01),
    # Model dự phòng khi bị refusal
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00, cache_read=0.50),
    "claude-sonnet-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20),
}

# Khi model chính trả stop_reason="refusal", gateway thử lại một lần với model này.
REFUSAL_FALLBACK: dict[str, str] = {
    "claude-opus-5-5": "claude-opus-4-8",
    "claude-sonnet-5-5": "claude-sonnet-5",
}


@dataclass(frozen=True)
class Route:
    effort: Effort
    max_tokens: int = 16000
    model: str | None = None  # None -> dùng model mặc định của gateway


DEFAULT_MODEL = "claude-opus-5-5"

TASK_ROUTES: dict[str, Route] = {
    "research_brief": Route("medium"),
    "query_expansion": Route("low", max_tokens=4000),
    "relevance_screen": Route("low", max_tokens=4000),
    "paper_card": Route("medium"),
    "rag_answer": Route("medium"),
    "synthesis": Route("high"),
    "review": Route("high"),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    price = PRICING.get(model)
    if price is None:
        return 0.0  # model local (Ollama) hoặc chưa có trong bảng
    per_m = 1_000_000
    return (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.input * price.cache_write_multiplier
    ) / per_m
