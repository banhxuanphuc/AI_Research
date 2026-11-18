"""Paper Analyst: trích PaperCard có evidence span từ các chunk của một paper.

Quyền hạn: chỉ đọc chunk được đưa vào; không gọi tool, không suy diễn ngoài văn bản.
"""

from dataclasses import dataclass

from research_scientist.agents.evidence import EvidenceReport, verify_card
from research_scientist.prompts import load_prompt
from research_scientist.providers.base import LLMResult
from research_scientist.providers.gateway import LLMGateway
from research_scientist.schemas.paper_card import PaperCard
from research_scientist.security.untrusted import UNTRUSTED_POLICY, wrap_chunk

PROMPT = load_prompt("paper_card", "v1")


@dataclass
class ChunkInput:
    chunk_id: str
    section_path: str
    page: int | None
    text: str


@dataclass
class PaperCardOutcome:
    card: PaperCard
    evidence: EvidenceReport
    llm: LLMResult[PaperCard]
    prompt_id: str


def build_user_content(title: str, chunks: list[ChunkInput]) -> str:
    body = "\n\n".join(
        wrap_chunk(c.chunk_id, c.text, source=f"{c.section_path} | page {c.page or '?'}")
        for c in chunks
    )
    return f"Paper title: {title}\n\n{body}\n\nExtract the PaperCard for this paper."


async def extract_paper_card(
    gateway: LLMGateway,
    title: str,
    chunks: list[ChunkInput],
    model_override: str | None = None,
) -> PaperCardOutcome:
    # System prompt cố định (không chứa dữ liệu thay đổi) -> có thể cache giữa các paper
    system = f"{PROMPT.text}\n\n{UNTRUSTED_POLICY}"
    result = await gateway.structured(
        "paper_card",
        system=system,
        user=build_user_content(title, chunks),
        schema=PaperCard,
        prompt_version=PROMPT.id,
        model_override=model_override,
    )
    report = verify_card(result.output, {c.chunk_id: c.text for c in chunks})
    return PaperCardOutcome(card=result.output, evidence=report, llm=result, prompt_id=PROMPT.id)
