"""Brief Planner: chuẩn hoá chủ đề thô thành ResearchBrief để người dùng duyệt."""

from research_scientist.prompts import load_prompt
from research_scientist.providers.base import LLMResult
from research_scientist.providers.gateway import LLMGateway
from research_scientist.schemas.research import ResearchBrief, ResearchBriefInput

PROMPT = load_prompt("research_brief", "v1")


def _render(inp: ResearchBriefInput) -> str:
    lines = [f"Topic: {inp.topic}"]
    if inp.goal:
        lines.append(f"Goal: {inp.goal}")
    if inp.questions:
        lines.append("Questions:\n" + "\n".join(f"- {q}" for q in inp.questions))
    if inp.fields_of_study:
        lines.append(f"Fields: {', '.join(inp.fields_of_study)}")
    if inp.year_from or inp.year_to:
        lines.append(f"Years: {inp.year_from or '...'} - {inp.year_to or '...'}")
    return "\n".join(lines)


async def plan_brief(gateway: LLMGateway, inp: ResearchBriefInput) -> LLMResult[ResearchBrief]:
    return await gateway.structured(
        "research_brief",
        system=PROMPT.text,
        user=_render(inp),
        schema=ResearchBrief,
        prompt_version=PROMPT.id,
    )
