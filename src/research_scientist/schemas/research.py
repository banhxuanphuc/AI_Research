"""Schema cho Research Brief, Discovery và các đầu ra workflow."""

from pydantic import BaseModel, Field


class ResearchBriefInput(BaseModel):
    """Người dùng nhập."""

    topic: str
    goal: str | None = None
    questions: list[str] = Field(default_factory=list)
    fields_of_study: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None


class ResearchBrief(BaseModel):
    """LLM chuẩn hoá từ input; người dùng duyệt trước khi Discovery chạy."""

    scope: str = Field(description="Phạm vi nghiên cứu đã làm rõ, 2-4 câu")
    research_questions: list[str]
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    assumptions: list[str] = Field(description="Giả định đang dùng khi brief còn mơ hồ")
    search_queries: list[str] = Field(description="3-6 truy vấn tiếng Anh cho API học thuật")


class PaperCandidate(BaseModel):
    """Kết quả discovery đã chuẩn hoá, có provenance."""

    title: str
    doi: str | None = None
    s2_id: str | None = None
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None
    open_access_pdf: str | None = None
    is_preprint: bool = False
    sources: list[str] = Field(default_factory=list)  # API nào trả về record này
