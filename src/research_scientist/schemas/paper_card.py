"""PaperCard: đầu ra có cấu trúc của Paper Analyst.

Quy tắc: mọi trường quan trọng đi kèm evidence (chunk_id + quote ngắn). Trường nào không tìm
thấy trong văn bản thì để rỗng và ghi `missing_reason` - không suy diễn.

Lưu ý structured outputs: API không hỗ trợ ràng buộc số/độ dài chuỗi (min/max...). SDK tự bỏ
chúng khỏi schema gửi đi và validate phía client, nên vẫn có thể dùng Field(ge=..., le=...).
"""

from pydantic import BaseModel, Field

PAPER_CARD_SCHEMA_VERSION = "1.0"


class Evidence(BaseModel):
    chunk_id: str = Field(description="ID của chunk chứa bằng chứng, lấy đúng từ context")
    quote: str = Field(description="Trích nguyên văn ngắn (<= 300 ký tự) từ chunk đó")


class TextField(BaseModel):
    value: str | None = Field(description="Nội dung trích xuất; null nếu paper không nêu")
    evidence: list[Evidence] = Field(default_factory=list)
    missing_reason: str | None = Field(default=None, description="Lý do khi value là null")


class ListItem(BaseModel):
    value: str
    evidence: list[Evidence] = Field(default_factory=list)


class DatasetUse(BaseModel):
    name: str
    version_or_split: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class ResultEntry(BaseModel):
    metric: str
    value: str = Field(description="Giữ nguyên định dạng trong paper, vd '84.3%' hoặc '0.912'")
    setting: str | None = Field(default=None, description="Dataset/split/điều kiện của kết quả")
    evidence: list[Evidence] = Field(default_factory=list)


class PaperCard(BaseModel):
    research_problem: TextField
    hypothesis: TextField
    contributions: list[ListItem]
    method: TextField
    datasets: list[DatasetUse]
    baselines: list[ListItem]
    metrics: list[ListItem]
    results: list[ResultEntry]
    limitations: list[ListItem]
    threats_to_validity: list[ListItem]
    future_work: list[ListItem]


# Các trường "trọng yếu" dùng để tính evidence coverage (mục tiêu MVP >= 80%)
KEY_FIELDS = ("research_problem", "method", "datasets", "metrics", "results", "limitations")
