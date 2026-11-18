"""Chính sách quyền cho tool/agent (least privilege)."""

from urllib.parse import urlparse

# Domain được phép gọi từ Discovery/ingestion. Thêm domain mới phải qua review.
ALLOWED_HOSTS = frozenset(
    {
        "api.semanticscholar.org",
        "export.arxiv.org",
        "arxiv.org",
        "api.crossref.org",
        "api.openalex.org",
    }
)


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "") in ALLOWED_HOSTS
