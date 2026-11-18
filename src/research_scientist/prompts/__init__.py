"""Prompt registry: prompt là code, có version, nằm trong git.

File đặt tên `<name>.<version>.md`, ví dụ `paper_card.v1.md`. Gọi `load_prompt("paper_card", "v1")`.
Khi đổi prompt: tạo version mới, chạy eval so sánh, rồi mới đổi version mặc định.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}.{self.version}"


@lru_cache
def load_prompt(name: str, version: str) -> Prompt:
    path = _DIR / f"{name}.{version}.md"
    return Prompt(name=name, version=version, text=path.read_text(encoding="utf-8").strip())
