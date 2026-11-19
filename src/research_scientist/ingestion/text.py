"""Text normalization shared by all parsers.

PDF extractors can emit NUL bytes and other control characters (font/encoding artifacts).
PostgreSQL rejects NUL (0x00) in text columns, and control characters also hurt embeddings and
quote matching, so every parser output goes through `clean_text`.
"""

import re
import unicodedata

# C0 controls except \t \n \r, plus DEL and C1 controls
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.sub("", text).strip()
