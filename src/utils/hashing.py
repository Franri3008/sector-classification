from __future__ import annotations

import hashlib
import re
from pathlib import Path


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def file_hash(path: Path | str, length: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


_WHITESPACE = re.compile("\\s+")


def normalize_tag(tag: str) -> str:
    return _WHITESPACE.sub(" ", tag.strip().lower())


def tag_id(tag: str) -> str:
    return stable_hash(normalize_tag(tag))
