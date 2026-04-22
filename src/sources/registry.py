from __future__ import annotations

from src.sources.base import SourceAdapter
from src.sources.crunchbase import CrunchbaseAdapter
from src.sources.openalex import OpenAlexAdapter
from src.sources.regpat import RegpatAdapter

ADAPTERS: dict[str, type[SourceAdapter]] = {
    "crunchbase": CrunchbaseAdapter,
    "regpat": RegpatAdapter,
    "openalex": OpenAlexAdapter,
}


def get_adapter(name: str) -> SourceAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError as e:
        raise KeyError(f"Unknown source '{name}'. Known: {sorted(ADAPTERS)}") from e


def list_sources() -> list[str]:
    return list(ADAPTERS)
