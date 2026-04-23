from __future__ import annotations

from abc import ABC, abstractmethod

from src.llm.schemas import ClassificationResult, DescriptionResult, SectorEnrichmentResult


class LLMClient(ABC):
    model_id: str

    @abstractmethod
    def describe(self, tag_name: str, source: str) -> DescriptionResult: ...

    @abstractmethod
    def classify(
        self, tag_name: str, tag_description: str, candidates: list[dict]
    ) -> ClassificationResult: ...

    @abstractmethod
    def enrich_sectors(
        self, section_code: str, section_name: str, divisions: list[dict]
    ) -> SectorEnrichmentResult: ...
