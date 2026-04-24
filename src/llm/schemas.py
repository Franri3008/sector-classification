from __future__ import annotations

from pydantic import BaseModel, Field


class SectorPick(BaseModel):
    division_code: str = Field(description="Two-digit NACE division code (e.g. '01', '62').")
    reason: str = Field(description="One-sentence justification for this pick.")
    confidence: float = Field(
        description=(
            "Confidence in this pick, from 0.0 (uncertain / weak fit) to 1.0 "
            "(canonical, unambiguous fit). Below ~0.5 means the pick is probably "
            "not good enough — prefer an empty picks list instead."
        )
    )


class ClassificationResult(BaseModel):
    tag_name: str
    picks: list[SectorPick] = Field(
        description=(
            "Exactly zero or one element. Empty when no candidate is a genuinely "
            "good fit. Never more than one."
        )
    )


class DescriptionResult(BaseModel):
    tag_name: str
    description: str


class SectorKeywords(BaseModel):
    division_code: str = Field(description="Two-digit NACE division code of this entry.")
    broad_keywords: list[str] = Field(
        description="8-12 keywords describing what this sector broadly encompasses."
    )
    distinctive_keywords: list[str] = Field(
        description="4-6 keywords that set this sector apart from its siblings in the same section."
    )


class SectorEnrichmentResult(BaseModel):
    section_code: str
    divisions: list[SectorKeywords]
