from __future__ import annotations

from pydantic import BaseModel, Field


class SectorPick(BaseModel):
    division_code: str = Field(description="Two-digit NACE division code (e.g. '01', '62').")
    reason: str = Field(description="One-sentence justification.")


class ClassificationResult(BaseModel):
    tag_name: str
    picks: list[SectorPick]


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
