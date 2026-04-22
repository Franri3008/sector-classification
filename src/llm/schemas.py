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
