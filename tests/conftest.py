from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.embeddings.base import EmbeddingProvider
from src.llm.base import LLMClient
from src.llm.schemas import (
    ClassificationResult,
    DescriptionResult,
    SectorEnrichmentResult,
    SectorKeywords,
    SectorPick,
)


class FakeEmbedder(EmbeddingProvider):
    model_id = "fake-embed"
    dim = 32

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            arr = arr / 127.5 - 1.0
            vecs.append(arr[: self.dim])
        return np.asarray(vecs, dtype=np.float32)


class FakeLLM(LLMClient):
    model_id = "fake-llm"

    def __init__(self) -> None:
        self.describe_calls = 0
        self.classify_calls = 0
        self.enrich_calls = 0

    def describe(self, tag_name: str, source: str) -> DescriptionResult:
        self.describe_calls += 1
        return DescriptionResult(
            tag_name=tag_name, description=f"A synthetic description of {tag_name} from {source}."
        )

    def classify(
        self, tag_name: str, tag_description: str, candidates: list[dict]
    ) -> ClassificationResult:
        self.classify_calls += 1
        if not candidates:
            return ClassificationResult(tag_name=tag_name, picks=[])
        picks = [
            SectorPick(
                division_code=c["division_code"],
                reason=f"matches {tag_name}",
                confidence=0.9 - 0.1 * i,
            )
            for i, c in enumerate(candidates[:2])
        ]
        return ClassificationResult(tag_name=tag_name, picks=picks)

    def enrich_sectors(
        self, section_code: str, section_name: str, divisions: list[dict]
    ) -> SectorEnrichmentResult:
        self.enrich_calls += 1
        return SectorEnrichmentResult(
            section_code=section_code,
            divisions=[
                SectorKeywords(
                    division_code=d["division_code"],
                    broad_keywords=[f"broad-{d['division_code']}-a", f"broad-{d['division_code']}-b"],
                    distinctive_keywords=[f"distinct-{d['division_code']}"],
                )
                for d in divisions
            ],
        )


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    outputs = tmp_path / "data" / "outputs"
    sectors_csv = tmp_path / "sectors.csv"
    for p in (raw, processed, outputs):
        p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "division_code": "01",
                "division_name": "Crop and animal production",
                "section_code": "A",
                "section_name": "AGRICULTURE",
            },
            {
                "division_code": "10",
                "division_name": "Manufacture of food products",
                "section_code": "C",
                "section_name": "MANUFACTURING",
            },
            {
                "division_code": "62",
                "division_name": "Computer programming",
                "section_code": "J",
                "section_name": "INFORMATION AND COMMUNICATION",
            },
            {
                "division_code": "72",
                "division_name": "Scientific research and development",
                "section_code": "M",
                "section_name": "PROFESSIONAL ACTIVITIES",
            },
            {
                "division_code": "85",
                "division_name": "Education",
                "section_code": "P",
                "section_name": "EDUCATION",
            },
        ]
    ).to_csv(sectors_csv, index=False)
    monkeypatch.setattr(settings.paths, "root", tmp_path)
    monkeypatch.setattr(settings.paths, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings.paths, "raw_dir", raw)
    monkeypatch.setattr(settings.paths, "processed_dir", processed)
    monkeypatch.setattr(settings.paths, "outputs_dir", outputs)
    monkeypatch.setattr(settings.paths, "sectors_csv", sectors_csv)
    return tmp_path


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
