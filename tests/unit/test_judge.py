from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline import judge as judge_mod
from src.pipeline import sectors as sectors_mod


@pytest.fixture
def fake_sectors_csv(tmp_path, monkeypatch):
    csv = tmp_path / "sectors.csv"
    pd.DataFrame(
        [
            {"division_code": "01", "division_name": "Crops, livestock and hunting",
             "section_code": "A", "section_name": "AGRI"},
            {"division_code": "03", "division_name": "Fishing and aquaculture",
             "section_code": "A", "section_name": "AGRI"},
        ]
    ).to_csv(csv, index=False)
    monkeypatch.setattr(judge_mod.settings.paths, "sectors_csv", csv)
    monkeypatch.setattr(sectors_mod.settings.paths, "sectors_csv", csv)
    return csv


@pytest.fixture
def fake_judge(tmp_path, monkeypatch):
    judge_csv = tmp_path / "judge_classifications.csv"
    pd.DataFrame(
        [
            {"source": "crunchbase", "tag": "Hunting",
             "correct_division_code": "01", "rationale": "in division name", "added_at": "2026-04-28"},
        ]
    ).to_csv(judge_csv, index=False)
    monkeypatch.setattr(judge_mod.settings.paths, "data_dir", tmp_path)
    return judge_csv


def test_load_judge_zero_pads_codes(fake_judge):
    df = judge_mod.load_judge()
    assert df.loc[0, "correct_division_code"] == "01"


def test_audit_postprocess_schema_ok(fake_sectors_csv, fake_judge):
    picks = pd.DataFrame(
        [
            {"source": "Crops, livestock and hunting", "key": "Hunting", "score": 0.9},
            {"source": "Fishing and aquaculture", "key": "Hunting", "score": 0.4},
        ]
    )
    out = judge_mod.audit("crunchbase", picks)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["status"] == "ok"
    assert row["picked_division_code"] == "01"


def test_audit_postprocess_schema_miss(fake_sectors_csv, fake_judge):
    picks = pd.DataFrame(
        [
            {"source": "Fishing and aquaculture", "key": "Hunting", "score": 0.7},
        ]
    )
    out = judge_mod.audit("crunchbase", picks)
    assert out.iloc[0]["status"] == "miss"
    assert out.iloc[0]["picked_division_code"] == "03"


def test_audit_handles_missing_tag(fake_sectors_csv, fake_judge):
    picks = pd.DataFrame(
        [{"source": "Fishing and aquaculture", "key": "Pickleball", "score": 0.7}]
    )
    out = judge_mod.audit("crunchbase", picks)
    assert out.iloc[0]["status"] == "missing-from-output"


def test_keyword_overrides_apply_to_embed_text(tmp_path, monkeypatch):
    overrides_path = tmp_path / "sector_keyword_overrides.json"
    overrides_path.write_text(
        '{"01": {"broad_add": ["hunting", "game"], "broad_drop": ["software"]}}'
    )
    monkeypatch.setattr(sectors_mod, "_OVERRIDES_PATH", overrides_path)

    sectors = pd.DataFrame(
        [
            {"division_code": "01", "division_name": "Agri",
             "section_code": "A", "section_name": "AGRI",
             "broad_keywords": "crops, livestock, software", "distinctive_keywords": ""},
            {"division_code": "62", "division_name": "Software",
             "section_code": "J", "section_name": "ICT",
             "broad_keywords": "code, programming", "distinctive_keywords": ""},
        ]
    )
    out = sectors_mod._apply_keyword_overrides(sectors)
    row01 = out[out["division_code"] == "01"].iloc[0]
    assert "hunting" in row01["broad_keywords"]
    assert "game" in row01["broad_keywords"]
    assert "software" not in row01["broad_keywords"]
    # untouched division left alone
    row62 = out[out["division_code"] == "62"].iloc[0]
    assert row62["broad_keywords"] == "code, programming"
