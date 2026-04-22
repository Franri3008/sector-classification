from __future__ import annotations

import pandas as pd

from src.pipeline.runner import run_source


def _seed_openalex(raw_dir):
    openalex_dir = raw_dir / "openalex"
    openalex_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"work_id": "W1", "concept_display_name": "Machine learning"},
            {"work_id": "W1", "concept_display_name": "Agriculture"},
            {"work_id": "W2", "concept_display_name": "Machine learning"},
            {"work_id": "W3", "concept_display_name": "Education"},
        ]
    ).to_csv(openalex_dir / "sample.csv", index=False)


def test_end_to_end_produces_csv(tmp_project, fake_embedder, fake_llm):
    _seed_openalex(tmp_project / "data" / "raw")
    out_path = run_source("openalex", embedder=fake_embedder, llm=fake_llm)
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["sector", "key"]
    assert len(df) > 0
    assert set(df["key"]) <= {"W1", "W2", "W3"}


def test_cache_reuse_on_second_run(tmp_project, fake_embedder, fake_llm):
    _seed_openalex(tmp_project / "data" / "raw")
    run_source("openalex", embedder=fake_embedder, llm=fake_llm)
    describe_after_first = fake_llm.describe_calls
    classify_after_first = fake_llm.classify_calls
    embed_after_first = fake_embedder.calls
    run_source("openalex", embedder=fake_embedder, llm=fake_llm)
    assert fake_llm.describe_calls == describe_after_first
    assert fake_llm.classify_calls == classify_after_first
    assert fake_embedder.calls == embed_after_first


def test_deleting_sector_cache_reembeds_only_sectors(tmp_project, fake_embedder, fake_llm):
    _seed_openalex(tmp_project / "data" / "raw")
    run_source("openalex", embedder=fake_embedder, llm=fake_llm)
    sector_cache_dir = tmp_project / "data" / "processed" / "sectors"
    for f in sector_cache_dir.iterdir():
        f.unlink()
    embed_before = fake_embedder.calls
    describe_before = fake_llm.describe_calls
    classify_before = fake_llm.classify_calls
    run_source("openalex", embedder=fake_embedder, llm=fake_llm)
    assert fake_embedder.calls > embed_before
    assert fake_llm.describe_calls == describe_before
    assert fake_llm.classify_calls == classify_before
