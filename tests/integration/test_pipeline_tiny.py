from __future__ import annotations

import pandas as pd

from src.pipeline.runner import run_source


def _seed_openalex(raw_dir):
    openalex_dir = raw_dir / "openalex"
    openalex_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"domain_id": "W1", "domain": "Machine learning", "keywords": "neural nets; deep learning"},
            {"domain_id": "W2", "domain": "Agriculture", "keywords": "crop; soil; farming"},
            {"domain_id": "W3", "domain": "Education", "keywords": "teaching; schools; curriculum"},
        ]
    ).to_csv(openalex_dir / "index.csv", index=False)


def test_end_to_end_produces_csv(tmp_project, fake_embedder, fake_llm):
    _seed_openalex(tmp_project / "data" / "raw")
    summary = run_source("openalex", embedder=fake_embedder, llm=fake_llm)
    assert summary.output_path.exists()
    assert summary.elapsed_s >= 0
    assert summary.output_rows > 0
    df = pd.read_csv(summary.output_path)
    assert list(df.columns) == ["sector", "key"]
    assert len(df) == summary.output_rows
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


def test_regpat_seed_description_skips_llm_describe(tmp_project, fake_embedder, fake_llm):
    regpat_dir = tmp_project / "data" / "raw" / "regpat"
    regpat_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"domain_id": "A61P35", "domain": "name", "description": "antineoplastic agents for cancer treatment"},
            {"domain_id": "A61P29", "domain": "name", "description": "anti-inflammatory compounds"},
        ]
    ).to_csv(regpat_dir / "index.csv", index=False)

    summary = run_source("regpat", embedder=fake_embedder, llm=fake_llm)
    assert summary.output_path.exists()
    assert fake_llm.describe_calls == 0
