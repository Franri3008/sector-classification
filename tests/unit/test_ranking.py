import numpy as np
import pandas as pd

from src.pipeline.ranking import top_candidates


def _meta():
    return pd.DataFrame(
        [
            {
                "division_code": "01",
                "division_name": "Agri",
                "section_code": "A",
                "section_name": "AGRI",
            },
            {
                "division_code": "62",
                "division_name": "Software",
                "section_code": "J",
                "section_name": "ICT",
            },
            {
                "division_code": "85",
                "division_name": "Education",
                "section_code": "P",
                "section_name": "EDUCATION",
            },
        ]
    )


def test_top_candidates_respects_floor_and_n():
    sectors = np.eye(3, dtype=np.float32)
    tag = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)
    tag /= np.linalg.norm(tag)
    picks = top_candidates(tag, sectors, _meta(), top_n=5, min_similarity=0.3)
    assert len(picks) == 1
    codes = [p["division_code"] for p in picks[0]]
    assert codes == ["01"]


def test_top_candidates_returns_top_n_sorted():
    sectors = np.array([[1.0, 0, 0], [0.5, 0.5, 0], [0, 0, 1.0]], dtype=np.float32)
    sectors /= np.linalg.norm(sectors, axis=1, keepdims=True)
    tag = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    picks = top_candidates(tag, sectors, _meta(), top_n=2, min_similarity=0.0)[0]
    assert [p["division_code"] for p in picks] == ["01", "62"]
    assert picks[0]["similarity"] >= picks[1]["similarity"]
