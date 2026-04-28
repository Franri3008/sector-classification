"""Single-source-of-truth judge for hand-curated correct classifications.

The judge file (``data/judge_classifications.csv``) lists per-source tags
whose correct NACE division has been verified by hand. Each entry records the
authoritative answer plus the rationale, so future iterations can detect
regressions and confirm fixes against a stable benchmark.

Usage from the CLI: ``python -m src.cli judge crunchbase`` runs the audit
against the latest output file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.pipeline.sectors import load_sectors

logger = get_logger(__name__)


JUDGE_COLUMNS = ["source", "tag", "correct_division_code", "rationale", "added_at"]


@dataclass(frozen=True)
class JudgeAuditRow:
    tag: str
    correct_division_code: str
    correct_division_name: str
    picked_division_code: str | None
    picked_division_name: str | None
    status: str  # "ok" | "miss" | "missing-from-output"


def judge_path() -> Path:
    return settings.paths.data_dir / "judge_classifications.csv"


def load_judge() -> pd.DataFrame:
    path = judge_path()
    if not path.exists():
        return pd.DataFrame(columns=JUDGE_COLUMNS)
    df = pd.read_csv(path, dtype={"correct_division_code": str})
    df["correct_division_code"] = df["correct_division_code"].str.zfill(2)
    return df


def latest_output_for(source: str) -> Path | None:
    out_dir = settings.paths.outputs_dir
    if not out_dir.exists():
        return None
    matches = sorted(
        out_dir.glob(f"{source}__*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _picks_top1_by_tag(picks: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return tag_name -> {"division_name": ..., "score": ...} for the top pick.

    Accepts either the postprocess output schema (``source, key, score`` where
    ``source`` is the division name and ``key`` is the tag name) or the raw
    classification schema (``tag_id, division_code, confidence, ...``).
    """
    if {"source", "key"}.issubset(picks.columns):
        df = picks.dropna(subset=["source"]).copy()
        df["score"] = pd.to_numeric(df.get("score"), errors="coerce")
        df = df.sort_values(["key", "score"], ascending=[True, False], na_position="last")
        return {
            str(r["key"]): {"division_name": str(r["source"]), "score": r["score"]}
            for _, r in df.drop_duplicates("key", keep="first").iterrows()
        }
    if "division_code" in picks.columns:
        sectors = load_sectors().set_index("division_code")["division_name"].to_dict()
        df = picks.dropna(subset=["division_code"]).copy()
        df["division_code"] = df["division_code"].astype(str).str.zfill(2)
        if "confidence" in df.columns:
            df = df.sort_values(["tag_id", "confidence"], ascending=[True, False])
        df = df.drop_duplicates("tag_id", keep="first")
        # The raw schema is keyed on tag_id; the judge file is keyed on tag name.
        # The caller is expected to reconcile.
        return {
            str(r["tag_id"]): {"division_name": sectors.get(r["division_code"]), "score": r.get("confidence")}
            for _, r in df.iterrows()
        }
    raise ValueError("unrecognised picks schema; expected (source, key) or (tag_id, division_code)")


def audit(source: str, picks: pd.DataFrame) -> pd.DataFrame:
    judge = load_judge()
    judge = judge[judge["source"] == source].reset_index(drop=True)
    if judge.empty:
        return pd.DataFrame(columns=[
            "tag", "correct_division_code", "correct_division_name",
            "picked_division_code", "picked_division_name", "status",
        ])
    code_to_name = load_sectors().set_index("division_code")["division_name"].to_dict()
    name_to_code = {v: k for k, v in code_to_name.items()}
    by_tag = _picks_top1_by_tag(picks)
    rows: list[JudgeAuditRow] = []
    for _, j in judge.iterrows():
        tag = str(j["tag"])
        correct_code = str(j["correct_division_code"]).zfill(2)
        correct_name = code_to_name.get(correct_code, "?")
        picked = by_tag.get(tag)
        if not picked or not picked.get("division_name"):
            rows.append(JudgeAuditRow(tag, correct_code, correct_name, None, None, "missing-from-output"))
            continue
        picked_name = picked["division_name"]
        picked_code = name_to_code.get(picked_name, "??")
        status = "ok" if picked_code == correct_code else "miss"
        rows.append(JudgeAuditRow(tag, correct_code, correct_name, picked_code, picked_name, status))
    return pd.DataFrame([r.__dict__ for r in rows])


def summarise_audit(audit_df: pd.DataFrame) -> str:
    if audit_df.empty:
        return "judge file is empty for this source — nothing to audit."
    counts = audit_df["status"].value_counts().to_dict()
    n_ok = counts.get("ok", 0)
    n_miss = counts.get("miss", 0)
    n_missing = counts.get("missing-from-output", 0)
    return f"{n_ok}/{len(audit_df)} ok, {n_miss} miss, {n_missing} missing-from-output"
