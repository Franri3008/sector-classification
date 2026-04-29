"""Apply v48/v49 override rules to an existing per-source picks CSV.

Usage::

    python -m pipeline.apply_overrides \\
        --source openalex \\
        --seed_csv data/outputs/openalex__seed.csv \\
        --sector_emb_npz data/embeddings/sector_embeddings_v47.npz \\
        --tag_emb_npz   data/embeddings/tag_emb_openalex__v1.npz \\
        --out data/outputs/openalex__final.csv

For ``--source openalex`` the v48 keyword rules apply; for ``--source
regpat`` the v49 IPC-prefix rules apply. ``crunchbase`` has no overrides
(pass-through).

Embeddings are expected as numpy ``.npz`` files:
- sector emb: an ``"div_v47"`` (or ``"div_<variant>"``) array plus a
  matching ``"division_codes"`` array.
- tag emb: a ``"tag_id"`` array plus an ``"emb"`` array of shape
  ``(n_tags, dim)``.

The applier matches each rank-1 pick against the rule set; if a rule
fires AND the current pick is outside the allowlist, it picks the
max-cosine division within the allowlist. Picks that are already inside
the allowlist (or for which no rule fires) are passed through. Ranks 2/3
are kept for non-overridden tags and dropped for overridden ones (their
ordering becomes stale relative to the new top-1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import overrides_openalex_v48, overrides_regpat_v49


REPO_ROOT = Path(__file__).resolve().parents[1]


def _renorm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _load_sectors() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "sectors.csv", dtype={"division_code": str})
    df["division_code"] = df["division_code"].str.zfill(2)
    return df.reset_index(drop=True)


def _load_div_emb(npz_path: Path, sectors: pd.DataFrame) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    div_array_key = next((k for k in data.files if k.startswith("div_")), None)
    if div_array_key is None:
        raise KeyError(f"no 'div_*' array in {npz_path}")
    div_emb = data[div_array_key].astype(np.float32)
    div_codes = list(data["division_codes"])
    order = np.array([div_codes.index(c) for c in sectors["division_code"]])
    return div_emb[order]


def _load_tag_emb(npz_path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    tag_ids = [str(t) for t in data["tag_id"]]
    emb = data["emb"].astype(np.float32)
    return tag_ids, emb


def _find_override(source: str, tag: str) -> list[str] | None:
    if source == "openalex":
        return overrides_openalex_v48.find_override(str(tag))
    if source == "regpat":
        return overrides_regpat_v49.find_override(str(tag))
    return None


def apply_overrides(
    source: str,
    seed_csv: Path,
    sector_emb_npz: Path,
    tag_emb_npz: Path,
    out_csv: Path,
) -> int:
    """Return the number of tags whose rank-1 pick was overridden."""
    sectors = _load_sectors()
    code_to_idx = {c: i for i, c in enumerate(sectors["division_code"])}
    div_emb = _load_div_emb(sector_emb_npz, sectors)
    tag_ids, tag_emb = _load_tag_emb(tag_emb_npz)
    tag_norm = _renorm(tag_emb)
    tag_idx = {t: i for i, t in enumerate(tag_ids)}

    seed = pd.read_csv(seed_csv)
    seed["division_code"] = seed["division_code"].apply(
        lambda x: f"{int(float(x)):02d}" if pd.notna(x) else None
    )
    seed = seed.dropna(subset=["division_code"]).copy()

    new_idx_map: dict[str, int] = {}
    for _, row in seed[seed["rank"] == 1].iterrows():
        tag = str(row["tag"])
        cur_code = row["division_code"]
        cur_idx = code_to_idx.get(cur_code)
        allow_codes = _find_override(source, tag)
        if allow_codes is None:
            continue
        allow_idx = [code_to_idx[c] for c in allow_codes if c in code_to_idx]
        if not allow_idx or cur_idx in allow_idx:
            continue
        if tag not in tag_idx:
            continue
        sub_sims = tag_norm[tag_idx[tag]] @ div_emb[allow_idx].T
        new_idx_map[tag] = allow_idx[int(np.argmax(sub_sims))]

    rows = []
    for tag, group in seed.groupby("tag", sort=False):
        if tag in new_idx_map:
            d_idx = new_idx_map[tag]
            rec = sectors.iloc[d_idx]
            sc = float(tag_norm[tag_idx[tag]] @ div_emb[d_idx])
            rows.append({
                "source": source, "tag": tag,
                "division_code": rec["division_code"],
                "division_name": rec["division_name"],
                "section_code": rec["section_code"],
                "section_name": rec["section_name"],
                "score": sc, "rank": 1, "cos": sc,
            })
        else:
            for _, r in group.iterrows():
                rows.append({k: r[k] for k in
                             ["source", "tag", "division_code", "division_name",
                              "section_code", "section_name", "score", "rank", "cos"]})

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return len(new_idx_map)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["openalex", "regpat", "crunchbase"])
    ap.add_argument("--seed_csv", required=True, type=Path)
    ap.add_argument("--sector_emb_npz", required=True, type=Path)
    ap.add_argument("--tag_emb_npz", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.source == "crunchbase":
        df = pd.read_csv(args.seed_csv)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"crunchbase: pass-through ({len(df)} rows) → {args.out}")
        return

    n = apply_overrides(
        args.source, args.seed_csv, args.sector_emb_npz, args.tag_emb_npz, args.out
    )
    print(f"{args.source}: {n} tags overridden → {args.out}")


if __name__ == "__main__":
    main()
