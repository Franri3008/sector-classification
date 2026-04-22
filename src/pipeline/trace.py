from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.io.local_cache import read_npz, read_parquet
from src.llm.prompts import CLASSIFY_SYSTEM, DESCRIBE_SYSTEM, classify_user, describe_user
from src.pipeline.classify import classification_cache_path
from src.pipeline.descriptions import _cache_path as desc_cache_path
from src.pipeline.embed_tags import cache_paths as tag_emb_cache_paths
from src.pipeline.sectors import _cache_paths as sector_cache_paths
from src.pipeline.sectors import load_sectors
from src.sources.registry import get_adapter
from src.utils.hashing import normalize_tag
from src.utils.hashing import tag_id as compute_tag_id


@dataclass
class TraceStep:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


@dataclass
class TagTrace:
    source: str
    tag_name: str
    tag_id: str
    normalized: str
    steps: list[TraceStep] = field(default_factory=list)


def _pick_random_tag(source: str) -> tuple[str, str]:
    path = desc_cache_path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"no description cache at {path}. Run the pipeline at least once, or pass a tag explicitly."
        )
    df = read_parquet(path)
    if df.empty:
        raise ValueError(f"description cache {path} is empty")
    row = df.sample(1).iloc[0]
    return (row["tag_id"], row["tag_name"])


def build_trace(source: str, tag: str | None = None, *, seed: int | None = None) -> TagTrace:
    if seed is not None:
        random.seed(seed)
    if tag is None:
        tag_id, tag_name = _pick_random_tag(source)
    else:
        tag_name = tag
        tag_id = compute_tag_id(tag)
    trace = TagTrace(
        source=source, tag_name=tag_name, tag_id=tag_id, normalized=normalize_tag(tag_name)
    )
    input_step = TraceStep(name="input_rows")
    try:
        adapter = get_adapter(source)
        records = adapter.load_records()
        tags_df = adapter.extract_tags(records)
        match = tags_df[tags_df["tag_id"] == tag_id]
        input_step.data = {
            "count": len(match),
            "raw_files": [str(p) for p in adapter.raw_files()],
            "rows": match[["key", "tag_name"]].head(20).to_dict(orient="records"),
            "truncated": len(match) > 20,
        }
        if len(match) == 0:
            input_step.note = (
                "tag not found in current raw inputs — trace will continue from caches"
            )
    except Exception as e:
        input_step.note = f"could not load raw inputs: {e}"
    trace.steps.append(input_step)
    desc_step = TraceStep(name="description")
    desc_path = desc_cache_path(source)
    description = None
    description_llm_model = None
    if desc_path.exists():
        ddf = read_parquet(desc_path)
        hit = ddf[ddf["tag_id"] == tag_id]
        if len(hit):
            description = str(hit.iloc[0]["description"])
            description_llm_model = str(hit.iloc[0].get("llm_model", ""))
    desc_step.data = {
        "cache_path": str(desc_path),
        "llm_model": description_llm_model,
        "system_prompt": DESCRIBE_SYSTEM,
        "user_prompt": describe_user(tag_name, source),
        "description": description,
    }
    if description is None:
        desc_step.note = "no cached description for this tag"
    trace.steps.append(desc_step)
    emb_step = TraceStep(name="embedding")
    emb_path, _ = tag_emb_cache_paths(source)
    tag_vec: np.ndarray | None = None
    if emb_path.exists():
        cached = read_npz(emb_path)
        ids = cached["tag_id"].tolist()
        if tag_id in ids:
            tag_vec = cached["emb"][ids.index(tag_id)]
    emb_step.data = {
        "cache_path": str(emb_path),
        "model": settings.embedding.model,
        "dim": int(tag_vec.shape[0]) if tag_vec is not None else None,
        "l2_norm": float(np.linalg.norm(tag_vec)) if tag_vec is not None else None,
        "embedded_text": f"{tag_name}. {description}" if description is not None else tag_name,
    }
    if tag_vec is None:
        emb_step.note = "no cached tag embedding"
    trace.steps.append(emb_step)
    ranking_step = TraceStep(name="full_ranking")
    sec_npz_path, _ = sector_cache_paths()
    ranking_df: pd.DataFrame | None = None
    if tag_vec is not None and sec_npz_path.exists():
        sec_cache = read_npz(sec_npz_path)
        sec_codes = sec_cache["division_code"].tolist()
        sec_emb = sec_cache["emb"]
        sectors = load_sectors()
        idx = [sec_codes.index(c) for c in sectors["division_code"].tolist()]
        sec_emb = sec_emb[idx]
        sims = (tag_vec @ sec_emb.T).astype(float)
        ranking_df = sectors.copy()
        ranking_df["similarity"] = sims
        ranking_df = ranking_df.sort_values("similarity", ascending=False).reset_index(drop=True)
        ranking_df["rank"] = ranking_df.index + 1
    ranking_step.data = {
        "sector_cache": str(sec_npz_path),
        "rows": ranking_df[
            ["rank", "similarity", "division_code", "division_name", "section_code", "section_name"]
        ].to_dict(orient="records")
        if ranking_df is not None
        else None,
    }
    if ranking_df is None:
        ranking_step.note = "missing tag embedding or sector embedding cache"
    trace.steps.append(ranking_step)
    cand_step = TraceStep(name="candidates")
    candidates: list[dict] = []
    if ranking_df is not None:
        shortlist = ranking_df.head(settings.ranking.top_n)
        shortlist = shortlist[shortlist["similarity"] >= settings.ranking.min_similarity]
        candidates = [
            {
                "division_code": r.division_code,
                "division_name": r.division_name,
                "section_code": r.section_code,
                "section_name": r.section_name,
                "similarity": float(r.similarity),
            }
            for r in shortlist.itertuples(index=False)
        ]
    cand_step.data = {
        "top_n": settings.ranking.top_n,
        "min_similarity": settings.ranking.min_similarity,
        "n_selected": len(candidates),
        "candidates": candidates,
    }
    trace.steps.append(cand_step)
    cls_step = TraceStep(name="classification")
    cls_path = classification_cache_path(source)
    picks: list[dict] = []
    raw_json: str | None = None
    if cls_path.exists():
        cdf = read_parquet(cls_path)
        hit = cdf[cdf["tag_id"] == tag_id]
        if len(hit):
            raw_json = str(hit.iloc[0]["raw_json"])
            for _, row in hit.iterrows():
                if pd.notna(row["division_code"]):
                    picks.append(
                        {
                            "division_code": row["division_code"],
                            "reason": row["reason"],
                            "similarity": float(row["similarity"])
                            if pd.notna(row["similarity"])
                            else None,
                        }
                    )
    cls_step.data = {
        "cache_path": str(cls_path),
        "backend": settings.llm.backend,
        "model": settings.llm.openai_model
        if settings.llm.backend == "openai"
        else settings.llm.vllm_model,
        "system_prompt": CLASSIFY_SYSTEM,
        "user_prompt": classify_user(tag_name, description or "", candidates)
        if candidates
        else "(no candidates — LLM was not called)",
        "raw_response": raw_json,
        "picks": picks,
    }
    if raw_json is None:
        cls_step.note = "no cached classification for this tag"
    trace.steps.append(cls_step)
    out_step = TraceStep(name="output_rows")
    output_rows: list[dict[str, str]] = []
    try:
        adapter = get_adapter(source)
        tags_df = adapter.extract_tags(adapter.load_records())
        keys = tags_df.loc[tags_df["tag_id"] == tag_id, "key"].unique()
        for key in keys:
            for p in picks:
                output_rows.append({"sector": str(p["division_code"]), "key": str(key)})
        output_rows = [dict(t) for t in {tuple(sorted(r.items())) for r in output_rows}]
        output_rows.sort(key=lambda r: (r["key"], r["sector"]))
    except Exception as e:
        out_step.note = f"could not reconstruct output rows: {e}"
    out_step.data = {"count": len(output_rows), "rows": output_rows}
    trace.steps.append(out_step)
    return trace


def _box(text: str, indent: str = "  ") -> str:
    lines = text.splitlines() or [""]
    top = f"{indent}┌─"
    body = "\n".join(f"{indent}│ {ln}" for ln in lines)
    bot = f"{indent}└─"
    return f"{top}\n{body}\n{bot}"


def format_trace(trace: TagTrace, *, ranking_limit: int = 20) -> str:
    out: list[str] = []
    title = f' TRACE  source={trace.source}  tag="{trace.tag_name}" '
    out.append("═" * len(title))
    out.append(title)
    out.append(f' tag_id={trace.tag_id}  normalized="{trace.normalized}" '.ljust(len(title)))
    out.append("═" * len(title))
    out.append("")
    for i, step in enumerate(trace.steps, start=1):
        out.append(f"── STEP {i}: {step.name.upper().replace('_', ' ')} ──")
        if step.note:
            out.append(f"  ⚠ {step.note}")
        if step.name == "input_rows":
            d = step.data
            out.append(f"  raw files: {len(d.get('raw_files', []))}")
            out.append(f"  rows referencing this tag: {d.get('count', 0)}")
            for r in d.get("rows", []) or []:
                out.append(f'''    {r["key"]}  "{r["tag_name"]}"''')
            if d.get("truncated"):
                out.append("    ... (truncated)")
        elif step.name == "description":
            d = step.data
            out.append(f"  model: {d.get('llm_model') or '(not cached)'}")
            out.append(f"  cache: {d['cache_path']}")
            out.append("  system prompt:")
            out.append(_box(d["system_prompt"]))
            out.append("  user prompt:")
            out.append(_box(d["user_prompt"]))
            out.append("  description:")
            out.append(_box(d.get("description") or "(none)"))
        elif step.name == "embedding":
            d = step.data
            out.append(f"  model: {d['model']}  dim: {d.get('dim')}")
            out.append(f"  L2 norm: {d.get('l2_norm')}")
            out.append(f"  cache: {d['cache_path']}")
            out.append("  embedded text:")
            out.append(_box(d["embedded_text"]))
        elif step.name == "full_ranking":
            d = step.data
            rows = d.get("rows")
            if rows is None:
                continue
            out.append(f"  showing top {min(ranking_limit, len(rows))} of {len(rows)} sectors")
            out.append(f"  {'rank':>4}  {'sim':>6}  {'code':<4}  {'division':<45}  section")
            for r in rows[:ranking_limit]:
                out.append(
                    f"  {r['rank']:>4}  {r['similarity']:>6.3f}  {r['division_code']:<4}  {r['division_name'][:45]:<45}  {r['section_code']} — {r['section_name'][:30]}"
                )
        elif step.name == "candidates":
            d = step.data
            out.append(
                f"  top_n={d['top_n']}  min_similarity={d['min_similarity']}  → {d['n_selected']} candidates"
            )
            for c in d["candidates"]:
                out.append(
                    f"    {c['similarity']:>6.3f}  {c['division_code']}  {c['division_name']}"
                )
        elif step.name == "classification":
            d = step.data
            out.append(f"  backend: {d['backend']}  model: {d['model']}")
            out.append(f"  cache: {d['cache_path']}")
            out.append("  system prompt:")
            out.append(_box(d["system_prompt"]))
            out.append("  user prompt:")
            out.append(_box(d["user_prompt"]))
            out.append("  raw response:")
            raw = d.get("raw_response") or "(none)"
            try:
                raw_pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                raw_pretty = raw
            out.append(_box(raw_pretty))
            out.append("  picks:")
            if not d["picks"]:
                out.append("    (empty)")
            for p in d["picks"]:
                sim = f" (sim={p['similarity']:.3f})" if p.get("similarity") is not None else ""
                out.append(f"    {p['division_code']}{sim}  {p['reason']}")
        elif step.name == "output_rows":
            d = step.data
            out.append(f"  {d['count']} (sector, key) row(s) contributed to the final CSV")
            for r in d["rows"][:30]:
                out.append(f"    {r['sector']}, {r['key']}")
            if d["count"] > 30:
                out.append(f"    ... and {d['count'] - 30} more")
        out.append("")
    return "\n".join(out)


def save_trace(trace: TagTrace, path: Path, *, ranking_limit: int = 88) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_trace(trace, ranking_limit=ranking_limit), encoding="utf-8")
    return path
