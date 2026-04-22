from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _atomic_replace(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)
    _atomic_replace(tmp_path, path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(tmp_path, **arrays)
        _atomic_replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, suffix=".tmp", encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True, default=str)
        tmp_path = Path(tmp.name)
    _atomic_replace(tmp_path, path)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_meta(path: Path, **fields: Any) -> None:
    fields.setdefault("created_at", datetime.now(UTC).isoformat())
    write_json(path, fields)
