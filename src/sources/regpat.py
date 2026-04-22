from __future__ import annotations

import pandas as pd

from src.sources.base import SourceAdapter


class RegpatAdapter(SourceAdapter):
    name = "regpat"

    def extract_tags(self, records: pd.DataFrame) -> pd.DataFrame:
        sub = records[["domain_id", "description"]].dropna(subset=["domain_id"])
        sub["domain_id"] = sub["domain_id"].astype(str).str.strip()
        sub["description"] = sub["description"].fillna("").astype(str).str.strip()
        sub = sub[sub["domain_id"] != ""].drop_duplicates("domain_id")
        sub = sub.rename(columns={"domain_id": "tag_name"})
        sub["key"] = sub["tag_name"]
        return self.assign_tag_ids(sub)[["tag_id", "tag_name", "key", "description"]]
