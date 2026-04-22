from __future__ import annotations

import pandas as pd

from src.sources.base import SourceAdapter


class OpenAlexAdapter(SourceAdapter):
    name = "openalex"

    def extract_tags(self, records: pd.DataFrame) -> pd.DataFrame:
        cols = ["domain_id", "domain"]
        if "keywords" in records.columns:
            cols.append("keywords")
        sub = records[cols].dropna(subset=["domain_id", "domain"])
        sub["domain_id"] = sub["domain_id"].astype(str).str.strip()
        sub["domain"] = sub["domain"].astype(str).str.strip()
        if "keywords" in sub.columns:
            sub["description"] = (
                sub["keywords"].fillna("").astype(str).str.replace(";", ",").str.strip()
            )
            sub = sub.drop(columns=["keywords"])
        sub = sub[(sub["domain_id"] != "") & (sub["domain"] != "")].drop_duplicates("domain_id")
        sub = sub.rename(columns={"domain_id": "key", "domain": "tag_name"})
        sub = self.assign_tag_ids(sub, tag_col="tag_name")
        out_cols = ["tag_id", "tag_name", "key"]
        if "description" in sub.columns:
            out_cols.append("description")
        return sub[out_cols]
