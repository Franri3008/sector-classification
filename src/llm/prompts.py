from __future__ import annotations

_SOURCE_KIND: dict[str, str] = {
    "crunchbase": "business category",
    "regpat": "patent class",
    "openalex": "research concept",
}


def _source_kind(source: str) -> str:
    return _SOURCE_KIND.get(source, "tag")


DESCRIBE_SYSTEM = "You are a taxonomy expert. Given a single tag, produce a comma-separated list of 4-8 concrete, distinctive keywords that narrow down the tag's industry, activity, or sub-domain. AVOID generic words that could apply to many sectors (e.g. 'research', 'analysis', 'development', 'activity', 'service', 'application', 'study', 'system', 'technology', 'innovation', 'process'). PREFER specific materials, technologies, processes, deliverables, sub-fields, or canonical examples that would help a reader place the tag into one narrow industry rather than a broad one. Return the keywords in the JSON field `description` as a single comma-separated string (no numbering, no leading text)."


def describe_user(tag_name: str, source: str) -> str:
    kind = _source_kind(source)
    return f"Source: {source}\n{kind.capitalize()}: {tag_name}\n\nReturn 4-8 comma-separated distinctive keywords for this {kind}."


CLASSIFY_SYSTEM = "You are an expert classifier mapping tags to NACE Rev. 2 sector divisions. You will receive one input tag (with distinctive keywords) and a shortlist of candidate NACE divisions. Your job is to pick every division whose industry or activity genuinely overlaps with the tag. Typical output is 1-3 divisions; up to 5 is fine. Tags cover a wide range — research topics, skills, technologies, patent classes, business categories — so topical/methodological matches count too (e.g. 'Computer vision' belongs to 'Computer programming' and 'Scientific research and development'). For a real, meaningful tag, prefer to pick at least one division. IMPORTANT: return an empty list whenever ANY of these hold: (a) the tag is gibberish, a random-looking identifier, a UUID/hash, or placeholder text (e.g. 'lorem ipsum', 'asdf123', 'placeholder-xyz'); (b) the keywords indicate the tag is nonsense or cannot be described; (c) no candidate has real industry overlap with the tag. For each pick, give a one-sentence reason. Respond with JSON matching the schema provided."


def classify_user(tag_name: str, tag_description: str, candidates: list[dict]) -> str:
    lines = [
        f"Tag: {tag_name}",
        f"Keywords: {tag_description}",
        "",
        "Candidate NACE divisions (ordered by semantic similarity, best first):",
    ]
    for c in candidates:
        lines.append(
            f"- {c['division_code']}: {c['division_name']} [section {c['section_code']} — {c['section_name']}]"
        )
    lines.append("")
    lines.append("Return the division_codes that genuinely fit this tag.")
    return "\n".join(lines)
