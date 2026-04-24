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


CLASSIFY_SYSTEM = (
    "You are an expert classifier mapping tags to NACE Rev. 2 sector divisions. "
    "You will receive one input tag (with distinctive keywords) and a shortlist of candidate "
    "NACE divisions.\n"
    "Your job is to identify EVERY division whose industry or activity genuinely fits the "
    "tag. A tag can match zero, one, or several sectors. Typical output is 1-3 picks; up to "
    "5 is fine when the tag legitimately spans them. Don't pad — if only one sector fits, "
    "pick only one. Tags cover a wide range — research topics, skills, technologies, patent "
    "classes, business categories — and topical/methodological matches count "
    "(e.g. 'Computer vision' → 'Computer programming' and 'Scientific research and "
    "development').\n"
    "For EACH pick, include a `confidence` score from 0.0 to 1.0 (score each pick "
    "independently — scores do NOT need to sum to 1):\n"
    "  - 1.0  canonical, unambiguous fit (this tag clearly belongs here)\n"
    "  - 0.8  strong fit, very likely correct\n"
    "  - 0.6  plausible fit but the tag is ambiguous or the match is partial\n"
    "  - 0.4  weak — strongly consider omitting this pick instead\n"
    "  - <0.4 do not emit this pick\n"
    "Also give a one-sentence `reason` for each pick.\n"
    "Return an EMPTY picks list whenever ANY of these hold: (a) the tag is gibberish, a "
    "random-looking identifier, a UUID/hash, or placeholder text (e.g. 'lorem ipsum', "
    "'asdf123', 'placeholder-xyz'); (b) the keywords indicate the tag is nonsense or cannot "
    "be described; (c) no candidate has real industry overlap with the tag; (d) every "
    "candidate you'd consider would fall below 0.4 confidence.\n"
    "Respond with JSON matching the schema provided."
)


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


ENRICH_SECTORS_SYSTEM = (
    "You enrich NACE Rev. 2 sector divisions with keywords used to match free-form tags "
    "(research topics, business categories, patent classes, skills) against sectors via "
    "semantic similarity.\n"
    "You will receive ALL divisions belonging to a single NACE section. For EACH division you "
    "return two keyword lists:\n"
    "1. `broad_keywords` (8-12): what this sector typically encompasses. These ATTRACT relevant "
    "tags — include common activities, materials, deliverables, technologies, sub-fields, "
    "canonical examples. Think 'what would a domain expert first associate with this sector'.\n"
    "2. `distinctive_keywords` (4-6): what specifically sets this division APART from the OTHER "
    "divisions shown in the same section. These must not appear (literally or as near-synonyms) "
    "in the broad/distinctive lists of sibling divisions. Pick the activities, materials, or "
    "processes that only this division does.\n"
    "Rules: single terms or short phrases (1-3 words). Lowercase. Concrete nouns/verbs — avoid "
    "vague words like 'activity', 'service', 'innovation', 'technology', 'system'. No overlap "
    "between broad and distinctive for the same division. No overlap in distinctive across "
    "siblings (that's the whole point of distinctive).\n"
    "IMPORTANT: NACE Rev. 2 predates many current industries. Include modern/contemporary "
    "examples wherever they apply — post-2010 technologies, processes, and products that a "
    "present-day company in this division would plausibly do (e.g. additive manufacturing / 3D "
    "printing in machinery or metal/plastic processing; IoT sensors in electronics; cloud "
    "computing and SaaS in IT; synthetic biology in pharma; electric vehicles in automotive). "
    "Include at least one modern example per division where one reasonably exists, alongside the "
    "canonical classical activities. Respond with JSON matching the schema."
)


def enrich_sectors_user(
    section_code: str, section_name: str, divisions: list[dict]
) -> str:
    lines = [
        f"NACE section: {section_code} — {section_name}",
        f"Divisions in this section ({len(divisions)}):",
    ]
    for d in divisions:
        lines.append(f"- {d['division_code']}: {d['division_name']}")
    lines.append("")
    lines.append(
        "For every division above, return broad + distinctive keywords. "
        "Distinctive lists must not overlap across these siblings."
    )
    return "\n".join(lines)
