"""v48 — tag-name keyword overrides for OpenAlex.

When a tag NAME (case-insensitive) contains any keyword in ``triggers``, the
final pick must be one of ``allow``. Within the allowlist, the caller picks
max-cosine using its own embeddings.

Order matters — more specific rules listed first; first match wins.
Word-boundary aware: ``re.search(r"\\b" + trigger, name)`` so "sport"
matches "sport" / "sports" / "sporting" but NOT "transport". Allows
trailing letters so stems like "athlet" match "athletic" / "athletes".

Fires on ~5 % of OpenAlex tags. Fixes 24/27 user-flagged drift errors.
See REPORT.md → Iteration 27 for the design rationale and per-rule hit
counts.
"""
from __future__ import annotations

import re

# Keyword-trigger → allowed-NACE-division-list. First match wins.
KEYWORD_RULES: list[tuple[list[str], list[str]]] = [
    # education / pedagogy / teaching → Education
    (["pedagogy", "didactic", "curricul", "schooling", "teacher train",
      "teaching", " education ", "education and pedagogy",
      "education research", "education methods", "innovations in education",
      "education and ", "and education "],
     ["85", "72", "91"]),

    # tourism / hospitality / culinary
    (["tourism", "hospitality", "culinary"],
     ["55", "56", "79", "91"]),

    # sports / athletic / doping
    (["sport", "athlet", "doping", "olympi", "fitness training"],
     ["93", "86", "85"]),

    # space / aerospace / planetary exploration
    (["space exploration", "planetary science", "space mission",
      "spacecraft", "astronaut", "aerospace", "satellite tech"],
     ["30", "84", "26"]),

    # cancer / oncology / clinical disease (medical research)
    (["cancer", "oncology", "tumor", "tumour", "carcinom",
      "leukemia", "leukaemia", "lymphoma", "neoplasm"],
     ["86", "21", "72"]),

    # memorial / commemoration / heritage — favor cultural
    # NOTE: "post-war society" excluded — that pattern fits 84 better
    (["commemoration", "memorial studies", "heritage stud", "monument"],
     ["91", "86", "72"]),

    # finance / banking / taxation
    (["banking", "financial market", "finance", "taxation",
      " tax ", "monetary"],
     ["64", "65", "66", "84"]),

    # consulting / management / organisational
    (["organizational", "organisation performance",
      "knowledge management", "management consult",
      "human resources", "leadership stud", "head office",
      "employee performance"],
     ["70", "82", "73"]),

    # disability / accessibility / social services
    (["disability rights", "disabilities", "social inclusion",
      "social work", "elderly care", "child welfare",
      "rights and representation"],
     ["88", "86"]),

    # bioethics / medical ethics
    (["bioethic", "medical ethic", "clinical ethic"],
     ["86", "91"]),

    # human rights / civil rights
    (["human rights", "civil rights", "patient rights"],
     ["84", "88", "91"]),

    # marine biology / fishing
    (["marine ecolog", "fishery", "fisheries", "aquacultur",
      "marine biology", "ocean ecolog", "coastal ecolog"],
     ["03", "01", "02", "72"]),

    # forestry / forest research / soil
    (["forest ecology", "forest soil", "silvicult", "forest management",
      "logging", "tree biolog", "forest biolog", "boreal forest",
      "tropical forest", "forest, soil"],
     ["02", "01", "72"]),

    # plant biology / soil / ecology — agriculture sphere
    (["plant ecology", "plant biology", "soil chemistry",
      "soil microbiolog", "soil biolog", "agronomy",
      "crop research", "livestock", "plant pathology", "agricultural"],
     ["01", "02", "72"]),

    # publishing / library / archive
    (["publishing activit", "library science", "library and information"],
     ["91", "58"]),

    # AI / NLP / text analysis (most go to 62/63 software)
    (["artificial intelligence applications", "machine learning applications",
      "deep learning applications", "natural language processing",
      "text analysis"],
     ["62", "63", "26", "72"]),

    # electrical / electromagnetic / lightning — energy or electronics
    (["lightning and electromagnet", "electromagnetic phenomena",
      "electric power systems", "power grid", "high-voltage",
      "renewable energy supply"],
     ["35", "27", "26", "72"]),

    # forecasting / market research / decision support
    (["forecasting", "market research", "consumer behavior",
      "decision support"],
     ["73", "70", "82"]),

    # nanotechnology / materials manufacturing-side
    (["nanopore", "nanochannel", "semiconductor manufactur"],
     ["26", "20"]),

    # post-war / fascism / political-history society research
    (["post-war society", "post-war reconstruction", "political history",
      "fascism", "socialism", "communism and"],
     ["84", "91", "72"]),
]


def matches_keyword(tag_name: str, triggers: list[str]) -> bool:
    """Word-boundary aware substring match."""
    name = tag_name.lower()
    for t in triggers:
        if re.search(r"\b" + re.escape(t.strip()), name):
            return True
    return False


def find_override(tag_name: str) -> list[str] | None:
    """Return the allowlist for the first matching rule, or None."""
    for triggers, allow in KEYWORD_RULES:
        if matches_keyword(tag_name, triggers):
            return allow
    return None
