"""v50 — Sector embedding overrides for over-absorbing divisions.

Provides hand-tightened keywords and concept-vector negative anchors for
NACE divisions that absorb too many tags due to generic vocabulary overlap.

Division 80 (Investigation/security): was absorbing 100 OpenAlex tags
including biology, philosophy, and geography because "surveillance",
"detection", "protection", "monitoring" overlap with research vocabulary.

Division 36 (Water supply): was absorbing hydrology/ecology tags.

Usage::

    from pipeline.sector_overrides_v50 import KEYWORD_OVERRIDES, NEGATIVE_ANCHORS

    # Replace keywords before embedding
    for code, override in KEYWORD_OVERRIDES.items():
        sector_keywords[code] = override

    # After embedding, subtract negative anchors
    for code, anchors in NEGATIVE_ANCHORS.items():
        for phrase, weight in anchors:
            neg_emb = embed(phrase)
            sector_embs[code_idx] -= weight * neg_emb
        sector_embs[code_idx] /= norm(sector_embs[code_idx])
"""
from __future__ import annotations

# Hand-tightened keywords for over-absorbing divisions
KEYWORD_OVERRIDES: dict[str, dict] = {
    "80": {
        "scope": "Private investigation firms, security guard companies, bodyguard services, burglar alarm monitoring, armoured car transport.",
        "keywords": [
            "detective", "bodyguard", "guard", "bouncer", "patrol",
            "CCTV", "alarm", "armoured", "locksmith", "polygraph",
            "bail", "fugitive", "repossession", "bailiff",
            "manned-guarding", "cash-in-transit",
        ],
    },
    "36": {
        "scope": "Water collection from rivers/wells, purification for drinking, distribution via mains/pipes.",
        "keywords": [
            "waterworks", "reservoir", "aqueduct", "desalination",
            "chlorination", "fluoridation", "pipeline", "mains",
            "potable", "pumping-station", "borehole", "wellhead",
            "filtration-plant", "drinking-water", "hydrant",
        ],
    },
    "61": {
        "scope": "Wired and wireless telecommunications carriers, ISPs, satellite operators.",
        "keywords": [
            "carrier", "operator", "broadband", "fibre-optic",
            "5G", "LTE", "satellite", "spectrum", "tower",
            "VoIP", "roaming", "interconnection", "ISP",
            "bandwidth", "switching", "multiplexing",
        ],
    },
}

# Concept-vector negative anchors: (phrase, weight) to subtract from division vector.
NEGATIVE_ANCHORS: dict[str, list[tuple[str, float]]] = {
    "80": [
        ("immune response biology immunology pathogen host defense invertebrate", 0.15),
        ("coral reef marine ecology ecosystem biodiversity ocean", 0.12),
        ("atmospheric ozone climate weather ice glacier arctic", 0.12),
        ("philosophy ethics moral theory theology humanities", 0.10),
        ("earthquake seismology geology detection monitoring sensor", 0.10),
    ],
    "36": [
        ("water transport shipping maritime vessel cargo port", 0.15),
    ],
}
