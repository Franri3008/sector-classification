"""v50 — Expanded IPC-prefix overrides for Regpat.

Extends v49 with additional rules derived from IPC hierarchy semantics.
New rules cover: G06K (character recognition), G06N (AI/ML), G06F40 (NLP),
G16B (bioinformatics), H04S (stereoscopic TV), A61K (pharma preparations),
A61B (medical instruments), B01D (separation/filtration), B60L/B60W (vehicles),
G09B (education devices), G10L (speech), A63F13 (video game control).

Preference boost: within an allowlist, earlier entries get a small cosine
boost (0.04 × position decay) reflecting the canonical IPC→NACE mapping.

Fires on ~2,000 / 8,528 (~23%) of Regpat tags.
"""
from __future__ import annotations

# IPC-prefix → allowed NACE divisions. First-match wins (longer prefix first).
IPC_RULES: list[tuple[str, list[str]]] = [
    # ---- Computing (G06) ----
    ("G06F21",   ["62", "63", "26"]),      # Security/auth → software
    ("G06F2221", ["62", "63", "26"]),      # Security indexing → software
    ("G06F3",    ["62", "63", "26"]),      # GUI/input → software
    ("G06F30",   ["62", "71", "26"]),      # CAD → software/engineering
    ("G06F40",   ["62", "63", "58"]),      # NLP → software/publishing
    ("G06F",     ["62", "63", "26", "27"]),# General computing
    ("G06T",     ["62", "26", "63", "59"]),# Image processing
    ("G06V",     ["62", "26", "63"]),      # Computer vision
    ("G06N",     ["62", "63", "26"]),      # AI/ML
    ("G06K",     ["62", "26", "63"]),      # Character recognition

    # ---- Health informatics (G16) ----
    ("G16H",     ["86", "62", "63"]),      # Health informatics → health/software
    ("G16B",     ["72", "62", "86"]),      # Bioinformatics → R&D/software/health

    # ---- Telecom (H04) ----
    ("H04L",     ["61", "62", "63", "26"]),# Data transmission → telecom/software
    ("H04W",     ["61", "26", "27", "62"]),# Wireless → telecom/hardware
    ("H04N",     ["60", "59", "26", "61"]),# Image/video → broadcasting
    ("H04R",     ["26", "27", "32"]),      # Electroacoustic → electronics
    ("H04M",     ["26", "61", "27"]),      # Telephonic → electronics/telecom
    ("H04H",     ["60", "26", "61"]),      # Broadcast systems
    ("H04S",     ["61", "26", "60"]),      # Stereoscopic TV

    # ---- Games (A63) ----
    ("A63F13",   ["59", "62", "58"]),      # Video game control → entertainment/software
    ("A63F",     ["59", "62", "32", "93"]),# Games/amusement

    # ---- Medical/pharma/hygiene (A61) ----
    ("A61K",     ["21", "20", "86"]),      # Pharma preparations → pharma/chemicals/health
    ("A61L",     ["20", "21", "32"]),      # Disinfection/sterilization → chemicals/pharma
    ("A61F13",   ["32", "17", "22"]),      # Diapers/sanitary → other mfg/textiles
    ("A61F",     ["32", "26", "86"]),      # Prosthetics/implants
    ("A61G",     ["32", "26", "86"]),      # Patient transport
    ("A61B",     ["32", "26", "86"]),      # Medical instruments

    # ---- Vehicles (B60) ----
    ("B60L",     ["29", "27", "35"]),      # Electric vehicles → motor vehicles
    ("B60W",     ["29", "62", "28"]),      # Vehicle control

    # ---- Separation/filtration (B01D) ----
    ("B01D",     ["20", "36", "38"]),      # Chemicals, not remediation

    # ---- Educational devices (G09B) ----
    ("G09B",     ["85", "62", "26"]),

    # ---- Audio/speech (G10L) ----
    ("G10L",     ["62", "26", "59"]),
]

PREFERENCE_BOOST: float = 0.04
"""Cosine boost for earlier items in an allowlist (decays by 0.5 per rank)."""


def find_override(ipc_code: str) -> list[str] | None:
    """Return the allowlist for the first matching IPC prefix, or None."""
    for prefix, allow in IPC_RULES:
        if ipc_code.startswith(prefix):
            return allow
    return None
