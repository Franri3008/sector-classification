"""v49 — IPC-prefix overrides for Regpat.

Patents have an authoritative IPC hierarchy (much more reliable than NLP
for category routing). Many regpat misclassifications come from the
partner's keyword-tail pollution in description text — e.g. ransomware
patents (G06F2221/2149) carrying "finance, payments, intermediation,
brokerage, fintech" in their description and consequently routing to
NACE 65 Insurance.

When a tag (IPC code) starts with any prefix in the rule list, the final
pick must be one of the allowed NACE divisions. Within the allowlist, the
caller picks max-cosine using its own embeddings.

Order matters — list more specific prefixes (e.g. ``G06F21``) before their
parents (``G06F``); first match wins.

Fires on ~91 / 8 521 (~1 %) of Regpat tags. See REPORT.md → Iteration 28
for design rationale and verified fixes.
"""
from __future__ import annotations

# IPC-prefix → allowed NACE divisions. First-match wins.
IPC_RULES: list[tuple[str, list[str]]] = [
    # G06F security/auth/anti-malware — software, not insurance/repair
    ("G06F21",      ["62", "63", "26"]),
    ("G06F2221",    ["62", "63", "26"]),

    # G06F GUI / I/O — software + hardware, not furniture/publishing
    ("G06F3",       ["62", "26", "27"]),

    # G06F CAD — software / engineering
    ("G06F30",      ["62", "71", "26"]),

    # G06F general computing
    ("G06F",        ["62", "63", "26", "27", "73"]),

    # No rule for G06Q — business methods legitimately map to 64–66/70/73.

    # G06T image processing / G06V vision — software / electronics / film
    ("G06T",        ["62", "26", "63", "59"]),
    ("G06V",        ["62", "26", "63"]),

    # G16H — health informatics. Currently misroutes to 65/66.
    ("G16H",        ["86", "62", "63"]),

    # H04L digital data communication — telecom / IT / electronics
    ("H04L",        ["61", "62", "63", "26"]),

    # H04W wireless networks — telecom / electronics
    ("H04W",        ["61", "26", "27", "62"]),

    # H04N image / video transmission — broadcasting / film / electronics
    ("H04N",        ["60", "59", "26", "61"]),

    # H04R electroacoustic / loudspeakers — electronics / electrical
    ("H04R",        ["26", "27", "32"]),

    # H04M telephonic — telecom / electronics
    ("H04M",        ["26", "61", "27"]),

    # H04H broadcast — broadcasting / electronics
    ("H04H",        ["60", "26", "61"]),

    # A63F video games / amusement — software / gaming / electronics
    ("A63F",        ["62", "32", "26", "59"]),

    # A61L disinfection / sterilization — pharma / medical / chemicals
    ("A61L",        ["21", "32", "20"]),

    # A61F orthopedics / hygiene / prostheses — medical / paper / textile
    ("A61F13",      ["17", "32", "22"]),
    ("A61F",        ["32", "17", "22"]),

    # A61G patient transport / wheelchairs / hospital furniture
    ("A61G",        ["32", "26", "86"]),
]


def find_override(ipc_code: str) -> list[str] | None:
    """Return the allowlist for the first matching IPC prefix, or None."""
    for prefix, allow in IPC_RULES:
        if ipc_code.startswith(prefix):
            return allow
    return None
