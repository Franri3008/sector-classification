"""Override-rule modules consumed by the deterministic post-classification step.

Two layers ship here:

- ``overrides_openalex_v48`` — tag-name keyword → division-allowlist rules
  for the OpenAlex source. Fixes ~250 drift errors caused by mismatched
  OpenAlex field/subfield labels (e.g. "Mathematics Education" labelled
  under "Civil Engineering" in OpenAlex's taxonomy).

- ``overrides_regpat_v49`` — IPC-prefix → division-allowlist rules for the
  Regpat source. Fixes ~91 cross-domain leakage errors caused by the
  partner's keyword-tail pollution in description text (e.g. ransomware
  patents carrying "finance, payments" → routed to NACE 65 Insurance).

Both modules expose a ``find_override(tag) -> list[str] | None`` function.
Callers should:

  1. Look up the override allowlist for the tag.
  2. If present and the current pick is NOT in the allowlist, restrict the
     pick to the allowlist and select max-cosine within it.

See ``REPORT.md`` for the design rationale and full results.
"""

from . import overrides_openalex_v48, overrides_regpat_v49

__all__ = ["overrides_openalex_v48", "overrides_regpat_v49"]
