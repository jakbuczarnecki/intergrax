# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded-size constants for functional evidence contracts (DIAG-FUNCTIONAL-1-R1)."""

from __future__ import annotations

# Direct inline upstream refs represent immediate parent facts for one pipeline step.
# Typical stages have 1–3 parents; 8 covers multi-input operations while forcing
# high-cardinality lineage through ``PipelineArtifactLineageFact`` records.
MAX_DIRECT_UPSTREAM_EVIDENCE_REFS = 8

# Supporting refs in reconstruction are illustrative samples, not full history.
# Six closed ``PipelineEvidenceKind`` values; 16 allows a small per-kind sample
# while keeping reconstruction memory O(1) regardless of persisted cardinality.
MAX_SUPPORTING_EVIDENCE_REFS = 16

__all__ = [
    "MAX_DIRECT_UPSTREAM_EVIDENCE_REFS",
    "MAX_SUPPORTING_EVIDENCE_REFS",
]
