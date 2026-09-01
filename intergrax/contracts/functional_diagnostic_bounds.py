# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded-size constants for functional diagnostic analysis (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

# Single specification must remain operator-reviewable and bounded.
MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS = 64

# Dependency fan-in per check — enough for small DAGs, not a workflow engine.
MAX_FUNCTIONAL_DIAGNOSTIC_DEPENDENCIES = 8

# Supporting refs are illustrative anchors, not full evidence history.
MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS = 8

# Per-check and analysis-level limitation caps.
MAX_FUNCTIONAL_DIAGNOSTIC_LIMITATIONS_PER_RESULT = 8
MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS = 16

# Factual claim strings are operator-safe summaries, not raw payloads.
MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH = 512

# Requirement payload identifiers (operation_id, query_id, artifact_ref) stay bounded.
MAX_FUNCTIONAL_DIAGNOSTIC_REQUIREMENT_TEXT_LENGTH = 256

# Analysis-scoped validation lookup — one entry per possible validation check at most.
MAX_FUNCTIONAL_DIAGNOSTIC_VALIDATIONS = 64

# Specification version is a positive integer identity component.
MAX_FUNCTIONAL_DIAGNOSTIC_SPECIFICATION_VERSION = 99999

__all__ = [
    "MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS",
    "MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS",
    "MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH",
    "MAX_FUNCTIONAL_DIAGNOSTIC_DEPENDENCIES",
    "MAX_FUNCTIONAL_DIAGNOSTIC_LIMITATIONS_PER_RESULT",
    "MAX_FUNCTIONAL_DIAGNOSTIC_REQUIREMENT_TEXT_LENGTH",
    "MAX_FUNCTIONAL_DIAGNOSTIC_SPECIFICATION_VERSION",
    "MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS",
    "MAX_FUNCTIONAL_DIAGNOSTIC_VALIDATIONS",
]
