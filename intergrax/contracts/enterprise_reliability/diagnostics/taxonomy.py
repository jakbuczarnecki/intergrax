# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""
Domain-neutral signal taxonomy and automation safety hints for ERL diagnostics.

Versioning (additive evolution):
- ``SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1`` pins the enum member set for v1.
- New signal kinds may be added in a minor observation schema release; removings or renames
  require a major contract version bump and explicit migration.
- ``AutomationSafetyHint`` follows the same additive-only rule under
  ``SCHEMA_AUTOMATION_SAFETY_HINT_V1``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1: Final = (
    "external_effect_reliability_signal_kind.v1"
)
SCHEMA_AUTOMATION_SAFETY_HINT_V1: Final = "automation_safety_hint.v1"


class ExternalEffectReliabilitySignalKind(StrEnum):
    """Material ERL lifecycle facts suitable for operator diagnostics — no domain semantics."""

    UNCERTAINTY_ADMITTED = "UNCERTAINTY_ADMITTED"
    RECONCILIATION_ATTEMPTED = "RECONCILIATION_ATTEMPTED"
    TRUTH_ESTABLISHED = "TRUTH_ESTABLISHED"
    TRUTH_UNAVAILABLE = "TRUTH_UNAVAILABLE"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    EVIDENCE_SUFFICIENT = "EVIDENCE_SUFFICIENT"
    RESOLUTION_POSTURE = "RESOLUTION_POSTURE"
    GOVERNANCE_POSTURE = "GOVERNANCE_POSTURE"
    RECOVERY_POSTURE = "RECOVERY_POSTURE"
    AUTOMATION_SAFETY_LIMIT = "AUTOMATION_SAFETY_LIMIT"


class AutomationSafetyHint(StrEnum):
    """
    Factual automation-safety hint derived from ERL state — not governance or execution authority.
    """

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


__all__ = [
    "AutomationSafetyHint",
    "ExternalEffectReliabilitySignalKind",
    "SCHEMA_AUTOMATION_SAFETY_HINT_V1",
    "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1",
]
