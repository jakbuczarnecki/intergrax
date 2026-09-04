# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Retired legacy Critic L2/HITL evidence — migration-only historical proof (DS-MIG-03).

Represents proven legacy behavior before L2 retirement. Not executable verification
authority; consumed only by migration parity qualification for architectural mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LegacyCriticRetiredLayer(str, Enum):
    """Layer identifier for retired legacy Critic human escalation."""

    L2_HUMAN = "l2_human"


class LegacyCriticRetiredAction(str, Enum):
    """Action identifier for retired legacy Critic HITL escalation."""

    ESCALATE_HITL = "escalate_hitl"


class LegacyCriticEvidenceProvenance(str, Enum):
    """Explicit provenance for retired-runtime parity evidence."""

    HISTORICAL_LEGACY_EVIDENCE = "historical_legacy_evidence"


@dataclass(frozen=True, slots=True)
class LegacyCriticHumanEscalationEvidence:
    """Immutable record of qualified pre-retirement L2 human escalation behavior."""

    layer: LegacyCriticRetiredLayer
    action: LegacyCriticRetiredAction
    provenance: LegacyCriticEvidenceProvenance


def proven_retired_l2_human_escalation_evidence() -> LegacyCriticHumanEscalationEvidence:
    """Factory for the historically qualified L2→HITL mapping evidence."""
    return LegacyCriticHumanEscalationEvidence(
        layer=LegacyCriticRetiredLayer.L2_HUMAN,
        action=LegacyCriticRetiredAction.ESCALATE_HITL,
        provenance=LegacyCriticEvidenceProvenance.HISTORICAL_LEGACY_EVIDENCE,
    )
