# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed acquisition reason codes (UCA-3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityAcquisitionReasonCode(StrEnum):
    """Stable, auditable reason codes — not free-form policy text."""

    NONE = "none"
    NO_STRATEGY = "no_strategy"
    AMBIGUOUS_STRATEGY = "ambiguous_strategy"
    STRATEGY_REJECTED = "strategy_rejected"
    STRATEGY_UNAVAILABLE = "strategy_unavailable"
    GOVERNANCE_BLOCKED = "governance_blocked"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    DOMAIN_HANDOFF_REJECTED = "domain_handoff_rejected"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"
    INVALID_REQUEST = "invalid_request"
    STRATEGY_FAILED = "strategy_failed"
    STRATEGY_METADATA_INCONSISTENT = "strategy_metadata_inconsistent"
    SELECTION_CONFLICT = "selection_conflict"


NORMATIVE_CAPABILITY_ACQUISITION_REASON_CODES: Final[
    frozenset[CapabilityAcquisitionReasonCode]
] = frozenset(CapabilityAcquisitionReasonCode)


__all__ = [
    "CapabilityAcquisitionReasonCode",
    "NORMATIVE_CAPABILITY_ACQUISITION_REASON_CODES",
]
