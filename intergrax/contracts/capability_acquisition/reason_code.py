# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed realization reason codes (UCA-2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityRealizationReasonCode(StrEnum):
    """Stable, auditable reason codes — not free-form policy text."""

    NONE = "none"
    NO_PROVIDER = "no_provider"
    AMBIGUOUS_PROVIDER = "ambiguous_provider"
    PROVIDER_REJECTED = "provider_rejected"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    GOVERNANCE_BLOCKED = "governance_blocked"
    DOMAIN_HANDOFF_REJECTED = "domain_handoff_rejected"
    DOMAIN_HANDOFF_DEFERRED = "domain_handoff_deferred"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"
    INVALID_REQUEST = "invalid_request"
    DOMAIN_REALIZATION_FAILED = "domain_realization_failed"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"


NORMATIVE_CAPABILITY_REALIZATION_REASON_CODES: Final[
    frozenset[CapabilityRealizationReasonCode]
] = frozenset(CapabilityRealizationReasonCode)


__all__ = [
    "CapabilityRealizationReasonCode",
    "NORMATIVE_CAPABILITY_REALIZATION_REASON_CODES",
]
