# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed qualification reason codes (UCA-4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityQualificationReasonCode(StrEnum):
    """Stable, auditable reason codes — not free-form policy text."""

    NONE = "none"
    NO_PROVIDER = "no_provider"
    SELECTION_CONFLICT = "selection_conflict"
    PROVIDER_REJECTED = "provider_rejected"
    PROVIDER_BLOCKED = "provider_blocked"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_FAILED = "provider_failed"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"
    INVALID_REQUEST = "invalid_request"
    PROVIDER_NOT_SUPPORTED = "provider_not_supported"
    PROVIDER_METADATA_INCONSISTENT = "provider_metadata_inconsistent"


NORMATIVE_CAPABILITY_QUALIFICATION_REASON_CODES: Final[
    frozenset[CapabilityQualificationReasonCode]
] = frozenset(CapabilityQualificationReasonCode)


__all__ = [
    "CapabilityQualificationReasonCode",
    "NORMATIVE_CAPABILITY_QUALIFICATION_REASON_CODES",
]
