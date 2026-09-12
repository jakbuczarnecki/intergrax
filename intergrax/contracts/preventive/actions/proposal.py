# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action proposal — WHAT SHOULD HAPPEN, never execution (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


def mint_preventive_action_proposal_id() -> str:
    return f"pract_prop_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class PreventiveActionProposal:
    """
    Governed preventive action intent.

    Invariant: proposals never execute — no ``execute()`` or remediation authority.
    """

    proposal_id: str
    tenant_id: str
    risk_signal_refs: tuple[str, ...]
    recommendation_refs: tuple[str, ...]
    action_type: str
    target_resource: str
    justification: str
    confidence: float
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.proposal_id.startswith("pract_prop_"):
            raise ValueError("proposal_id must be pract_prop_*")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.risk_signal_refs:
            raise ValueError("risk_signal_refs must be non-empty")
        if not self.recommendation_refs:
            raise ValueError("recommendation_refs must be non-empty")
        if not self.action_type.strip():
            raise ValueError("action_type required")
        if not self.target_resource.strip():
            raise ValueError("target_resource required")
        if not self.justification.strip():
            raise ValueError("justification required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")


__all__ = ["PreventiveActionProposal", "mint_preventive_action_proposal_id"]
