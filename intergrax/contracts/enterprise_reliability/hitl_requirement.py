# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-level human approval requirement — HITL owns workflow, ERL owns the signal."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HumanApprovalRequirement:
    """Typed approval need before automatic execution may proceed — no users or UI."""

    tenant_id: str
    correlation_id: str
    contract_id: str
    requirement_ref: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.correlation_id.strip():
            raise ValueError("correlation_id required")
        if not self.contract_id.strip():
            raise ValueError("contract_id required")
        if not self.requirement_ref.strip():
            raise ValueError("requirement_ref required")


__all__ = ["HumanApprovalRequirement"]
