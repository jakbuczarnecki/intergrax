# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision / governance bridge for external operation admission (R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.external_operations.admission import OperationAdmissionDecision
from intergrax.contracts.external_operations.intent import ExternalOperationIntent


@dataclass(frozen=True, slots=True)
class ExternalOperationGovernanceContext:
    """Links LLM proposal to Decision System without direct execution."""

    decision_id: str | None
    governance_approved: bool
    tenant_id: str


@dataclass(frozen=True, slots=True)
class ExternalOperationGovernanceDecision:
    """Outcome after Decision → Governance → Admission chain."""

    intent: ExternalOperationIntent
    admission: OperationAdmissionDecision
    may_execute: bool

    def __post_init__(self) -> None:
        if self.may_execute and self.admission.verdict.value == "DENY":
            raise ValueError("may_execute cannot be true when admission is DENY")
        if self.may_execute and self.admission.verdict.value == "REQUIRES_APPROVAL":
            if self.admission.approval_id is None:
                raise ValueError(
                    "REQUIRES_APPROVAL with may_execute needs approval_id"
                )
