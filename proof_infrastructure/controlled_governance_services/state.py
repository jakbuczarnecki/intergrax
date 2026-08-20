# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from proof_infrastructure.controlled_governance_services.models import (
    ChangeApprovalResponseV1,
    GovernanceApprovalResponseV1,
    SecurityStatusResponseV1,
)

ORION_FIXTURE_CHANGE_ID = "ORION-DEPLOY-1"
ORION_FIXTURE_SUBJECT_ID = "ORION-ARCH-1"


class GovernanceServicesStore:
    def __init__(self) -> None:
        self._security: dict[str, SecurityStatusResponseV1] = {}
        self._change: dict[str, ChangeApprovalResponseV1] = {}
        self._governance: dict[str, GovernanceApprovalResponseV1] = {}
        self._security_reads = 0
        self._change_reads = 0
        self._governance_reads = 0

    def put_security(self, status: SecurityStatusResponseV1) -> None:
        self._security[status.project_id] = status

    def get_security(self, project_id: str) -> SecurityStatusResponseV1 | None:
        self._security_reads += 1
        return self._security.get(project_id)

    def put_change(self, status: ChangeApprovalResponseV1) -> None:
        self._change[status.change_id] = status

    def get_change(self, change_id: str) -> ChangeApprovalResponseV1 | None:
        self._change_reads += 1
        return self._change.get(change_id)

    def put_governance(self, status: GovernanceApprovalResponseV1) -> None:
        self._governance[status.subject_id] = status

    def get_governance(self, subject_id: str) -> GovernanceApprovalResponseV1 | None:
        self._governance_reads += 1
        return self._governance.get(subject_id)

    def reset_read_counts(self) -> None:
        self._security_reads = 0
        self._change_reads = 0
        self._governance_reads = 0

    def read_counts(self) -> tuple[int, int, int]:
        return (self._security_reads, self._change_reads, self._governance_reads)
