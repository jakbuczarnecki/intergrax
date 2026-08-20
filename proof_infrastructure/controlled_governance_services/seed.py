# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from proof_infrastructure.controlled_governance_services.models import (
    ChangeApprovalResponseV1,
    ChangeApprovalStateV1,
    GovernanceApprovalResponseV1,
    GovernanceDecisionStateV1,
    SecurityBlockerStatusV1,
    SecurityBlockerV1,
    SecurityStatusResponseV1,
)
from proof_infrastructure.controlled_governance_services.state import GovernanceServicesStore
from proof_infrastructure.controlled_project_status_service.seed import ORION_FIXTURE_PROJECT_ID

ORION_FIXTURE_CHANGE_ID = "ORION-DEPLOY-1"
ORION_FIXTURE_SUBJECT_ID = "ORION-ARCH-1"


def seed_orion_governance_fixture(
    store: GovernanceServicesStore,
    *,
    updated_at: datetime | None = None,
) -> None:
    timestamp = updated_at or datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    store.put_security(
        SecurityStatusResponseV1(
            project_id=ORION_FIXTURE_PROJECT_ID,
            blockers=[
                SecurityBlockerV1(
                    id="SEC-BLOCK-1",
                    status=SecurityBlockerStatusV1.CLOSED,
                ),
            ],
            status="clear",
            updated_at=timestamp,
        )
    )
    store.put_change(
        ChangeApprovalResponseV1(
            change_id=ORION_FIXTURE_CHANGE_ID,
            approval_state=ChangeApprovalStateV1.APPROVED,
            approved=True,
            updated_at=timestamp,
        )
    )
    store.put_governance(
        GovernanceApprovalResponseV1(
            subject_id=ORION_FIXTURE_SUBJECT_ID,
            decision_state=GovernanceDecisionStateV1.APPROVED,
            approved=True,
            updated_at=timestamp,
        )
    )
