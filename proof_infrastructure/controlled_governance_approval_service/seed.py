# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from proof_infrastructure.controlled_governance_approval_service.models import (
    GovernanceApprovalResponseV1,
    GovernanceDecisionStateV1,
)
from proof_infrastructure.controlled_governance_approval_service.state import (
    GovernanceApprovalStore,
)

ORION_FIXTURE_SUBJECT_ID = "ORION-ARCH-1"


def seed_orion_governance_fixture(
    store: GovernanceApprovalStore,
    *,
    updated_at: datetime | None = None,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
) -> GovernanceApprovalResponseV1:
    timestamp = updated_at or datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    status = GovernanceApprovalResponseV1(
        subject_id=ORION_FIXTURE_SUBJECT_ID,
        decision_state=GovernanceDecisionStateV1.APPROVED,
        approved=True,
        updated_at=timestamp,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    store.put_governance(status)
    return status
