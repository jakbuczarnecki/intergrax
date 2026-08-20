# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from proof_infrastructure.controlled_change_approval_service.models import (
    ChangeApprovalResponseV1,
    ChangeApprovalStateV1,
)
from proof_infrastructure.controlled_change_approval_service.state import ChangeApprovalStore

ORION_FIXTURE_CHANGE_ID = "ORION-DEPLOY-1"


def seed_orion_change_fixture(
    store: ChangeApprovalStore,
    *,
    updated_at: datetime | None = None,
) -> ChangeApprovalResponseV1:
    timestamp = updated_at or datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    status = ChangeApprovalResponseV1(
        change_id=ORION_FIXTURE_CHANGE_ID,
        approval_state=ChangeApprovalStateV1.APPROVED,
        approved=True,
        updated_at=timestamp,
    )
    store.put_change(status)
    return status
