# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from proof_infrastructure.controlled_project_status_service.seed import ORION_FIXTURE_PROJECT_ID
from proof_infrastructure.controlled_security_status_service.models import (
    SecurityBlockerStatusV1,
    SecurityBlockerV1,
    SecurityStatusResponseV1,
)
from proof_infrastructure.controlled_security_status_service.state import SecurityStatusStore


def seed_orion_security_fixture(
    store: SecurityStatusStore,
    *,
    updated_at: datetime | None = None,
) -> SecurityStatusResponseV1:
    timestamp = updated_at or datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    status = SecurityStatusResponseV1(
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
    store.put_security(status)
    return status
