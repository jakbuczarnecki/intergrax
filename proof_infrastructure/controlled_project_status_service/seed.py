# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from proof_infrastructure.controlled_project_status_service.models import (
    ProjectBlockerStatusV1,
    ProjectBlockerV1,
    ProjectStatusResponseV1,
)
from proof_infrastructure.controlled_project_status_service.state import (
    ProjectStatusStore,
)

ORION_FIXTURE_PROJECT_ID = "ORION"
ORION_FIXTURE_READINESS_SCORE = 94
ORION_FIXTURE_BLOCKER_ID = "SEC-417"


def seed_orion_fixture(
    store: ProjectStatusStore,
    *,
    updated_at: datetime | None = None,
) -> ProjectStatusResponseV1:
    timestamp = updated_at or datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    status = ProjectStatusResponseV1(
        project_id=ORION_FIXTURE_PROJECT_ID,
        readiness_score=ORION_FIXTURE_READINESS_SCORE,
        blockers=[
            ProjectBlockerV1(
                id=ORION_FIXTURE_BLOCKER_ID,
                status=ProjectBlockerStatusV1.OPEN,
            ),
        ],
        status="active",
        updated_at=timestamp,
    )
    store.put_status(status)
    return status
