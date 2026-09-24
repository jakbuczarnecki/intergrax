# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9 — PostgreSQL durable worker obstacle capability need."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.postgresql_worker_recovery_obstacle_capability_need_repository import (
    PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository,
)
from intergrax.autonomous_work.repository import AutonomousWorkEntityConflict
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    ProfileVersion,
)
from tests.integration.autonomous_work.conftest import resolve_postgresql_config
from tests.integration.autonomous_work.test_postgresql_recovery_episode import (
    open_bundle,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _sample_need() -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=WorkerInstanceId("winst_00000000000000000000000000000002"),
        obstacle_id="obs-pg-r59",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("invoke",),
        capability_profile_ref=CapabilityProfileRef(
            profile_id="cap-profile",
            version=ProfileVersion(1),
        ),
        requested_at=datetime(2026, 9, 24, tzinfo=UTC),
        recovery_decision_id="decision-pg-r59",
    )


@pytest.fixture(scope="module")
def _require_postgresql() -> None:
    if resolve_postgresql_config() is None:
        pytest.skip("PostgreSQL DSN not configured")


def test_postgresql_need_survives_process_restart(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
    _require_postgresql: None,
) -> None:
    need = _sample_need()
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    repo = PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository(
        postgresql_autonomous_work_bundle.store,
    )
    repo.record_obstacle_capability_need(need)
    postgresql_autonomous_work_bundle.close()

    reopened = open_bundle(schema_name)
    try:
        repo_b = PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository(
            reopened.store
        )
        loaded = repo_b.get_obstacle_capability_need(
            worker_instance_id=need.worker_instance_id,
            obstacle_id=need.obstacle_id,
        )
        assert loaded == need
    finally:
        reopened.close()


def test_postgresql_need_conflict_fail_closed(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
    _require_postgresql: None,
) -> None:
    need = _sample_need()
    repo = PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository(
        postgresql_autonomous_work_bundle.store,
    )
    repo.record_obstacle_capability_need(need)
    with pytest.raises(AutonomousWorkEntityConflict):
        repo.record_obstacle_capability_need(
            replace(need, recovery_decision_id="other-decision"),
        )


def test_postgresql_multi_host_read(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
    _require_postgresql: None,
) -> None:
    need = _sample_need()
    store = postgresql_autonomous_work_bundle.store
    host_a = PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository(store)
    host_b = PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository(store)
    host_a.record_obstacle_capability_need(need)
    assert (
        host_a.get_obstacle_capability_need(
            worker_instance_id=need.worker_instance_id,
            obstacle_id=need.obstacle_id,
        )
        == host_b.get_obstacle_capability_need(
            worker_instance_id=need.worker_instance_id,
            obstacle_id=need.obstacle_id,
        )
        == need
    )
