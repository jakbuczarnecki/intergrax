# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9 — durable obstacle capability need store semantics."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.in_memory_worker_recovery_obstacle_capability_need_repository import (
    InMemoryWorkerRecoveryObstacleCapabilityNeedRepository,
)
from intergrax.autonomous_work.repository import AutonomousWorkEntityConflict
from intergrax.autonomous_work.worker_capability_need_serialization import (
    worker_capability_need_from_json,
    worker_capability_need_to_json,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    ProfileVersion,
)


def _sample_need() -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=WorkerInstanceId("winst_00000000000000000000000000000001"),
        obstacle_id="obs-r59",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("invoke",),
        capability_profile_ref=CapabilityProfileRef(
            profile_id="cap-profile",
            version=ProfileVersion(1),
        ),
        requested_at=datetime(2026, 9, 24, tzinfo=UTC),
        recovery_decision_id="decision-r59",
        recovery_episode_id="episode-r59",
    )


def test_need_codec_roundtrip() -> None:
    need = _sample_need()
    restored = worker_capability_need_from_json(worker_capability_need_to_json(need))
    assert restored == need


def test_need_conflict_fail_closed() -> None:
    need = _sample_need()
    repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    repo.record_obstacle_capability_need(need)
    mismatched = replace(need, recovery_decision_id="other")
    with pytest.raises(AutonomousWorkEntityConflict):
        repo.record_obstacle_capability_need(mismatched)


def test_need_idempotent_same_value() -> None:
    need = _sample_need()
    repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    repo.record_obstacle_capability_need(need)
    repo.record_obstacle_capability_need(need)
    assert (
        repo.get_obstacle_capability_need(
            worker_instance_id=need.worker_instance_id,
            obstacle_id=need.obstacle_id,
        )
        == need
    )
