# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""JSON codec for durable worker recovery episode records (AW-6B)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Callable

from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WakeUpId,
    WorkerGoalId,
    WorkerInstanceId,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryStrategy,
    WorkerObstacleSourceKind,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    RecoveryEpisodeStatus,
    WorkerOriginalWorkSource,
    WorkerRecoveryEpisode,
    WorkerRecoveryResumeTarget,
    WorkerRecoveryResumeTargetKind,
)
from intergrax.contracts.autonomous_work.references import ExternalDependencyReference
from intergrax.contracts.autonomous_work.revision import Revision
from intergrax.contracts.execution_identity import ExecutionId

RECOVERY_EPISODE_CODEC_VERSION = 1


def worker_recovery_episode_to_json(episode: WorkerRecoveryEpisode) -> str:
    return json.dumps(
        worker_recovery_episode_to_payload(episode),
        sort_keys=True,
        separators=(",", ":"),
    )


def worker_recovery_episode_from_json(payload: str) -> WorkerRecoveryEpisode:
    return worker_recovery_episode_from_payload(json.loads(payload))


def worker_recovery_episode_to_payload(episode: WorkerRecoveryEpisode) -> dict[str, object]:
    return {
        "codec_version": RECOVERY_EPISODE_CODEC_VERSION,
        "recovery_episode_id": episode.recovery_episode_id,
        "worker_instance_id": episode.worker_instance_id,
        "obstacle_id": episode.obstacle_id,
        "recovery_decision_id": episode.recovery_decision_id,
        "decision_policy_version": episode.decision_policy_version,
        "strategy": episode.strategy.value,
        "original_source": _encode_original_source(episode.original_source),
        "resume_target": _encode_resume_target(episode.resume_target),
        "started_at": episode.started_at.isoformat(),
        "status": episode.status.value,
        "attempt_count": episode.attempt_count,
        "revision": episode.revision.value,
        "max_attempts": episode.max_attempts,
        "last_attempt_at": _optional_iso(episode.last_attempt_at),
        "next_retry_at": _optional_iso(episode.next_retry_at),
        "last_execution_id": episode.last_execution_id,
        "last_failure_ref": episode.last_failure_ref,
        "terminal_reason": episode.terminal_reason,
        "completed_at": _optional_iso(episode.completed_at),
        "pre_recovery_lifecycle_state": (
            episode.pre_recovery_lifecycle_state.value
            if episode.pre_recovery_lifecycle_state is not None
            else None
        ),
        "dependency_ref": episode.dependency_ref,
        "human_decision_ref": episode.human_decision_ref,
        "claimed_attempt_number": episode.claimed_attempt_number,
    }


def worker_recovery_episode_from_payload(payload: dict[str, object]) -> WorkerRecoveryEpisode:
    pre_recovery = payload.get("pre_recovery_lifecycle_state")
    return WorkerRecoveryEpisode(
        recovery_episode_id=str(payload["recovery_episode_id"]),
        worker_instance_id=WorkerInstanceId(str(payload["worker_instance_id"])),
        obstacle_id=str(payload["obstacle_id"]),
        recovery_decision_id=str(payload["recovery_decision_id"]),
        decision_policy_version=str(payload["decision_policy_version"]),
        strategy=RecoveryStrategy(str(payload["strategy"])),
        original_source=_decode_original_source(payload["original_source"]),
        resume_target=_decode_resume_target(payload["resume_target"]),
        started_at=datetime.fromisoformat(str(payload["started_at"])),
        status=RecoveryEpisodeStatus(str(payload["status"])),
        attempt_count=int(payload["attempt_count"]),
        revision=Revision(int(payload["revision"])),
        max_attempts=_optional_int(payload.get("max_attempts")),
        last_attempt_at=_optional_datetime(payload.get("last_attempt_at")),
        next_retry_at=_optional_datetime(payload.get("next_retry_at")),
        last_execution_id=_optional_execution_id(payload.get("last_execution_id")),
        last_failure_ref=_optional_str(payload.get("last_failure_ref")),
        terminal_reason=_optional_str(payload.get("terminal_reason")),
        completed_at=_optional_datetime(payload.get("completed_at")),
        pre_recovery_lifecycle_state=(
            WorkerLifecycleState(str(pre_recovery)) if pre_recovery is not None else None
        ),
        dependency_ref=_optional_dependency_ref(payload.get("dependency_ref")),
        human_decision_ref=_optional_str(payload.get("human_decision_ref")),
        claimed_attempt_number=_optional_int(payload.get("claimed_attempt_number")),
    )


def _encode_original_source(source: WorkerOriginalWorkSource) -> dict[str, str]:
    return {
        "worker_instance_id": source.worker_instance_id,
        "source_kind": source.source_kind.value,
        "source_ref": source.source_ref,
    }


def _decode_original_source(payload: object) -> WorkerOriginalWorkSource:
    if not isinstance(payload, dict):
        raise TypeError("original_source payload must be dict")
    return WorkerOriginalWorkSource(
        worker_instance_id=WorkerInstanceId(str(payload["worker_instance_id"])),
        source_kind=WorkerObstacleSourceKind(str(payload["source_kind"])),
        source_ref=str(payload["source_ref"]),
    )


def _encode_resume_target(target: WorkerRecoveryResumeTarget) -> dict[str, object]:
    return {
        "kind": target.kind.value,
        "source_ref": target.source_ref,
        "goal_id": target.goal_id,
        "goal_revision": target.goal_revision.value if target.goal_revision else None,
        "responsibility_id": target.responsibility_id,
        "wake_up_id": target.wake_up_id,
        "collaborative_work_ref": target.collaborative_work_ref,
        "execution_id": target.execution_id,
        "run_id": target.run_id,
        "requested_scopes": list(target.requested_scopes),
    }


def _decode_resume_target(payload: object) -> WorkerRecoveryResumeTarget:
    if not isinstance(payload, dict):
        raise TypeError("resume_target payload must be dict")
    goal_revision = payload.get("goal_revision")
    return WorkerRecoveryResumeTarget(
        kind=WorkerRecoveryResumeTargetKind(str(payload["kind"])),
        source_ref=str(payload["source_ref"]),
        goal_id=_optional_goal_id(payload.get("goal_id")),
        goal_revision=Revision(int(goal_revision)) if goal_revision is not None else None,
        responsibility_id=_optional_responsibility_id(payload.get("responsibility_id")),
        wake_up_id=_optional_wake_up_id(payload.get("wake_up_id")),
        collaborative_work_ref=payload.get("collaborative_work_ref"),
        execution_id=_optional_execution_id(payload.get("execution_id")),
        run_id=payload.get("run_id"),
        requested_scopes=tuple(str(scope) for scope in payload.get("requested_scopes", ())),
    )


def _optional_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_execution_id(value: object) -> ExecutionId | None:
    if value is None:
        return None
    return ExecutionId(str(value))


def _optional_goal_id(value: object) -> WorkerGoalId | None:
    if value is None:
        return None
    return WorkerGoalId(str(value))


def _optional_responsibility_id(value: object) -> ResponsibilityId | None:
    if value is None:
        return None
    return ResponsibilityId(str(value))


def _optional_wake_up_id(value: object) -> WakeUpId | None:
    if value is None:
        return None
    return WakeUpId(str(value))


def _optional_dependency_ref(value: object) -> ExternalDependencyReference | None:
    if value is None:
        return None
    return ExternalDependencyReference(str(value))
