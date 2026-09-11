# © Artur Czarnecki. All rights reserved.

"""Durable session checkpoint with atomic writes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.atomic_io import atomic_write_json
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QUALIFICATION_SESSION_SCHEMA_VERSION,
    FinalizationPhase,
    QualificationSessionState,
    QualificationSpec,
    SourceFingerprintSnapshot,
)


class IllegalSessionTransitionError(RuntimeError):
    """Session state machine violation."""


_ALLOWED_TRANSITIONS: dict[QualificationSessionState, frozenset[QualificationSessionState]] = {
    QualificationSessionState.CREATED: frozenset(
        {QualificationSessionState.BLOCKED, QualificationSessionState.PRECONDITIONS_PASSED}
    ),
    QualificationSessionState.PRECONDITIONS_PASSED: frozenset(
        {QualificationSessionState.RUNNING, QualificationSessionState.BLOCKED}
    ),
    QualificationSessionState.RUNNING: frozenset(
        {
            QualificationSessionState.PARTIAL,
            QualificationSessionState.INVALID,
            QualificationSessionState.FINALIZING,
        }
    ),
    QualificationSessionState.PARTIAL: frozenset(
        {QualificationSessionState.INVALID, QualificationSessionState.FINALIZING}
    ),
    QualificationSessionState.FINALIZING: frozenset(
        {
            QualificationSessionState.FINALIZED,
            QualificationSessionState.FAILED_FINALIZATION,
        }
    ),
    QualificationSessionState.FAILED_FINALIZATION: frozenset(
        {QualificationSessionState.FINALIZING}
    ),
    QualificationSessionState.BLOCKED: frozenset(),
    QualificationSessionState.INVALID: frozenset(),
    QualificationSessionState.FINALIZED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class SessionCheckpoint:
    schema_version: str
    session_id: str
    task_id: str
    state: QualificationSessionState
    spec: QualificationSpec
    frozen_source: SourceFingerprintSnapshot
    completed_run_indices: tuple[int, ...]
    canonical_run_ids: tuple[str, ...]
    integrity_failures: tuple[str, ...]
    finalization_phase: FinalizationPhase | None
    artifact_generation_state: dict[str, str]


def transition_state(
    current: QualificationSessionState,
    target: QualificationSessionState,
) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise IllegalSessionTransitionError(
            f"illegal session transition: {current.value} -> {target.value}"
        )


def checkpoint_path(session_dir: Path) -> Path:
    return session_dir / "session-checkpoint.json"


def load_checkpoint(session_dir: Path) -> SessionCheckpoint | None:
    path = checkpoint_path(session_dir)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must be a JSON object")
    return _checkpoint_from_dict(raw)


def persist_checkpoint(session_dir: Path, checkpoint: SessionCheckpoint) -> None:
    atomic_write_json(checkpoint_path(session_dir), _checkpoint_to_dict(checkpoint))


def _checkpoint_to_dict(checkpoint: SessionCheckpoint) -> dict[str, object]:
    return {
        "schema_version": checkpoint.schema_version,
        "session_id": checkpoint.session_id,
        "task_id": checkpoint.task_id,
        "state": checkpoint.state.value,
        "spec": _spec_to_dict(checkpoint.spec),
        "frozen_source": _source_to_dict(checkpoint.frozen_source),
        "completed_run_indices": list(checkpoint.completed_run_indices),
        "canonical_run_ids": list(checkpoint.canonical_run_ids),
        "integrity_failures": list(checkpoint.integrity_failures),
        "finalization_phase": (
            checkpoint.finalization_phase.value if checkpoint.finalization_phase else None
        ),
        "artifact_generation_state": dict(checkpoint.artifact_generation_state),
    }


def _checkpoint_from_dict(raw: dict[str, object]) -> SessionCheckpoint:
    from testing_support.decision_e2e.local_qualification_session.session_spec import (
        spec_from_dict,
    )

    state_raw = raw.get("state")
    phase_raw = raw.get("finalization_phase")
    frozen_source_raw = raw.get("frozen_source")
    spec_raw = raw.get("spec")
    if not isinstance(state_raw, str):
        raise ValueError("checkpoint.state required")
    if not isinstance(frozen_source_raw, dict):
        raise ValueError("checkpoint.frozen_source required")
    if not isinstance(spec_raw, dict):
        raise ValueError("checkpoint.spec required")
    completed = raw.get("completed_run_indices")
    run_ids = raw.get("canonical_run_ids")
    failures = raw.get("integrity_failures")
    artifact_state = raw.get("artifact_generation_state")
    return SessionCheckpoint(
        schema_version=str(raw.get("schema_version", QUALIFICATION_SESSION_SCHEMA_VERSION)),
        session_id=str(raw.get("session_id", "")),
        task_id=str(raw.get("task_id", "")),
        state=QualificationSessionState(state_raw),
        spec=spec_from_dict(spec_raw),
        frozen_source=_source_from_dict(frozen_source_raw),
        completed_run_indices=tuple(int(item) for item in completed) if isinstance(completed, list) else (),
        canonical_run_ids=tuple(str(item) for item in run_ids) if isinstance(run_ids, list) else (),
        integrity_failures=tuple(str(item) for item in failures) if isinstance(failures, list) else (),
        finalization_phase=(
            FinalizationPhase(str(phase_raw)) if isinstance(phase_raw, str) else None
        ),
        artifact_generation_state=(
            {str(k): str(v) for k, v in artifact_state.items()}
            if isinstance(artifact_state, dict)
            else {}
        ),
    )


def _source_to_dict(snapshot: SourceFingerprintSnapshot) -> dict[str, object]:
    return {
        "repository_head_sha": snapshot.repository_head_sha,
        "blobs": [
            {
                "path": item.path,
                "content_hash": item.content_hash,
                "semantic_group": item.semantic_group,
            }
            for item in snapshot.blobs
        ],
    }


def _source_from_dict(raw: dict[str, object]) -> SourceFingerprintSnapshot:
    from testing_support.decision_e2e.local_qualification_session.contracts import (
        SourceBlobFingerprint,
    )

    blobs_raw = raw.get("blobs")
    blobs: list[SourceBlobFingerprint] = []
    if isinstance(blobs_raw, list):
        for item in blobs_raw:
            if not isinstance(item, dict):
                continue
            blobs.append(
                SourceBlobFingerprint(
                    path=str(item.get("path", "")),
                    content_hash=str(item.get("content_hash", "")),
                    semantic_group=str(item.get("semantic_group", "")),
                )
            )
    return SourceFingerprintSnapshot(
        repository_head_sha=str(raw.get("repository_head_sha", "")),
        blobs=tuple(blobs),
    )


def _spec_to_dict(spec: QualificationSpec) -> dict[str, object]:
    from testing_support.decision_e2e.local_qualification_session.session_spec import spec_to_dict

    return spec_to_dict(spec)
