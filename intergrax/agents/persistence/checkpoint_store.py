# © Artur Czarnecki. All rights reserved.

"""Agent run step checkpoint store (architecture §40.1 · ACP-PROD-1)."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from intergrax.contracts.checkpoint_revision import (
    CheckpointAgentIdentityConflictError,
    CheckpointDurableCorruptionError,
    CheckpointRevisionConflictError,
    CheckpointSideEffectLineageError,
    CheckpointStepRegressionError,
    CheckpointStreamIdentityConflictError,
)
from intergrax.contracts.side_effect import AgentRunCheckpoint, SideEffectRecord


def validate_agent_checkpoint_for_persistence(
    checkpoint: AgentRunCheckpoint,
) -> AgentRunCheckpoint:
    """Canonical persistence acceptance rule for all AgentCheckpointStore providers."""
    validated = AgentRunCheckpoint.model_validate(checkpoint.model_dump(mode="json"))
    for record in validated.side_effect_ledger:
        if record.run_id != validated.run_id:
            raise CheckpointSideEffectLineageError(
                f"Side-effect run_id={record.run_id} does not match "
                f"checkpoint run_id={validated.run_id}.",
            )
        if record.step_index > validated.step_index:
            raise CheckpointSideEffectLineageError(
                f"Side-effect step_index={record.step_index} exceeds "
                f"checkpoint step_index={validated.step_index}.",
            )
    return validated


def _assert_stream_agent_continuity(
    current: AgentRunCheckpoint,
    incoming: AgentRunCheckpoint,
) -> None:
    if current.run_id != incoming.run_id or current.tenant_id != incoming.tenant_id:
        raise CheckpointStreamIdentityConflictError(
            "Checkpoint stream identity (run_id, tenant_id) cannot change on update.",
        )
    if current.agent_id != incoming.agent_id:
        raise CheckpointAgentIdentityConflictError(
            f"Checkpoint agent_id cannot change for run_id={incoming.run_id}: "
            f"stored={current.agent_id}, incoming={incoming.agent_id}.",
        )


class AgentCheckpointStore(ABC):
    """Persistence port for ACP step checkpoints."""

    @abstractmethod
    def save(
        self,
        checkpoint: AgentRunCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> AgentRunCheckpoint:
        """
        Persist checkpoint with revision CAS.

        Create: ``expected_revision=None`` when no row exists → revision=1.
        Update: ``expected_revision=N`` atomically succeeds only when stored revision=N.
        """

    @abstractmethod
    def get_latest(self, run_id: str, tenant_id: str) -> AgentRunCheckpoint | None: ...


class InMemoryAgentCheckpointStore(AgentCheckpointStore):
    """Process-local checkpoint store for tests and lab hosts."""

    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str], AgentRunCheckpoint] = {}
        self._lock = threading.Lock()

    def save(
        self,
        checkpoint: AgentRunCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> AgentRunCheckpoint:
        validated = validate_agent_checkpoint_for_persistence(checkpoint)
        key = (validated.run_id, validated.tenant_id)
        with self._lock:
            current = self._checkpoints.get(key)
            if current is None:
                if expected_revision is not None:
                    raise CheckpointRevisionConflictError(
                        f"No checkpoint for run_id={validated.run_id}; "
                        f"expected_revision={expected_revision} is invalid for create.",
                    )
                stored = validated.model_copy(update={"revision": 1})
                self._checkpoints[key] = stored
                return stored

            if expected_revision is None:
                raise CheckpointRevisionConflictError(
                    f"Checkpoint exists for run_id={validated.run_id}; "
                    "expected_revision is required for update.",
                )
            if expected_revision != current.revision:
                raise CheckpointRevisionConflictError(
                    f"Stale checkpoint writer for run_id={validated.run_id}: "
                    f"expected_revision={expected_revision}, current={current.revision}.",
                )
            _assert_stream_agent_continuity(current, validated)
            if validated.step_index < current.step_index:
                raise CheckpointStepRegressionError(
                    f"Checkpoint step regression for run_id={validated.run_id}: "
                    f"new step_index={validated.step_index}, current={current.step_index}.",
                )
            stored = validated.model_copy(update={"revision": current.revision + 1})
            self._checkpoints[key] = stored
            return stored

    def get_latest(self, run_id: str, tenant_id: str) -> AgentRunCheckpoint | None:
        return self._checkpoints.get((run_id, tenant_id))


class SQLiteAgentCheckpointStore(AgentCheckpointStore):
    """SQLite-backed agent checkpoint store."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_run_checkpoints (
                    run_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (run_id, tenant_id)
                )
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(agent_run_checkpoints)").fetchall()
            }
            if "revision" not in columns:
                conn.execute(
                    "ALTER TABLE agent_run_checkpoints ADD COLUMN revision INTEGER NOT NULL DEFAULT 1",
                )

    def save(
        self,
        checkpoint: AgentRunCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> AgentRunCheckpoint:
        validated = validate_agent_checkpoint_for_persistence(checkpoint)
        payload = validated.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT revision, payload FROM agent_run_checkpoints
                WHERE run_id = ? AND tenant_id = ?
                """,
                (validated.run_id, validated.tenant_id),
            ).fetchone()

            if row is None:
                if expected_revision is not None:
                    conn.rollback()
                    raise CheckpointRevisionConflictError(
                        f"No checkpoint for run_id={validated.run_id}; "
                        f"expected_revision={expected_revision} is invalid for create.",
                    )
                stored = validated.model_copy(update={"revision": 1})
                payload["revision"] = 1
                conn.execute(
                    """
                    INSERT INTO agent_run_checkpoints
                        (run_id, tenant_id, payload, saved_at, revision)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        validated.run_id,
                        validated.tenant_id,
                        json.dumps(payload),
                        validated.saved_at.isoformat(),
                        1,
                    ),
                )
                conn.commit()
                return stored

            current_revision = int(row[0])
            current_payload = json.loads(row[1])
            current_checkpoint = validate_agent_checkpoint_for_persistence(
                AgentRunCheckpoint.model_validate(current_payload),
            )
            current_step_index = current_checkpoint.step_index

            if expected_revision is None:
                conn.rollback()
                raise CheckpointRevisionConflictError(
                    f"Checkpoint exists for run_id={validated.run_id}; "
                    "expected_revision is required for update.",
                )
            if expected_revision != current_revision:
                conn.rollback()
                raise CheckpointRevisionConflictError(
                    f"Stale checkpoint writer for run_id={validated.run_id}: "
                    f"expected_revision={expected_revision}, current={current_revision}.",
                )
            _assert_stream_agent_continuity(current_checkpoint, validated)
            if validated.step_index < current_step_index:
                conn.rollback()
                raise CheckpointStepRegressionError(
                    f"Checkpoint step regression for run_id={validated.run_id}: "
                    f"new step_index={validated.step_index}, current={current_step_index}.",
                )

            next_revision = current_revision + 1
            stored = validated.model_copy(update={"revision": next_revision})
            payload["revision"] = next_revision
            updated = conn.execute(
                """
                UPDATE agent_run_checkpoints
                SET payload = ?, saved_at = ?, revision = ?
                WHERE run_id = ? AND tenant_id = ? AND revision = ?
                """,
                (
                    json.dumps(payload),
                    validated.saved_at.isoformat(),
                    next_revision,
                    validated.run_id,
                    validated.tenant_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                conn.rollback()
                raise CheckpointRevisionConflictError(
                    f"Concurrent checkpoint CAS failure for run_id={validated.run_id}.",
                )
            conn.commit()
            return stored

    def get_latest(self, run_id: str, tenant_id: str) -> AgentRunCheckpoint | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload, revision FROM agent_run_checkpoints
                WHERE run_id = ? AND tenant_id = ?
                """,
                (run_id, tenant_id),
            ).fetchone()
        if row is None:
            return None
        try:
            raw_payload = json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise CheckpointDurableCorruptionError(
                f"Invalid checkpoint JSON for run_id={run_id}, tenant_id={tenant_id}.",
            ) from exc
        checkpoint = AgentRunCheckpoint.model_validate(raw_payload)
        revision = int(row[1])
        if checkpoint.revision != revision:
            raise CheckpointDurableCorruptionError(
                f"Checkpoint payload revision={checkpoint.revision} disagrees with "
                f"authoritative column revision={revision} for run_id={run_id}.",
            )
        if checkpoint.run_id != run_id or checkpoint.tenant_id != tenant_id:
            raise CheckpointStreamIdentityConflictError(
                f"Stored checkpoint identity does not match lookup "
                f"run_id={run_id}, tenant_id={tenant_id}.",
            )
        return validate_agent_checkpoint_for_persistence(checkpoint)


def build_checkpoint(
    *,
    run_id: str,
    tenant_id: str,
    agent_id: str,
    step_index: int,
    state_root: dict[str, Any],
    side_effect_ledger: list[SideEffectRecord],
    trace_step_count: int,
) -> AgentRunCheckpoint:
    return AgentRunCheckpoint(
        run_id=run_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        step_index=step_index,
        state_root=dict(state_root),
        side_effect_ledger=list(side_effect_ledger),
        trace_step_count=trace_step_count,
    )
