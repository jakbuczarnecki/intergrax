# © Artur Czarnecki. All rights reserved.

"""Agent run step checkpoint store (architecture §40.1 · ACP-PROD-1)."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from intergrax.contracts.side_effect import AgentRunCheckpoint, SideEffectRecord


class AgentCheckpointStore(ABC):
    """Persistence port for ACP step checkpoints."""

    @abstractmethod
    def save(self, checkpoint: AgentRunCheckpoint) -> None: ...

    @abstractmethod
    def get_latest(self, run_id: str, tenant_id: str) -> AgentRunCheckpoint | None: ...


class InMemoryAgentCheckpointStore(AgentCheckpointStore):
    """Process-local checkpoint store for tests and lab hosts."""

    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str], AgentRunCheckpoint] = {}

    def save(self, checkpoint: AgentRunCheckpoint) -> None:
        self._checkpoints[(checkpoint.run_id, checkpoint.tenant_id)] = checkpoint

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
                    PRIMARY KEY (run_id, tenant_id)
                )
                """
            )

    def save(self, checkpoint: AgentRunCheckpoint) -> None:
        payload = checkpoint.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_run_checkpoints (run_id, tenant_id, payload, saved_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, tenant_id) DO UPDATE SET
                    payload = excluded.payload,
                    saved_at = excluded.saved_at
                """,
                (
                    checkpoint.run_id,
                    checkpoint.tenant_id,
                    json.dumps(payload),
                    checkpoint.saved_at.isoformat(),
                ),
            )

    def get_latest(self, run_id: str, tenant_id: str) -> AgentRunCheckpoint | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM agent_run_checkpoints
                WHERE run_id = ? AND tenant_id = ?
                """,
                (run_id, tenant_id),
            ).fetchone()
        if row is None:
            return None
        return AgentRunCheckpoint.model_validate(json.loads(row[0]))


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
