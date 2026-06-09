# © Artur Czarnecki. All rights reserved.

"""Durable cross-host registry snapshot persistence (AUDIT-IDEAL-19.1)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot


class SqliteRegistrySnapshotStore:
    """Persist materialized registry id sets for cross-host audit and replay."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS registry_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    host_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def persist(self, snapshot: HarnessRegistrySnapshot, *, host_id: str, snapshot_id: str) -> None:
        payload = {
            "tool_ids": list(snapshot.tool_ids()),
            "skill_ids": list(snapshot.skill_ids()),
            "prompt_ids": list(snapshot.prompt_ids()),
            "agent_contract_ids": list(snapshot.agent_contract_ids()),
            "evaluation_registry_ids": list(snapshot.evaluation_registry_ids()),
        }
        captured_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO registry_snapshots(snapshot_id, host_id, captured_at, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    host_id=excluded.host_id,
                    captured_at=excluded.captured_at,
                    payload_json=excluded.payload_json
                """,
                (snapshot_id, host_id, captured_at, json.dumps(payload)),
            )

    def load(self, snapshot_id: str) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM registry_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])


def persist_registry_snapshot(
    snapshot: HarnessRegistrySnapshot,
    *,
    host_id: str,
    db_path: Path,
    snapshot_id: str | None = None,
) -> str:
    """Persist snapshot; returns assigned snapshot_id."""
    sid = snapshot_id or f"snap_{host_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    SqliteRegistrySnapshotStore(db_path).persist(snapshot, host_id=host_id, snapshot_id=sid)
    return sid
