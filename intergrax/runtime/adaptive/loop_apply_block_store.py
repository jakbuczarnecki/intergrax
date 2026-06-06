# © Artur Czarnecki. All rights reserved.

"""Blocked adaptive loop kinds after verification failure (Phase W-ADAPT-5.2)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind


class LoopApplyBlockRecord(BaseModel):
    """Blocked loop kind with audit metadata."""

    model_config = ConfigDict(extra="forbid")

    loop_kind: AdaptiveLoopKind
    reason: str
    blocked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tenant_id: str | None = None


class LoopApplyBlockStore(Protocol):
    """Tracks loop kinds blocked from auto-apply after verify failure."""

    def is_blocked(
        self,
        loop_kind: AdaptiveLoopKind,
        *,
        tenant_id: str | None = None,
    ) -> bool: ...

    def block(
        self,
        loop_kind: AdaptiveLoopKind,
        *,
        reason: str,
        tenant_id: str | None = None,
    ) -> LoopApplyBlockRecord: ...

    def list_blocks(self) -> list[LoopApplyBlockRecord]: ...

    def clear(self) -> None: ...


def default_loop_apply_block_store_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "adaptive_harness" / "loop_apply_blocks.db"


class InMemoryLoopApplyBlockStore:
    """In-process loop apply block store for unit tests."""

    def __init__(self) -> None:
        self._blocks: list[LoopApplyBlockRecord] = []

    def is_blocked(
        self,
        loop_kind: AdaptiveLoopKind,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        for record in self._blocks:
            if record.loop_kind != loop_kind:
                continue
            if tenant_id is not None and record.tenant_id not in {None, tenant_id}:
                continue
            return True
        return False

    def block(
        self,
        loop_kind: AdaptiveLoopKind,
        *,
        reason: str,
        tenant_id: str | None = None,
    ) -> LoopApplyBlockRecord:
        record = LoopApplyBlockRecord(
            loop_kind=loop_kind,
            reason=reason,
            tenant_id=tenant_id,
        )
        self._blocks.append(record)
        return record

    def list_blocks(self) -> list[LoopApplyBlockRecord]:
        return list(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()


class SQLiteLoopApplyBlockStore:
    """SQLite-backed loop apply block store."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or default_loop_apply_block_store_path()
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS loop_apply_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def is_blocked(
        self,
        loop_kind: AdaptiveLoopKind,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        for record in self.list_blocks():
            if record.loop_kind != loop_kind:
                continue
            if tenant_id is not None and record.tenant_id not in {None, tenant_id}:
                continue
            return True
        return False

    def block(
        self,
        loop_kind: AdaptiveLoopKind,
        *,
        reason: str,
        tenant_id: str | None = None,
    ) -> LoopApplyBlockRecord:
        record = LoopApplyBlockRecord(
            loop_kind=loop_kind,
            reason=reason,
            tenant_id=tenant_id,
        )
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO loop_apply_blocks (payload_json) VALUES (?)",
                (record.model_dump_json(),),
            )
        return record

    def list_blocks(self) -> list[LoopApplyBlockRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM loop_apply_blocks ORDER BY id ASC"
            ).fetchall()
        return [LoopApplyBlockRecord.model_validate_json(row["payload_json"]) for row in rows]

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM loop_apply_blocks")
