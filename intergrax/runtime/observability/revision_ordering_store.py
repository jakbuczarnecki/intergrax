# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable SQLite persistence for tenant-scoped revision ordering (TRACE-BITEMP-2)."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from intergrax.contracts.bitemporal_knowledge import (
    KnowledgeOrderingScope,
    KnowledgeRevisionAcceptance,
    KnowledgeRevisionId,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionPositionRecord,
    KnowledgeRevisionResolutionRecord,
    KnowledgeRevisionResolutionReason,
    KnowledgeRevisionResolutionSource,
    KnowledgeRevisionWatermark,
    OrphanedDurableRevisionDetectionSource,
    OrphanedDurableRevisionDisposition,
    OrphanedDurableRevisionReason,
    OrphanedDurableRevisionRecord,
    ResolutionAuthority,
    RevisionAcceptanceConflictError,
    RevisionAcceptanceKey,
    RevisionFencingGeneration,
    StaleRevisionFencingError,
    UnknownKnowledgeRevisionPositionError,
    UnresolvedPositionResolutionError,
    compute_finalized_watermark,
    lifecycle_blocks_watermark,
)
from intergrax.utils.time_provider import SystemTimeProvider, TimeProvider

_CLOSED_ERROR = "Revision ordering store is closed"


@dataclass(frozen=True, slots=True)
class RevisionOrderingStoreTestHooks:
    """Deterministic fault injection for provider qualification tests."""

    pause_after_allocate: bool = False
    allow_late_physical_write: bool = False


class RevisionOrderingSQLiteStore:
    """Shared SQLite backend for multi-instance revision ordering."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        time_provider: type[TimeProvider] = SystemTimeProvider,
        test_hooks: RevisionOrderingStoreTestHooks | None = None,
    ) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._time_provider = time_provider
        self._test_hooks = test_hooks or RevisionOrderingStoreTestHooks()
        self._connection = sqlite3.connect(
            str(path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        self._initialize_schema()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._ensure_open()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def accept_revision(
        self,
        *,
        scope: KnowledgeOrderingScope,
        revision_id: KnowledgeRevisionId,
        acceptance_key: RevisionAcceptanceKey,
    ) -> KnowledgeRevisionAcceptance:
        tenant = scope.tenant_id
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT revision_id, position
                FROM knowledge_acceptance_bindings
                WHERE tenant_id = ? AND acceptance_key = ?
                """,
                (tenant, acceptance_key.value),
            ).fetchone()
            if existing is not None:
                bound_revision = KnowledgeRevisionId(existing["revision_id"])
                if bound_revision != revision_id:
                    raise RevisionAcceptanceConflictError(
                        "acceptance key already bound to a different knowledge revision"
                    )
                position = KnowledgeRevisionPosition(scope=scope, value=int(existing["position"]))
                self._finalize_acceptance_if_needed(
                    conn,
                    scope=scope,
                    position=position,
                    revision_id=revision_id,
                )
                return KnowledgeRevisionAcceptance(
                    revision_id=revision_id,
                    acceptance_key=acceptance_key,
                    position=position,
                )

            writer_fence = self._current_fencing_generation(conn, scope)
            position_value = self._allocate_position(conn, tenant)
            position = KnowledgeRevisionPosition(scope=scope, value=position_value)
            lifecycle = (
                KnowledgeRevisionPositionLifecycle.UNRESOLVED
                if self._test_hooks.pause_after_allocate
                else KnowledgeRevisionPositionLifecycle.ACCEPTED
            )
            conn.execute(
                """
                INSERT INTO knowledge_acceptance_bindings (
                    tenant_id, acceptance_key, revision_id, position
                ) VALUES (?, ?, ?, ?)
                """,
                (tenant, acceptance_key.value, revision_id.value, position_value),
            )
            conn.execute(
                """
                INSERT INTO knowledge_position_states (
                    tenant_id,
                    position,
                    lifecycle,
                    revision_id,
                    writer_fencing_generation,
                    canonical_fencing_generation,
                    canonical_accepted
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant,
                    position_value,
                    lifecycle.value,
                    revision_id.value if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED else None,
                    writer_fence,
                    writer_fence if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED else None,
                    1 if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED else 0,
                ),
            )
            if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
                return KnowledgeRevisionAcceptance(
                    revision_id=revision_id,
                    acceptance_key=acceptance_key,
                    position=position,
                )
            return KnowledgeRevisionAcceptance(
                revision_id=revision_id,
                acceptance_key=acceptance_key,
                position=position,
            )

    def acquire_resolution_authority(self, scope: KnowledgeOrderingScope) -> ResolutionAuthority:
        tenant = scope.tenant_id
        with self._transaction() as conn:
            self._ensure_scope_row(conn, tenant)
            conn.execute(
                """
                UPDATE knowledge_ordering_scope
                SET resolution_fencing_generation = resolution_fencing_generation + 1
                WHERE tenant_id = ?
                """,
                (tenant,),
            )
            generation = int(
                conn.execute(
                    "SELECT resolution_fencing_generation FROM knowledge_ordering_scope WHERE tenant_id = ?",
                    (tenant,),
                ).fetchone()["resolution_fencing_generation"]
            )
        fencing = RevisionFencingGeneration(scope=scope, value=generation)
        return ResolutionAuthority(scope=scope, fencing_generation=fencing)

    def resolve_unresolved_position(
        self,
        *,
        position: KnowledgeRevisionPosition,
        authority: ResolutionAuthority,
        reason: KnowledgeRevisionResolutionReason,
        source: KnowledgeRevisionResolutionSource,
        actor_identity: str | None = None,
        correlation_id: str | None = None,
    ) -> KnowledgeRevisionResolutionRecord:
        scope = position.scope
        tenant = scope.tenant_id
        if authority.scope != scope:
            raise UnresolvedPositionResolutionError("resolution authority scope mismatch")
        with self._transaction() as conn:
            current_fence = self._current_fencing_generation(conn, scope)
            if authority.fencing_generation.value < current_fence:
                raise StaleRevisionFencingError("resolution authority fencing generation is stale")

            row = conn.execute(
                """
                SELECT lifecycle, writer_fencing_generation
                FROM knowledge_position_states
                WHERE tenant_id = ? AND position = ?
                """,
                (tenant, position.value),
            ).fetchone()
            if row is None:
                raise UnknownKnowledgeRevisionPositionError(
                    f"knowledge revision position {position.value} was never allocated"
                )
            lifecycle = KnowledgeRevisionPositionLifecycle(row["lifecycle"])
            if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
                raise UnresolvedPositionResolutionError("position is already accepted")
            if lifecycle is KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED:
                raise UnresolvedPositionResolutionError("position is already terminal non-committed")
            if lifecycle not in (
                KnowledgeRevisionPositionLifecycle.UNRESOLVED,
                KnowledgeRevisionPositionLifecycle.ALLOCATED,
            ):
                raise UnresolvedPositionResolutionError(f"position lifecycle {lifecycle} is not resolvable")

            updated = conn.execute(
                """
                UPDATE knowledge_position_states
                SET lifecycle = ?,
                    canonical_accepted = 0,
                    revision_id = NULL,
                    canonical_fencing_generation = ?
                WHERE tenant_id = ?
                  AND position = ?
                  AND lifecycle IN (?, ?)
                  AND writer_fencing_generation <= ?
                """,
                (
                    KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED.value,
                    authority.fencing_generation.value,
                    tenant,
                    position.value,
                    KnowledgeRevisionPositionLifecycle.UNRESOLVED.value,
                    KnowledgeRevisionPositionLifecycle.ALLOCATED.value,
                    authority.fencing_generation.value,
                ),
            )
            if updated.rowcount != 1:
                refreshed = conn.execute(
                    "SELECT lifecycle FROM knowledge_position_states WHERE tenant_id = ? AND position = ?",
                    (tenant, position.value),
                ).fetchone()
                if refreshed is None:
                    raise UnknownKnowledgeRevisionPositionError(
                        f"knowledge revision position {position.value} was never allocated"
                    )
                refreshed_lifecycle = KnowledgeRevisionPositionLifecycle(refreshed["lifecycle"])
                if refreshed_lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
                    raise UnresolvedPositionResolutionError("position became accepted concurrently")
                if refreshed_lifecycle is KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED:
                    raise UnresolvedPositionResolutionError("position already terminalized concurrently")
                raise UnresolvedPositionResolutionError("position resolution lost authoritative race")

            detected_at = self._time_provider.utc_now()
            record = KnowledgeRevisionResolutionRecord(
                scope=scope,
                position=position,
                prior_lifecycle=lifecycle,
                resulting_lifecycle=KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED,
                reason=reason,
                source=source,
                fencing_generation=authority.fencing_generation,
                detected_at=detected_at,
                actor_identity=actor_identity,
                correlation_id=correlation_id,
            )
            conn.execute(
                """
                INSERT INTO knowledge_resolution_records (
                    record_id,
                    tenant_id,
                    position,
                    prior_lifecycle,
                    resulting_lifecycle,
                    reason,
                    source,
                    fencing_generation,
                    detected_at_utc,
                    actor_identity,
                    correlation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_record_id("kres"),
                    tenant,
                    position.value,
                    record.prior_lifecycle.value,
                    record.resulting_lifecycle.value,
                    record.reason.value,
                    record.source.value,
                    record.fencing_generation.value,
                    record.detected_at.isoformat(),
                    record.actor_identity,
                    record.correlation_id,
                ),
            )
            return record

    def inject_late_physical_write(
        self,
        *,
        position: KnowledgeRevisionPosition,
        revision_id: KnowledgeRevisionId,
        stale_fencing_generation: RevisionFencingGeneration,
    ) -> OrphanedDurableRevisionRecord | None:
        if not self._test_hooks.allow_late_physical_write:
            raise RuntimeError("late physical write injection requires test hooks")
        scope = position.scope
        tenant = scope.tenant_id
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT lifecycle, canonical_fencing_generation
                FROM knowledge_position_states
                WHERE tenant_id = ? AND position = ?
                """,
                (tenant, position.value),
            ).fetchone()
            if row is None:
                raise UnknownKnowledgeRevisionPositionError(
                    f"knowledge revision position {position.value} was never allocated"
                )
            canonical_lifecycle = KnowledgeRevisionPositionLifecycle(row["lifecycle"])
            canonical_fence_value = row["canonical_fencing_generation"]
            if canonical_fence_value is None:
                return None
            winning_fence = RevisionFencingGeneration(
                scope=scope,
                value=int(canonical_fence_value),
            )
            conn.execute(
                """
                INSERT INTO knowledge_physical_payloads (
                    tenant_id, position, revision_id, writer_fencing_generation, is_quarantined
                ) VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(tenant_id, position, writer_fencing_generation) DO UPDATE SET
                    revision_id = excluded.revision_id,
                    is_quarantined = 1
                """,
                (tenant, position.value, revision_id.value, stale_fencing_generation.value),
            )
            if canonical_lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
                return None
            if canonical_lifecycle is not KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED:
                return None
            detected_at = self._time_provider.utc_now()
            orphan = OrphanedDurableRevisionRecord(
                scope=scope,
                position=position,
                stale_fencing_generation=stale_fencing_generation,
                winning_fencing_generation=winning_fence,
                canonical_lifecycle=canonical_lifecycle,
                revision_id=revision_id,
                reason=OrphanedDurableRevisionReason.TERMINAL_POSITION_LATE_WRITE,
                detection_source=OrphanedDurableRevisionDetectionSource.PROVIDER_RECONCILIATION,
                detected_at=detected_at,
                disposition=OrphanedDurableRevisionDisposition.QUARANTINED,
            )
            conn.execute(
                """
                INSERT INTO knowledge_orphan_records (
                    record_id,
                    tenant_id,
                    position,
                    stale_fencing_generation,
                    winning_fencing_generation,
                    canonical_lifecycle,
                    revision_id,
                    reason,
                    detection_source,
                    detected_at_utc,
                    disposition
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_record_id("korph"),
                    tenant,
                    position.value,
                    orphan.stale_fencing_generation.value,
                    orphan.winning_fencing_generation.value,
                    orphan.canonical_lifecycle.value,
                    orphan.revision_id.value,
                    orphan.reason.value,
                    orphan.detection_source.value,
                    orphan.detected_at.isoformat(),
                    orphan.disposition.value,
                ),
            )
            return orphan

    def canonical_fencing_generation(
        self,
        position: KnowledgeRevisionPosition,
    ) -> RevisionFencingGeneration | None:
        tenant = position.scope.tenant_id
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT canonical_fencing_generation
                FROM knowledge_position_states
                WHERE tenant_id = ? AND position = ?
                """,
                (tenant, position.value),
            ).fetchone()
        if row is None:
            raise UnknownKnowledgeRevisionPositionError(
                f"knowledge revision position {position.value} was never allocated"
            )
        value = row["canonical_fencing_generation"]
        if value is None:
            return None
        return RevisionFencingGeneration(scope=position.scope, value=int(value))

    def position_lifecycle(
        self,
        position: KnowledgeRevisionPosition,
    ) -> KnowledgeRevisionPositionLifecycle:
        tenant = position.scope.tenant_id
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT lifecycle FROM knowledge_position_states WHERE tenant_id = ? AND position = ?",
                (tenant, position.value),
            ).fetchone()
        if row is None:
            raise UnknownKnowledgeRevisionPositionError(
                f"knowledge revision position {position.value} was never allocated"
            )
        return KnowledgeRevisionPositionLifecycle(row["lifecycle"])

    def watermark(self, scope: KnowledgeOrderingScope) -> KnowledgeRevisionWatermark:
        records = self._all_position_records(scope)
        return compute_finalized_watermark(scope=scope, records=records)

    def records_through(
        self,
        watermark: KnowledgeRevisionWatermark,
    ) -> tuple[KnowledgeRevisionPositionRecord, ...]:
        scope = watermark.scope
        tenant = scope.tenant_id
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT position, lifecycle, revision_id, canonical_accepted
                FROM knowledge_position_states
                WHERE tenant_id = ? AND position <= ?
                ORDER BY position ASC
                """,
                (tenant, watermark.finalized_through_value),
            ).fetchall()
        records: list[KnowledgeRevisionPositionRecord] = []
        for row in rows:
            lifecycle = KnowledgeRevisionPositionLifecycle(row["lifecycle"])
            if lifecycle_blocks_watermark(lifecycle):
                raise ValueError("records_through watermark includes blocking lifecycle")
            records.append(
                _position_record_from_row(scope=scope, row=row),
            )
        return tuple(records)

    def unresolved_positions(
        self,
        scope: KnowledgeOrderingScope,
    ) -> tuple[KnowledgeRevisionPosition, ...]:
        tenant = scope.tenant_id
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT position
                FROM knowledge_position_states
                WHERE tenant_id = ?
                  AND lifecycle IN (?, ?)
                ORDER BY position ASC
                """,
                (
                    tenant,
                    KnowledgeRevisionPositionLifecycle.UNRESOLVED.value,
                    KnowledgeRevisionPositionLifecycle.ALLOCATED.value,
                ),
            ).fetchall()
        return tuple(
            KnowledgeRevisionPosition(scope=scope, value=int(row["position"]))
            for row in rows
        )

    def list_orphan_records(
        self,
        scope: KnowledgeOrderingScope,
    ) -> tuple[OrphanedDurableRevisionRecord, ...]:
        tenant = scope.tenant_id
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT position,
                       stale_fencing_generation,
                       winning_fencing_generation,
                       canonical_lifecycle,
                       revision_id,
                       reason,
                       detection_source,
                       detected_at_utc,
                       disposition
                FROM knowledge_orphan_records
                WHERE tenant_id = ?
                ORDER BY position ASC, detected_at_utc ASC
                """,
                (tenant,),
            ).fetchall()
        return tuple(self._orphan_from_row(scope, row) for row in rows)

    def _finalize_acceptance_if_needed(
        self,
        conn: sqlite3.Connection,
        *,
        scope: KnowledgeOrderingScope,
        position: KnowledgeRevisionPosition,
        revision_id: KnowledgeRevisionId,
    ) -> None:
        tenant = scope.tenant_id
        row = conn.execute(
            """
            SELECT lifecycle, writer_fencing_generation
            FROM knowledge_position_states
            WHERE tenant_id = ? AND position = ?
            """,
            (tenant, position.value),
        ).fetchone()
        if row is None:
            raise UnknownKnowledgeRevisionPositionError(
                f"knowledge revision position {position.value} was never allocated"
            )
        lifecycle = KnowledgeRevisionPositionLifecycle(row["lifecycle"])
        if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
            return
        if lifecycle is KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED:
            raise StaleRevisionFencingError("position is terminal non-committed")
        current_fence = self._current_fencing_generation(conn, scope)
        writer_fence = int(row["writer_fencing_generation"])
        if writer_fence < current_fence:
            raise StaleRevisionFencingError("stale writer cannot canonically accept position")
        updated = conn.execute(
            """
            UPDATE knowledge_position_states
            SET lifecycle = ?,
                revision_id = ?,
                canonical_accepted = 1,
                canonical_fencing_generation = ?
            WHERE tenant_id = ?
              AND position = ?
              AND lifecycle IN (?, ?)
              AND writer_fencing_generation >= ?
            """,
            (
                KnowledgeRevisionPositionLifecycle.ACCEPTED.value,
                revision_id.value,
                writer_fence,
                tenant,
                position.value,
                KnowledgeRevisionPositionLifecycle.UNRESOLVED.value,
                KnowledgeRevisionPositionLifecycle.ALLOCATED.value,
                current_fence,
            ),
        )
        if updated.rowcount != 1:
            refreshed = conn.execute(
                "SELECT lifecycle FROM knowledge_position_states WHERE tenant_id = ? AND position = ?",
                (tenant, position.value),
            ).fetchone()
            if refreshed is None:
                raise UnknownKnowledgeRevisionPositionError(
                    f"knowledge revision position {position.value} was never allocated"
                )
            refreshed_lifecycle = KnowledgeRevisionPositionLifecycle(refreshed["lifecycle"])
            if refreshed_lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
                return
            if refreshed_lifecycle is KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED:
                raise StaleRevisionFencingError("position terminalized before acceptance completed")
            raise StaleRevisionFencingError("acceptance completion lost authoritative race")

    def _all_position_records(
        self,
        scope: KnowledgeOrderingScope,
    ) -> tuple[KnowledgeRevisionPositionRecord, ...]:
        tenant = scope.tenant_id
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT position, lifecycle, revision_id, canonical_accepted
                FROM knowledge_position_states
                WHERE tenant_id = ?
                ORDER BY position ASC
                """,
                (tenant,),
            ).fetchall()
        return tuple(
            _position_record_from_row(scope=scope, row=row)
            for row in rows
        )

    def _allocate_position(self, conn: sqlite3.Connection, tenant: str) -> int:
        self._ensure_scope_row(conn, tenant)
        row = conn.execute(
            """
            UPDATE knowledge_ordering_scope
            SET next_position = next_position + 1
            WHERE tenant_id = ?
            RETURNING next_position - 1
            """,
            (tenant,),
        ).fetchone()
        return int(row[0])

    def _current_fencing_generation(
        self,
        conn: sqlite3.Connection,
        scope: KnowledgeOrderingScope,
    ) -> int:
        tenant = scope.tenant_id
        self._ensure_scope_row(conn, tenant)
        row = conn.execute(
            "SELECT resolution_fencing_generation FROM knowledge_ordering_scope WHERE tenant_id = ?",
            (tenant,),
        ).fetchone()
        return int(row["resolution_fencing_generation"])

    def _ensure_scope_row(self, conn: sqlite3.Connection, tenant: str) -> None:
        conn.execute(
            """
            INSERT INTO knowledge_ordering_scope (tenant_id)
            VALUES (?)
            ON CONFLICT(tenant_id) DO NOTHING
            """,
            (tenant,),
        )

    def _initialize_schema(self) -> None:
        with self._lock:
            self._ensure_open()
            self._connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 30000;

                CREATE TABLE IF NOT EXISTS knowledge_ordering_scope (
                    tenant_id TEXT PRIMARY KEY,
                    next_position INTEGER NOT NULL DEFAULT 1,
                    resolution_fencing_generation INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS knowledge_acceptance_bindings (
                    tenant_id TEXT NOT NULL,
                    acceptance_key TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, acceptance_key),
                    UNIQUE (tenant_id, position)
                );

                CREATE TABLE IF NOT EXISTS knowledge_position_states (
                    tenant_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    lifecycle TEXT NOT NULL,
                    revision_id TEXT,
                    writer_fencing_generation INTEGER NOT NULL DEFAULT 0,
                    canonical_fencing_generation INTEGER,
                    canonical_accepted INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, position)
                );

                CREATE TABLE IF NOT EXISTS knowledge_physical_payloads (
                    tenant_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    revision_id TEXT NOT NULL,
                    writer_fencing_generation INTEGER NOT NULL,
                    is_quarantined INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (tenant_id, position, writer_fencing_generation)
                );

                CREATE TABLE IF NOT EXISTS knowledge_resolution_records (
                    record_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    prior_lifecycle TEXT NOT NULL,
                    resulting_lifecycle TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    source TEXT NOT NULL,
                    fencing_generation INTEGER NOT NULL,
                    detected_at_utc TEXT NOT NULL,
                    actor_identity TEXT,
                    correlation_id TEXT
                );

                CREATE TABLE IF NOT EXISTS knowledge_orphan_records (
                    record_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    stale_fencing_generation INTEGER NOT NULL,
                    winning_fencing_generation INTEGER NOT NULL,
                    canonical_lifecycle TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    detection_source TEXT NOT NULL,
                    detected_at_utc TEXT NOT NULL,
                    disposition TEXT NOT NULL
                );
                """
            )

    def _orphan_from_row(
        self,
        scope: KnowledgeOrderingScope,
        row: sqlite3.Row,
    ) -> OrphanedDurableRevisionRecord:
        position = KnowledgeRevisionPosition(scope=scope, value=int(row["position"]))
        return OrphanedDurableRevisionRecord(
            scope=scope,
            position=position,
            stale_fencing_generation=RevisionFencingGeneration(scope=scope, value=int(row["stale_fencing_generation"])),
            winning_fencing_generation=RevisionFencingGeneration(
                scope=scope,
                value=int(row["winning_fencing_generation"]),
            ),
            canonical_lifecycle=KnowledgeRevisionPositionLifecycle(row["canonical_lifecycle"]),
            revision_id=KnowledgeRevisionId(row["revision_id"]),
            reason=OrphanedDurableRevisionReason(row["reason"]),
            detection_source=OrphanedDurableRevisionDetectionSource(row["detection_source"]),
            detected_at=datetime.fromisoformat(row["detected_at_utc"]),
            disposition=OrphanedDurableRevisionDisposition(row["disposition"]),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(_CLOSED_ERROR)


def _new_record_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _position_record_from_row(
    *,
    scope: KnowledgeOrderingScope,
    row: sqlite3.Row,
) -> KnowledgeRevisionPositionRecord:
    lifecycle = KnowledgeRevisionPositionLifecycle(row["lifecycle"])
    accepted_revision_id: KnowledgeRevisionId | None = None
    if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
        if int(row["canonical_accepted"]) != 1 or row["revision_id"] is None:
            raise ValueError("ACCEPTED position missing canonical revision binding")
        accepted_revision_id = KnowledgeRevisionId(row["revision_id"])
    return KnowledgeRevisionPositionRecord(
        position=KnowledgeRevisionPosition(scope=scope, value=int(row["position"])),
        lifecycle=lifecycle,
        accepted_revision_id=accepted_revision_id,
    )
