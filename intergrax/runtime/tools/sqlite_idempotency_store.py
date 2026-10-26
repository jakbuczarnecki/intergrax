# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
import pickle
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
    assert_operation_identity_compatible,
    assert_tool_idempotency_ledger_schema_version,
    ClaimOutcome,
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationOperationIdentity,
    InvocationStatus,
    PreEffectSuspendedWorkRecoveryAuthority,
    validate_pre_effect_recovery_authority,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.tools.execution_models import ToolExecutionResult


class SQLiteIdempotencyStore(IdempotencyStore):
    """Durable single-host ledger — local file SQLite, not a multi-host primitive."""

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.DURABLE_SINGLE_HOST

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_ledger (
                    tenant_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_blob TEXT,
                    owner_id TEXT,
                    lease_expires_at TEXT,
                    fence INTEGER NOT NULL DEFAULT 0,
                    operation_tool_id TEXT,
                    operation_fingerprint TEXT,
                    schema_version TEXT,
                    PRIMARY KEY (tenant_id, key)
                )
                """
            )
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(idempotency_ledger)"
                ).fetchall()
            }
            if "result_blob" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_ledger ADD COLUMN result_blob TEXT"
                )
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE idempotency_ledger ADD COLUMN owner_id TEXT")
            if "lease_expires_at" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_ledger ADD COLUMN lease_expires_at TEXT"
                )
            if "fence" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_ledger ADD COLUMN fence INTEGER NOT NULL DEFAULT 0",
                )
            if "operation_tool_id" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_ledger ADD COLUMN operation_tool_id TEXT"
                )
            if "operation_fingerprint" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_ledger ADD COLUMN operation_fingerprint TEXT",
                )
            if "external_effect_may_have_started" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_ledger "
                    "ADD COLUMN external_effect_may_have_started INTEGER NOT NULL DEFAULT 0",
                )
            if "schema_version" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_ledger ADD COLUMN schema_version TEXT"
                )

    @staticmethod
    def _require_ledger_schema(row: sqlite3.Row) -> None:
        assert_tool_idempotency_ledger_schema_version(row["schema_version"])

    def _reject_stale_or_schema_after_mutation_failure(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        now: datetime,
        stale_message: str,
    ) -> None:
        row = conn.execute(
            """
            SELECT schema_version, status, owner_id, fence, lease_expires_at
            FROM idempotency_ledger
            WHERE tenant_id = ? AND key = ?
            """,
            (tenant_id, key),
        ).fetchone()
        if row is None:
            raise StaleClaimError(stale_message)
        self._require_ledger_schema(row)
        if (
            InvocationStatus(row["status"]) is not InvocationStatus.STARTED
            or row["owner_id"] != claim.owner_id
            or int(row["fence"]) != claim.fence
            or datetime.fromisoformat(row["lease_expires_at"]) <= now
        ):
            raise StaleClaimError(stale_message)
        raise StaleClaimError(stale_message)

    def _row_to_claim(
        self,
        row: sqlite3.Row,
        *,
        tenant_id: str,
        key: str,
    ) -> InvocationClaim | None:
        if row["owner_id"] is None or row["lease_expires_at"] is None:
            return None
        operation_identity = self._row_operation_identity(row)
        return InvocationClaim(
            tenant_id=tenant_id,
            key=key,
            owner_id=row["owner_id"],
            lease_expires_at=datetime.fromisoformat(row["lease_expires_at"]),
            fence=int(row["fence"]),
            operation_identity=operation_identity,
        )

    @staticmethod
    def _row_operation_identity(row: sqlite3.Row) -> InvocationOperationIdentity | None:
        tool_id = row["operation_tool_id"]
        fingerprint = row["operation_fingerprint"]
        if tool_id is None or fingerprint is None:
            return None
        return InvocationOperationIdentity(
            tool_id=tool_id,
            operation_fingerprint=fingerprint,
        )

    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status, schema_version FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()
        if row is None:
            return None
        self._require_ledger_schema(row)
        return InvocationStatus(row["status"])

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
        operation_identity: InvocationOperationIdentity | None = None,
    ) -> ClaimResult:
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status, result_blob, owner_id, lease_expires_at, fence,
                       operation_tool_id, operation_fingerprint, schema_version
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()

            if row is None:
                claim = InvocationClaim(
                    tenant_id=tenant_id,
                    key=key,
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    fence=1,
                    operation_identity=operation_identity,
                )
                conn.execute(
                    """
                    INSERT INTO idempotency_ledger
                        (tenant_id, key, status, owner_id, lease_expires_at, fence,
                         operation_tool_id, operation_fingerprint, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        key,
                        InvocationStatus.STARTED.value,
                        owner_id,
                        lease_expires_at.isoformat(),
                        claim.fence,
                        operation_identity.tool_id if operation_identity else None,
                        (
                            operation_identity.operation_fingerprint
                            if operation_identity
                            else None
                        ),
                        SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                    ),
                )
                conn.commit()
                return ClaimResult(outcome=ClaimOutcome.ACQUIRED, claim=claim)

            self._require_ledger_schema(row)
            status = InvocationStatus(row["status"])
            if status == InvocationStatus.COMPLETED:
                assert_operation_identity_compatible(
                    self._row_operation_identity(row),
                    operation_identity,
                )
                result_blob = row["result_blob"]
                if result_blob is None:
                    raise RuntimeError(
                        "Ledger inconsistency: COMPLETED without result_blob."
                    )
                completed = pickle.loads(base64.b64decode(result_blob.encode("ascii")))
                conn.commit()
                return ClaimResult(
                    outcome=ClaimOutcome.REPLAY_COMPLETED,
                    completed_result=completed,
                )

            if status == InvocationStatus.UNCERTAIN:
                conn.commit()
                return ClaimResult(outcome=ClaimOutcome.UNCERTAIN)

            stored_claim = self._row_to_claim(row, tenant_id=tenant_id, key=key)
            if stored_claim is None:
                raise RuntimeError(
                    f"Ledger inconsistency: STARTED without ownership for key={key}"
                )

            if stored_claim.lease_expires_at > now:
                if stored_claim.owner_id == owner_id:
                    assert_operation_identity_compatible(
                        self._row_operation_identity(row),
                        operation_identity,
                    )
                    conn.commit()
                    return ClaimResult(
                        outcome=ClaimOutcome.ACQUIRED, claim=stored_claim
                    )
                conn.commit()
                return ClaimResult(outcome=ClaimOutcome.BLOCKED_ACTIVE)

            updated = conn.execute(
                """
                UPDATE idempotency_ledger
                SET status = ?
                WHERE tenant_id = ? AND key = ? AND fence = ?
                  AND schema_version = ?
                """,
                (
                    InvocationStatus.UNCERTAIN.value,
                    tenant_id,
                    key,
                    stored_claim.fence,
                    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                ),
            )
            if updated.rowcount != 1:
                self._reject_stale_or_schema_after_mutation_failure(
                    conn,
                    tenant_id=tenant_id,
                    key=key,
                    claim=stored_claim,
                    now=now,
                    stale_message=(
                        f"Stale uncertain transition rejected for key={key} "
                        f"fence={stored_claim.fence}."
                    ),
                )
            conn.commit()
            return ClaimResult(outcome=ClaimOutcome.UNCERTAIN)

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        del completed_ttl_seconds
        blob = base64.b64encode(
            pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL),
        ).decode("ascii")
        now = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE idempotency_ledger
                SET status = ?, result_blob = ?
                WHERE tenant_id = ? AND key = ?
                  AND status = ?
                  AND owner_id = ?
                  AND fence = ?
                  AND lease_expires_at > ?
                  AND schema_version = ?
                """,
                (
                    InvocationStatus.COMPLETED.value,
                    blob,
                    tenant_id,
                    key,
                    InvocationStatus.STARTED.value,
                    claim.owner_id,
                    claim.fence,
                    now.isoformat(),
                    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                ),
            )
            if updated.rowcount != 1:
                self._reject_stale_or_schema_after_mutation_failure(
                    conn,
                    tenant_id=tenant_id,
                    key=key,
                    claim=claim,
                    now=now,
                    stale_message=(
                        f"Stale completion rejected for key={key} fence={claim.fence}."
                    ),
                )
            conn.commit()

    def mark_uncertain_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
    ) -> None:
        now = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE idempotency_ledger
                SET status = ?
                WHERE tenant_id = ? AND key = ?
                  AND status = ?
                  AND owner_id = ?
                  AND fence = ?
                  AND lease_expires_at > ?
                  AND schema_version = ?
                """,
                (
                    InvocationStatus.UNCERTAIN.value,
                    tenant_id,
                    key,
                    InvocationStatus.STARTED.value,
                    claim.owner_id,
                    claim.fence,
                    now.isoformat(),
                    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                ),
            )
            if updated.rowcount != 1:
                self._reject_stale_or_schema_after_mutation_failure(
                    conn,
                    tenant_id=tenant_id,
                    key=key,
                    claim=claim,
                    now=now,
                    stale_message=(
                        f"Stale uncertain transition rejected for key={key} "
                        f"fence={claim.fence}."
                    ),
                )
            conn.commit()

    def abandon_pre_effect_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
    ) -> None:
        now = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            deleted = conn.execute(
                """
                DELETE FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                  AND status = ?
                  AND owner_id = ?
                  AND fence = ?
                  AND lease_expires_at > ?
                  AND external_effect_may_have_started = 0
                  AND schema_version = ?
                """,
                (
                    tenant_id,
                    key,
                    InvocationStatus.STARTED.value,
                    claim.owner_id,
                    claim.fence,
                    now.isoformat(),
                    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                ),
            )
            if deleted.rowcount != 1:
                self._reject_stale_or_schema_after_mutation_failure(
                    conn,
                    tenant_id=tenant_id,
                    key=key,
                    claim=claim,
                    now=now,
                    stale_message=(
                        f"Stale pre-effect abandon rejected for key={key} "
                        f"fence={claim.fence}."
                    ),
                )
            conn.commit()

    def external_effect_may_have_started(
        self,
        tenant_id: str,
        key: str,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT external_effect_may_have_started, schema_version
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()
        if row is None:
            return False
        self._require_ledger_schema(row)
        return bool(row["external_effect_may_have_started"])

    def admit_external_effect_may_have_started_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
    ) -> None:
        now = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE idempotency_ledger
                SET external_effect_may_have_started = 1
                WHERE tenant_id = ? AND key = ?
                  AND status = ?
                  AND owner_id = ?
                  AND fence = ?
                  AND lease_expires_at > ?
                  AND external_effect_may_have_started = 0
                  AND schema_version = ?
                """,
                (
                    tenant_id,
                    key,
                    InvocationStatus.STARTED.value,
                    claim.owner_id,
                    claim.fence,
                    now.isoformat(),
                    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                ),
            )
            if updated.rowcount == 1:
                conn.commit()
                return
            row = conn.execute(
                """
                SELECT external_effect_may_have_started, owner_id, fence, lease_expires_at,
                       schema_version
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ? AND status = ?
                """,
                (tenant_id, key, InvocationStatus.STARTED.value),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise StaleClaimError(
                    f"Cannot admit effect for key={key}: missing active claim.",
                )
            self._require_ledger_schema(row)
            if (
                row["owner_id"] != claim.owner_id
                or int(row["fence"]) != claim.fence
                or datetime.fromisoformat(row["lease_expires_at"]) <= now
            ):
                conn.rollback()
                raise StaleClaimError(
                    f"Stale effect admission rejected for key={key} fence={claim.fence}.",
                )
            if bool(row["external_effect_may_have_started"]):
                conn.commit()
                return
            conn.rollback()
            raise StaleClaimError(
                f"Stale effect admission rejected for key={key} fence={claim.fence}.",
            )

    def reconcile_abandoned_pre_effect_not_started(
        self,
        tenant_id: str,
        key: str,
        operation_identity: InvocationOperationIdentity,
        *,
        recovery_authority: PreEffectSuspendedWorkRecoveryAuthority,
    ) -> bool:
        validate_pre_effect_recovery_authority(recovery_authority)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status, operation_tool_id, operation_fingerprint,
                       external_effect_may_have_started, schema_version
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()
            if row is not None:
                self._require_ledger_schema(row)
            if (
                row is None
                or InvocationStatus(row["status"]) is not InvocationStatus.STARTED
            ):
                conn.commit()
                return False
            if bool(row["external_effect_may_have_started"]):
                conn.commit()
                return False
            assert_operation_identity_compatible(
                self._row_operation_identity(row),
                operation_identity,
            )
            deleted = conn.execute(
                """
                DELETE FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ? AND status = ?
                  AND external_effect_may_have_started = 0
                  AND schema_version = ?
                """,
                (
                    tenant_id,
                    key,
                    InvocationStatus.STARTED.value,
                    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                ),
            )
            conn.commit()
            return deleted.rowcount == 1

    def reconcile_suspended_pre_effect_for_sanctioned_physical_redrive(
        self,
        tenant_id: str,
        key: str,
        operation_identity: InvocationOperationIdentity,
        *,
        recovery_authority: PreEffectSuspendedWorkRecoveryAuthority,
    ) -> bool:
        validate_pre_effect_recovery_authority(recovery_authority)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status, operation_tool_id, operation_fingerprint,
                       external_effect_may_have_started, schema_version
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()
            if row is not None:
                self._require_ledger_schema(row)
            if row is None:
                conn.commit()
                return False
            assert_operation_identity_compatible(
                self._row_operation_identity(row),
                operation_identity,
            )
            status = InvocationStatus(row["status"])
            if status is InvocationStatus.UNCERTAIN:
                deleted = conn.execute(
                    """
                    DELETE FROM idempotency_ledger
                    WHERE tenant_id = ? AND key = ? AND status = ?
                      AND schema_version = ?
                    """,
                    (
                        tenant_id,
                        key,
                        InvocationStatus.UNCERTAIN.value,
                        SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                    ),
                )
                conn.commit()
                return deleted.rowcount == 1
            if status is not InvocationStatus.STARTED:
                conn.commit()
                return False
            if not bool(row["external_effect_may_have_started"]):
                conn.commit()
                return False
            deleted = conn.execute(
                """
                DELETE FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ? AND status = ?
                  AND external_effect_may_have_started = 1
                  AND schema_version = ?
                """,
                (
                    tenant_id,
                    key,
                    InvocationStatus.STARTED.value,
                    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
                ),
            )
            conn.commit()
            return deleted.rowcount == 1

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        lease = lease_seconds if lease_seconds is not None else 300
        owner_id = f"legacy-{uuid4().hex}"
        outcome = self.claim(tenant_id, key, owner_id, lease)
        if outcome.outcome == ClaimOutcome.REPLAY_COMPLETED:
            raise RuntimeError(f"Invocation already exists for key={key}")
        if outcome.outcome == ClaimOutcome.BLOCKED_ACTIVE:
            raise RuntimeError(f"Invocation already exists for key={key}")
        if outcome.outcome == ClaimOutcome.UNCERTAIN:
            raise RuntimeError(
                f"Invocation uncertain for key={key}; reconciliation required.",
            )

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
    ) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT owner_id, lease_expires_at, fence, schema_version
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ? AND status = ?
                """,
                (tenant_id, key, InvocationStatus.STARTED.value),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise RuntimeError("Invalid state transition to COMPLETED.")
            self._require_ledger_schema(row)
            claim = InvocationClaim(
                tenant_id=tenant_id,
                key=key,
                owner_id=row["owner_id"],
                lease_expires_at=datetime.fromisoformat(row["lease_expires_at"]),
                fence=int(row["fence"]),
            )
            conn.commit()
        self.complete_with_claim(tenant_id, key, claim, result)

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status, result_blob, schema_version
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()

        if row is None:
            return None

        self._require_ledger_schema(row)

        if row["status"] != InvocationStatus.COMPLETED.value:
            return None

        result_blob = row["result_blob"]
        if result_blob is None:
            raise RuntimeError("Ledger inconsistency: COMPLETED without result_blob.")

        return pickle.loads(base64.b64decode(result_blob.encode("ascii")))
