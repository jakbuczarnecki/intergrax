# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite durable TOOL-LIFE-X stores (L3-R8 — single-host shared durability).

Cross-process visibility via WAL-backed local SQLite — not multi-host PostgreSQL/Redis.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

from intergrax.contracts.certified_owner_loss_evidence import (
    CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
    CertifiedOwnerLossEvidenceRejectedError,
    StagedCertifiedOwnerLossEvidenceV1,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
    ExternalOperationState,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.contracts.worker_execution_commit_authority_loss_decision import (
    WorkerExecutionCommitAuthorityLossDecisionPort,
    WorkerExecutionCommitAuthorityLossDecisionV1,
)
from intergrax.contracts.worker_execution_ownership_revocation import (
    CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID,
    WorkerExecutionDirectRevocationInvocationError,
    WorkerExecutionOwnershipRevocationAuthorizationV1,
    WorkerExecutionOwnershipRevocationPort,
    WorkerExecutionOwnershipRevocationV1,
    worker_execution_revocation_proof_ref,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    require_certified_owner_loss_composition,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateUpdate,
    ExternalOperationTenantScopeViolationError,
    StaleExternalOperationStateError,
    _ExternalOperationStateCertificationSemantics,
    _require_tenant_bound,
    _store_key,
    _utc_now,
    _validate_tenant_id,
    apply_external_operation_state_update,
    cas_update_blocked_by_bound_worker_revocation,
)
from intergrax.runtime.external_operations.worker_execution_commit_authority_loss import (
    _require_decision_authority_invocation,
    mint_worker_execution_commit_authority_loss_decision,
    verify_worker_execution_commit_authority_loss_decision,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
    assert_authoritative_worker_execution_revocation_allowed,
    verify_worker_execution_ownership_revocation_authorization,
    _require_recorder_revocation_invocation,
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=FULL;")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tool_life_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS external_operation_states (
            tenant_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            PRIMARY KEY (tenant_id, operation_id)
        );
        """,
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_execution_ownership_revocations (
            tenant_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            worker_instance_id TEXT NOT NULL,
            ownership_epoch INTEGER NOT NULL,
            revocation_id TEXT NOT NULL,
            revoked_at_utc TEXT NOT NULL,
            proof_ref TEXT NOT NULL,
            PRIMARY KEY (
                tenant_id,
                operation_id,
                worker_instance_id,
                ownership_epoch
            )
        );
        """,
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_execution_commit_authority_loss_decisions (
            tenant_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            worker_instance_id TEXT NOT NULL,
            ownership_epoch INTEGER NOT NULL,
            decision_id TEXT NOT NULL,
            decision_authority_id TEXT NOT NULL,
            bound_store_revision INTEGER NOT NULL,
            loss_reason TEXT NOT NULL,
            decided_at_utc TEXT NOT NULL,
            valid_until_utc TEXT NOT NULL,
            decision_digest TEXT NOT NULL,
            PRIMARY KEY (
                tenant_id,
                operation_id,
                worker_instance_id,
                ownership_epoch
            )
        );
        """,
    )


def _insert_commit_authority_loss_decision_row(
    conn: sqlite3.Connection,
    *,
    decision: WorkerExecutionCommitAuthorityLossDecisionV1,
) -> None:
    conn.execute(
        """
        INSERT INTO worker_execution_commit_authority_loss_decisions
        (tenant_id, operation_id, worker_instance_id, ownership_epoch,
         decision_id, decision_authority_id, bound_store_revision,
         loss_reason, decided_at_utc, valid_until_utc, decision_digest)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision.tenant_id,
            decision.operation_id,
            decision.worker_instance_id,
            decision.ownership_epoch,
            decision.decision_id,
            decision.decision_authority_id,
            decision.bound_store_revision,
            decision.loss_reason.value,
            decision.decided_at.isoformat(),
            decision.valid_until.isoformat(),
            decision.decision_digest,
        ),
    )


def _decode_commit_authority_loss_decision(
    row: sqlite3.Row,
) -> WorkerExecutionCommitAuthorityLossDecisionV1:
    from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason

    return WorkerExecutionCommitAuthorityLossDecisionV1(
        decision_id=row["decision_id"],
        decision_authority_id=row["decision_authority_id"],
        tenant_id=row["tenant_id"],
        operation_id=row["operation_id"],
        worker_instance_id=row["worker_instance_id"],
        ownership_epoch=int(row["ownership_epoch"]),
        bound_store_revision=int(row["bound_store_revision"]),
        loss_reason=CertifiedOwnerLossReason(row["loss_reason"]),
        decided_at=datetime.fromisoformat(row["decided_at_utc"]),
        valid_until=datetime.fromisoformat(row["valid_until_utc"]),
        decision_digest=row["decision_digest"],
    )


def _encode_external_operation_state(state: ExternalOperationState) -> str:
    staged_payload = None
    if state.staged_certified_owner_loss is not None:
        staged_payload = state.staged_certified_owner_loss.model_dump(mode="json")
    payload = {
        "operation_id": state.operation_id,
        "tenant_id": state.tenant_id,
        "intent_state": state.intent_state.value,
        "physical_state": state.physical_state.value,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "revision": state.revision,
        "owner_token": state.owner_token,
        "ownership_epoch_fence": state.ownership_epoch_fence,
        "staged_certified_owner_loss": staged_payload,
        "consumed_certified_loss_evidence_ids": sorted(
            state.consumed_certified_loss_evidence_ids,
        ),
        "worker_instance_id": state.worker_instance_id,
        "worker_ownership_epoch": state.worker_ownership_epoch,
        "consumed_redrive_admission_ids": sorted(
            state.consumed_redrive_admission_ids,
        ),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _decode_external_operation_state(raw: str) -> ExternalOperationState:
    payload = json.loads(raw)
    staged = payload.get("staged_certified_owner_loss")
    staged_model = (
        StagedCertifiedOwnerLossEvidenceV1.model_validate(staged) if staged else None
    )
    return ExternalOperationState(
        operation_id=payload["operation_id"],
        tenant_id=payload["tenant_id"],
        intent_state=ExternalOperationIntentState(payload["intent_state"]),
        physical_state=ExternalOperationPhysicalState(payload["physical_state"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
        revision=int(payload["revision"]),
        owner_token=payload.get("owner_token"),
        ownership_epoch_fence=payload.get("ownership_epoch_fence"),
        staged_certified_owner_loss=staged_model,
        consumed_certified_loss_evidence_ids=frozenset(
            payload.get("consumed_certified_loss_evidence_ids") or [],
        ),
        worker_instance_id=payload.get("worker_instance_id"),
        worker_ownership_epoch=payload.get("worker_ownership_epoch"),
        consumed_redrive_admission_ids=frozenset(
            payload.get("consumed_redrive_admission_ids") or [],
        ),
    )


class SQLiteExternalOperationStateStore(_ExternalOperationStateCertificationSemantics):
    """File-backed external operation store — ``DURABLE_SINGLE_HOST`` topology."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with _connect(self._db_path) as conn:
            _ensure_tool_life_schema(conn)
            conn.commit()

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.DURABLE_SINGLE_HOST

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _get_scoped(
        self,
        tenant_id: str,
        operation_id: str,
    ) -> ExternalOperationState | None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        tid = _validate_tenant_id(tenant_id)
        key = _store_key(tid, operation_id)
        with self._lock:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    """
                    SELECT state_json FROM external_operation_states
                    WHERE tenant_id = ? AND operation_id = ?
                    """,
                    key,
                ).fetchone()
                if row is None:
                    return None
                record = _decode_external_operation_state(row["state_json"])
                _require_tenant_bound(record)
                if record.tenant_id != key[0]:
                    return None
                return record

    def create_if_absent(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        owner_token: str | None = None,
    ) -> ExternalOperationState:
        tid = _validate_tenant_id(tenant_id)
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        now = _utc_now()
        key = (tid, operation_id)
        with self._lock:
            with _connect(self._db_path) as conn:
                _ensure_tool_life_schema(conn)
                row = conn.execute(
                    """
                    SELECT state_json FROM external_operation_states
                    WHERE tenant_id = ? AND operation_id = ?
                    """,
                    key,
                ).fetchone()
                if row is not None:
                    existing = _decode_external_operation_state(row["state_json"])
                    _require_tenant_bound(existing)
                    return existing
                record = ExternalOperationState(
                    operation_id=operation_id,
                    tenant_id=tid,
                    intent_state=ExternalOperationIntentState.ACTIVE,
                    physical_state=ExternalOperationPhysicalState.NOT_STARTED,
                    created_at=now,
                    updated_at=now,
                    revision=1,
                    owner_token=owner_token,
                )
                conn.execute(
                    """
                    INSERT INTO external_operation_states
                    (tenant_id, operation_id, revision, state_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        tid,
                        operation_id,
                        record.revision,
                        _encode_external_operation_state(record),
                    ),
                )
                conn.commit()
                return record

    def _compare_and_set_locked(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        update: ExternalOperationStateUpdate,
    ) -> ExternalOperationState:
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be int >= 1")
        tid = _validate_tenant_id(tenant_id)
        now = _utc_now()
        key = (tid, operation_id)
        with self._lock:
            with _connect(self._db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT revision, state_json FROM external_operation_states
                    WHERE tenant_id = ? AND operation_id = ?
                    """,
                    key,
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise ExternalOperationNotFoundError(operation_id)
                current = _decode_external_operation_state(row["state_json"])
                _require_tenant_bound(current)
                if current.tenant_id != tid:
                    conn.execute("ROLLBACK")
                    raise ExternalOperationTenantScopeViolationError(
                        f"tenant scope mismatch for {operation_id!r}",
                    )
                if current.revision != expected_revision:
                    conn.execute("ROLLBACK")
                    raise StaleExternalOperationStateError(
                        operation_id=operation_id,
                        expected_revision=expected_revision,
                        actual_revision=current.revision,
                    )
                if (
                    current.worker_instance_id is not None
                    and current.worker_ownership_epoch is not None
                ):
                    revoked_row = conn.execute(
                        """
                        SELECT 1 FROM worker_execution_ownership_revocations
                        WHERE tenant_id = ? AND operation_id = ?
                          AND worker_instance_id = ? AND ownership_epoch = ?
                        """,
                        (
                            tid,
                            operation_id,
                            current.worker_instance_id,
                            current.worker_ownership_epoch,
                        ),
                    ).fetchone()
                    if cas_update_blocked_by_bound_worker_revocation(
                        current=current,
                        update=update,
                        epoch_is_revoked=revoked_row is not None,
                    ):
                        conn.execute("ROLLBACK")
                        raise CertifiedOwnerLossEvidenceRejectedError(
                            f"revoked worker execution epoch cannot mutate "
                            f"{operation_id!r} under transaction",
                        )
                staged_slot = update.staged_certified_owner_loss
                if (
                    staged_slot is not None
                    and staged_slot.staged_by_issuer_id
                    == CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID
                    and current.worker_instance_id is not None
                    and current.worker_ownership_epoch is not None
                ):
                    proof_row = conn.execute(
                        """
                        SELECT proof_ref FROM worker_execution_ownership_revocations
                        WHERE tenant_id = ? AND operation_id = ?
                          AND worker_instance_id = ? AND ownership_epoch = ?
                        """,
                        (
                            tid,
                            operation_id,
                            current.worker_instance_id,
                            current.worker_ownership_epoch,
                        ),
                    ).fetchone()
                    expected_proof = (
                        proof_row["proof_ref"] if proof_row is not None else None
                    )
                    cited = staged_slot.evidence.liveness_proof_ref
                    if expected_proof is None or cited != expected_proof:
                        conn.execute("ROLLBACK")
                        raise CertifiedOwnerLossEvidenceRejectedError(
                            "liveness proof does not match authoritative revocation "
                            "record under transaction",
                        )
                next_record = apply_external_operation_state_update(
                    current,
                    update,
                    now=now,
                )
                updated = conn.execute(
                    """
                    UPDATE external_operation_states
                    SET revision = ?, state_json = ?
                    WHERE tenant_id = ? AND operation_id = ? AND revision = ?
                    """,
                    (
                        next_record.revision,
                        _encode_external_operation_state(next_record),
                        tid,
                        operation_id,
                        expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    conn.execute("ROLLBACK")
                    raise StaleExternalOperationStateError(
                        operation_id=operation_id,
                        expected_revision=expected_revision,
                        actual_revision=current.revision,
                    )
                conn.commit()
                return next_record

    def list_running(self, tenant_id: str) -> tuple[ExternalOperationState, ...]:
        tid = _validate_tenant_id(tenant_id)
        with self._lock:
            with _connect(self._db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT state_json FROM external_operation_states
                    WHERE tenant_id = ?
                    """,
                    (tid,),
                ).fetchall()
        running: list[ExternalOperationState] = []
        for row in rows:
            record = _decode_external_operation_state(row["state_json"])
            if record.physical_state is ExternalOperationPhysicalState.RUNNING:
                running.append(record)
        return tuple(running)


class SQLiteWorkerExecutionOwnershipRevocationStore(
    WorkerExecutionOwnershipRevocationPort,
    WorkerExecutionCommitAuthorityLossDecisionPort,
):
    """Durable revocation ledger colocated with external-operation SQLite file."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with _connect(self._db_path) as conn:
            _ensure_tool_life_schema(conn)
            conn.commit()

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.DURABLE_SINGLE_HOST

    @property
    def authoritative_state_store(self) -> SQLiteExternalOperationStateStore:
        return SQLiteExternalOperationStateStore(db_path=self._db_path)

    def authoritative_commit_authority_loss_decision(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1 | None:
        key = (tenant_id, operation_id, worker_instance_id, ownership_epoch)
        with self._lock:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    """
                    SELECT * FROM worker_execution_commit_authority_loss_decisions
                    WHERE tenant_id = ? AND operation_id = ?
                      AND worker_instance_id = ? AND ownership_epoch = ?
                    """,
                    key,
                ).fetchone()
                return _decode_commit_authority_loss_decision(row) if row is not None else None

    def record_commit_authority_loss_decision(
        self,
        *,
        decision: WorkerExecutionCommitAuthorityLossDecisionV1,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1:
        _require_decision_authority_invocation()
        key = (
            decision.tenant_id,
            decision.operation_id,
            decision.worker_instance_id,
            decision.ownership_epoch,
        )
        with self._lock:
            with _connect(self._db_path) as conn:
                _ensure_tool_life_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """
                    SELECT * FROM worker_execution_commit_authority_loss_decisions
                    WHERE tenant_id = ? AND operation_id = ?
                      AND worker_instance_id = ? AND ownership_epoch = ?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return _decode_commit_authority_loss_decision(existing)
                _insert_commit_authority_loss_decision_row(conn, decision=decision)
                conn.commit()
                return decision

    def authoritative_revocation_proof_ref(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> str | None:
        key = (tenant_id, operation_id, worker_instance_id, ownership_epoch)
        with self._lock:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    """
                    SELECT proof_ref FROM worker_execution_ownership_revocations
                    WHERE tenant_id = ? AND operation_id = ?
                      AND worker_instance_id = ? AND ownership_epoch = ?
                    """,
                    key,
                ).fetchone()
                return row["proof_ref"] if row is not None else None

    def revoke_worker_execution_ownership(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        authorization: WorkerExecutionOwnershipRevocationAuthorizationV1,
    ) -> WorkerExecutionOwnershipRevocationV1:
        _require_recorder_revocation_invocation()
        composition = require_certified_owner_loss_composition()
        if composition.worker_execution_revocation_supervision is None:
            raise WorkerExecutionDirectRevocationInvocationError(
                "revocation supervision is not configured",
            )
        if (
            authorization.supervision_authority_id
            != CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID
        ):
            raise WorkerExecutionDirectRevocationInvocationError(
                "revocation supervision authority mismatch",
            )
        key = (tenant_id, operation_id, worker_instance_id, ownership_epoch)
        tid = _validate_tenant_id(tenant_id)
        with self._lock:
            with _connect(self._db_path) as conn:
                _ensure_tool_life_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """
                    SELECT revocation_id, revoked_at_utc, proof_ref
                    FROM worker_execution_ownership_revocations
                    WHERE tenant_id = ? AND operation_id = ?
                      AND worker_instance_id = ? AND ownership_epoch = ?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return WorkerExecutionOwnershipRevocationV1(
                        tenant_id=tenant_id,
                        operation_id=operation_id,
                        worker_instance_id=worker_instance_id,
                        ownership_epoch=ownership_epoch,
                        revocation_id=existing["revocation_id"],
                        revoked_at=datetime.fromisoformat(existing["revoked_at_utc"]),
                        proof_ref=existing["proof_ref"],
                    )
                state_row = conn.execute(
                    """
                    SELECT state_json FROM external_operation_states
                    WHERE tenant_id = ? AND operation_id = ?
                    """,
                    (tid, operation_id),
                ).fetchone()
                if state_row is None:
                    conn.execute("ROLLBACK")
                    raise ExternalOperationNotFoundError(operation_id)
                current = _decode_external_operation_state(state_row["state_json"])
                _require_tenant_bound(current)
                try:
                    assert_authoritative_worker_execution_revocation_allowed(
                        current,
                        tenant_id=tenant_id,
                        operation_id=operation_id,
                        worker_instance_id=worker_instance_id,
                        ownership_epoch=ownership_epoch,
                    )
                    verify_worker_execution_ownership_revocation_authorization(
                        authorization,
                        tenant_id=tenant_id,
                        operation_id=operation_id,
                        worker_instance_id=worker_instance_id,
                        ownership_epoch=ownership_epoch,
                        bound_store_revision=current.revision,
                    )
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                revocation_id = uuid.uuid4().hex
                proof = worker_execution_revocation_proof_ref(
                    tenant_id=tenant_id,
                    operation_id=operation_id,
                    worker_instance_id=worker_instance_id,
                    ownership_epoch=ownership_epoch,
                    revocation_id=revocation_id,
                )
                revoked_at = _utc_now()
                conn.execute(
                    """
                    INSERT INTO worker_execution_ownership_revocations
                    (tenant_id, operation_id, worker_instance_id, ownership_epoch,
                     revocation_id, revoked_at_utc, proof_ref)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        operation_id,
                        worker_instance_id,
                        ownership_epoch,
                        revocation_id,
                        revoked_at.isoformat(),
                        proof,
                    ),
                )
                decision_row = conn.execute(
                    """
                    SELECT 1 FROM worker_execution_commit_authority_loss_decisions
                    WHERE tenant_id = ? AND operation_id = ?
                      AND worker_instance_id = ? AND ownership_epoch = ?
                    """,
                    key,
                ).fetchone()
                if decision_row is None:
                    decision = mint_worker_execution_commit_authority_loss_decision(
                        tenant_id=tenant_id,
                        operation_id=operation_id,
                        worker_instance_id=worker_instance_id,
                        ownership_epoch=ownership_epoch,
                        bound_store_revision=current.revision,
                        loss_reason=authorization.loss_reason,
                    )
                    verify_worker_execution_commit_authority_loss_decision(
                        decision,
                        tenant_id=tenant_id,
                        operation_id=operation_id,
                        worker_instance_id=worker_instance_id,
                        ownership_epoch=ownership_epoch,
                        bound_store_revision=current.revision,
                        loss_reason=authorization.loss_reason,
                    )
                    _insert_commit_authority_loss_decision_row(conn, decision=decision)
                conn.commit()
                return WorkerExecutionOwnershipRevocationV1(
                    tenant_id=tenant_id,
                    operation_id=operation_id,
                    worker_instance_id=worker_instance_id,
                    ownership_epoch=ownership_epoch,
                    revocation_id=revocation_id,
                    revoked_at=revoked_at,
                    proof_ref=proof,
                )


class SqliteToolLifeDurableStoresBundle:
    """Single SQLite file hosting external-operation state and revocation ledger."""

    def __init__(self, *, db_path: Path) -> None:
        self.db_path = db_path
        self.external_operation_state = SQLiteExternalOperationStateStore(
            db_path=db_path,
        )
        self.worker_execution_ownership_revocation = (
            SQLiteWorkerExecutionOwnershipRevocationStore(db_path=db_path)
        )

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.DURABLE_SINGLE_HOST


__all__ = [
    "SqliteToolLifeDurableStoresBundle",
    "SQLiteExternalOperationStateStore",
    "SQLiteWorkerExecutionOwnershipRevocationStore",
]
