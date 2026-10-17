# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Transaction-coordinated serving execution admission (PROD-Q-R1-R7-D1-R6-R2)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.errors import (
    AgentDistributionError,
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
    run_under_activation_domain,
    _application_environment_scope,
    _deployment_instance_key,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.sqlite_stores import (
    SqliteAgentDistributionDatabase,
    SqliteAgentDistributionStoreBundle,
    SqliteApplicationEnvironmentServingStore,
    SqliteDeploymentInstanceStore,
    SqliteRuntimeRevisionStore,
)
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    RuntimeRevisionStore,
)

_NON_EMPTY = Field(min_length=1)
_SQLITE_BUSY_MAX_ATTEMPTS = 5
_SQLITE_BUSY_BASE_DELAY_S = 0.01


class ActivationDomainCoordinatorError(AgentDistributionError):
    """Activation domain coordination failed."""


class ServingExecutionAdmissionRequest(BaseModel):
    """Immutable requested serving identity for one execution admission decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    serving_unit_ref: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY


class ServingExecutionEligibilityEvidence(BaseModel):
    """Non-authoritative eligibility snapshot for an already-coordinated admission decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    serving_unit_ref: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    serving_pointer_revision: int = Field(ge=0)


class PhysicalExecutionReservation(Protocol):
    """Adapter-owned physical in-flight slot; release exactly once after coordination."""

    def release(self) -> None: ...


class CoordinatorVerifiedUnitServingAdmissionOpen(Protocol):
    """Unit-bound physical admission open invoked synchronously by the coordinator."""

    def apply_coordinator_verified_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Stage coordinator-verified admission; gate stays closed until publish."""

    def publish_coordinator_verified_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Publish staged admission after canonical coordination succeeded."""

    def compensate_failed_coordination_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Revert staged or published admission after canonical persistence failed."""

    def reconcile_coordinator_verified_existing_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Confirm physical admission already matches coordinator-verified evidence (lifecycle reopen)."""


def _eligibility_evidence_from_request(
    *,
    request: ServingExecutionAdmissionRequest,
    serving_pointer_revision: int,
) -> ServingExecutionEligibilityEvidence:
    return ServingExecutionEligibilityEvidence(
        application_id=request.application_id,
        application_environment_id=request.application_environment_id,
        runtime_revision_id=request.runtime_revision_id,
        serving_unit_ref=request.serving_unit_ref,
        materialization_artifact_digest=request.materialization_artifact_digest,
        serving_pointer_revision=serving_pointer_revision,
    )


def verify_serving_execution_eligibility_from_state(
    state: AgentDistributionStoreState,
    request: ServingExecutionAdmissionRequest,
) -> ServingExecutionEligibilityEvidence:
    """Fail closed unless canonical state permits execution admission now."""
    scope = _application_environment_scope(
        request.application_id,
        request.application_environment_id,
    )
    serving = state.serving_records.get(scope)
    if serving is None:
        raise ServingExecutionAdmissionError("canonical serving record is missing")
    if serving.application_id != request.application_id:
        raise ServingExecutionAdmissionError("serving record application mismatch")
    if serving.application_environment_id != request.application_environment_id:
        raise ServingExecutionAdmissionError("serving record environment mismatch")
    if serving.traffic_serving_revision_id != request.runtime_revision_id:
        raise ServingExecutionAdmissionError(
            "canonical serving pointer does not target requested runtime revision"
        )

    revision = state.revisions.get(request.runtime_revision_id)
    if revision is None:
        raise ServingExecutionAdmissionError("runtime revision was not found")
    if revision.application_id != request.application_id:
        raise ServingExecutionAdmissionError("runtime revision application mismatch")
    if revision.application_environment_id != request.application_environment_id:
        raise ServingExecutionAdmissionError("runtime revision environment mismatch")
    if revision.revision_state is not RuntimeRevisionState.ACTIVE:
        raise ServingExecutionAdmissionError(
            "runtime revision is not active for execution admission"
        )
    if revision.materialization_artifact_digest != request.materialization_artifact_digest:
        raise ServingExecutionAdmissionError(
            "runtime revision artifact digest mismatch for execution admission"
        )

    instance_key = _deployment_instance_key(
        request.application_id,
        request.application_environment_id,
        request.runtime_revision_id,
    )
    instance = state.deployment_instances.get(instance_key)
    if instance is None:
        raise ServingExecutionAdmissionError("canonical deployment instance is missing")
    if instance.instance_state is not DeploymentInstanceState.SERVING:
        raise ServingExecutionAdmissionError(
            "deployment instance is not in serving state for execution admission"
        )
    if instance.serving_unit_ref != request.serving_unit_ref:
        raise ServingExecutionAdmissionError(
            "deployment instance serving unit does not match requested unit"
        )

    return _eligibility_evidence_from_request(
        request=request,
        serving_pointer_revision=serving.serving_pointer_revision,
    )


def _sqlite_busy_operational_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _sqlite_connection_cleanup_errors(
    conn: sqlite3.Connection,
    *,
    rollback: bool,
) -> list[sqlite3.Error]:
    errors: list[sqlite3.Error] = []
    if rollback:
        try:
            conn.rollback()
        except sqlite3.Error as exc:
            errors.append(exc)
    try:
        conn.close()
    except sqlite3.Error as exc:
        errors.append(exc)
    return errors


def _release_sqlite_activation_domain_lock(db: SqliteAgentDistributionDatabase) -> None:
    db._release_activation_domain_lock()


def _raise_sqlite_admission_coordination_failure(
    primary: BaseException,
    cleanup_errors: list[sqlite3.Error],
    *,
    message: str = "sqlite admission coordination failed",
) -> None:
    if cleanup_errors:
        raised = ServingExecutionAdmissionError(f"{message}; acquisition cleanup also failed")
        raised.__cause__ = primary
        raised.__context__ = cleanup_errors[0]
        if len(cleanup_errors) > 1:
            for extra in cleanup_errors[1:]:
                raised.add_note(f"additional cleanup error: {extra}")
        raise raised
    if isinstance(primary, ServingExecutionAdmissionError):
        raise primary
    raise ServingExecutionAdmissionError(message) from primary


def _cleanup_failed_sqlite_admission_acquisition(
    db: SqliteAgentDistributionDatabase,
    conn: sqlite3.Connection | None,
) -> list[sqlite3.Error]:
    cleanup_errors: list[sqlite3.Error] = []
    if conn is not None:
        cleanup_errors.extend(_sqlite_connection_cleanup_errors(conn, rollback=True))
    try:
        _release_sqlite_activation_domain_lock(db)
    except RuntimeError as lock_exc:
        cleanup_errors.append(sqlite3.OperationalError(str(lock_exc)))
    return cleanup_errors


class _SqliteAdmissionDomainLease:
    """One operation-owned activation-domain lock + one IMMEDIATE SQLite transaction."""

    __slots__ = (
        "_conn",
        "_conn_closed",
        "_db",
        "_domain_lock_released",
        "_durable_committed",
        "_state",
        "_transaction_active",
    )

    def __init__(
        self,
        db: SqliteAgentDistributionDatabase,
        conn: sqlite3.Connection,
        state: AgentDistributionStoreState,
    ) -> None:
        self._db = db
        self._conn = conn
        self._state = state
        self._domain_lock_released = False
        self._conn_closed = False
        self._durable_committed = False
        self._transaction_active = True

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    @property
    def durable_committed(self) -> bool:
        return self._durable_committed

    def commit_durable(self) -> None:
        if self._durable_committed:
            raise ServingExecutionAdmissionError(
                "sqlite admission coordination commit invoked more than once"
            )
        commit_error: sqlite3.Error | None = None
        try:
            self._conn.commit()
        except sqlite3.Error as exc:
            commit_error = exc
        else:
            self._durable_committed = True
            self._transaction_active = False
            return

        rollback_error: sqlite3.Error | None = None
        try:
            self._conn.rollback()
            self._transaction_active = False
        except sqlite3.Error as exc:
            rollback_error = exc
            self._transaction_active = True
        if rollback_error is not None:
            raised = ServingExecutionAdmissionError(
                "sqlite admission coordination commit failed; rollback also failed; "
                "transaction outcome uncertain"
            )
            raised.__cause__ = commit_error
            raised.__context__ = rollback_error
            raise raised
        raise ServingExecutionAdmissionError(
            "sqlite admission coordination commit failed"
        ) from commit_error

    def rollback(self) -> None:
        if not self._transaction_active or self._durable_committed:
            return
        self._conn.rollback()
        self._transaction_active = False

    def close(self) -> None:
        if self._conn_closed:
            return
        self._conn_closed = True
        close_error: sqlite3.Error | None = None
        try:
            self._conn.close()
        except sqlite3.Error as exc:
            close_error = exc
        finally:
            if not self._domain_lock_released:
                _release_sqlite_activation_domain_lock(self._db)
                self._domain_lock_released = True
        if close_error is not None:
            raise ServingExecutionAdmissionError(
                "sqlite admission coordination connection close failed"
            ) from close_error


def _open_sqlite_admission_domain_lease(
    db: SqliteAgentDistributionDatabase,
) -> _SqliteAdmissionDomainLease:
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(1, _SQLITE_BUSY_MAX_ATTEMPTS + 1):
        db._acquire_activation_domain_lock()
        conn: sqlite3.Connection | None = None
        try:
            conn = db._connect()
            conn.execute("BEGIN IMMEDIATE")
            state = db._load_state(conn)
            return _SqliteAdmissionDomainLease(db, conn, state)
        except sqlite3.OperationalError as exc:
            cleanup_errors = _cleanup_failed_sqlite_admission_acquisition(db, conn)
            if cleanup_errors:
                _raise_sqlite_admission_coordination_failure(exc, cleanup_errors)
            if _sqlite_busy_operational_error(exc) and attempt < _SQLITE_BUSY_MAX_ATTEMPTS:
                last_error = exc
                time.sleep(_SQLITE_BUSY_BASE_DELAY_S * attempt)
                continue
            raise ServingExecutionAdmissionError(
                "sqlite admission coordination failed"
            ) from exc
        except Exception as exc:
            cleanup_errors = _cleanup_failed_sqlite_admission_acquisition(db, conn)
            if cleanup_errors:
                _raise_sqlite_admission_coordination_failure(exc, cleanup_errors)
            raise
    if last_error is not None:
        raise ServingExecutionAdmissionError(
            "sqlite admission coordination exceeded busy retry budget"
        ) from last_error
    raise ServingExecutionAdmissionError("sqlite admission coordination failed")


@contextmanager
def _sqlite_admission_domain_lease(db: SqliteAgentDistributionDatabase):
    """Lexical owner for one SQLite admission coordination operation."""
    lease = _open_sqlite_admission_domain_lease(db)
    try:
        yield lease
    except Exception:
        lease.rollback()
        raise
    finally:
        if not lease.durable_committed:
            try:
                lease.rollback()
            except sqlite3.Error:
                pass
        lease.close()


@contextmanager
def _sqlite_admission_coordination(db: SqliteAgentDistributionDatabase):
    with _sqlite_admission_domain_lease(db) as lease:
        yield lease.state
        lease.commit_durable()


def _release_physical_reservation_once(
    reservation: PhysicalExecutionReservation | None,
) -> None:
    if reservation is None:
        return
    reservation.release()


def _compensate_unit_open_after_coordination_failure(
    unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
    evidence: ServingExecutionEligibilityEvidence,
    *,
    physical_admission_affected: bool,
) -> None:
    if not physical_admission_affected:
        return
    unit_admission_open.compensate_failed_coordination_open(evidence)


class ActivationDomainCoordinator(Protocol):
    """Provider-neutral boundary between canonical serving state and physical reservation."""

    def coordinate_physical_execution_reservation(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        physical_reserve: Callable[[], PhysicalExecutionReservation],
    ) -> ServingExecutionEligibilityEvidence: ...

    def coordinate_unit_serving_admission_open(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
        expected_serving_pointer_revision: int | None = None,
    ) -> ServingExecutionEligibilityEvidence: ...

    def coordinate_unit_serving_admission_reopen(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
        expected_serving_pointer_revision: int | None = None,
    ) -> ServingExecutionEligibilityEvidence: ...


def _verify_unit_admission_evidence(
    state: AgentDistributionStoreState,
    request: ServingExecutionAdmissionRequest,
    *,
    expected_serving_pointer_revision: int | None,
) -> ServingExecutionEligibilityEvidence:
    evidence = verify_serving_execution_eligibility_from_state(state, request)
    if expected_serving_pointer_revision is not None:
        if evidence.serving_pointer_revision != expected_serving_pointer_revision:
            raise ServingExecutionAdmissionError(
                "serving pointer revision mismatch for unit admission open"
            )
    return evidence


class InMemoryActivationDomainCoordinator:
    """Shared in-memory Agent Distribution activation domain coordinator."""

    def __init__(self, state: AgentDistributionStoreState) -> None:
        self._state = state

    def coordinate_physical_execution_reservation(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        physical_reserve: Callable[[], PhysicalExecutionReservation],
    ) -> ServingExecutionEligibilityEvidence:
        reservation: PhysicalExecutionReservation | None = None
        evidence: ServingExecutionEligibilityEvidence | None = None

        def _work() -> ServingExecutionEligibilityEvidence:
            nonlocal reservation, evidence
            evidence = verify_serving_execution_eligibility_from_state(
                self._state,
                request,
            )
            reservation = physical_reserve()
            return evidence

        try:
            return run_under_activation_domain(self._state, _work)
        except Exception:
            _release_physical_reservation_once(reservation)
            raise

    def coordinate_unit_serving_admission_open(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
        expected_serving_pointer_revision: int | None = None,
    ) -> ServingExecutionEligibilityEvidence:
        physical_admission_affected = False
        evidence: ServingExecutionEligibilityEvidence | None = None

        def _work() -> ServingExecutionEligibilityEvidence:
            nonlocal physical_admission_affected, evidence
            evidence = _verify_unit_admission_evidence(
                self._state,
                request,
                expected_serving_pointer_revision=expected_serving_pointer_revision,
            )
            unit_admission_open.apply_coordinator_verified_open(evidence)
            physical_admission_affected = True
            unit_admission_open.publish_coordinator_verified_open(evidence)
            return evidence

        with self._state._activation_lock:
            try:
                return _work()
            except Exception as primary:
                if evidence is not None:
                    try:
                        _compensate_unit_open_after_coordination_failure(
                            unit_admission_open,
                            evidence,
                            physical_admission_affected=physical_admission_affected,
                        )
                    except Exception as compensation_error:
                        raise compensation_error from primary
                raise

    def coordinate_unit_serving_admission_reopen(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
        expected_serving_pointer_revision: int | None = None,
    ) -> ServingExecutionEligibilityEvidence:
        def _work() -> ServingExecutionEligibilityEvidence:
            evidence = _verify_unit_admission_evidence(
                self._state,
                request,
                expected_serving_pointer_revision=expected_serving_pointer_revision,
            )
            unit_admission_open.reconcile_coordinator_verified_existing_open(evidence)
            return evidence

        with self._state._activation_lock:
            return _work()


class SqliteActivationDomainCoordinator:
    """SQLite-backed activation domain coordinator (one IMMEDIATE snapshot per admission)."""

    def __init__(self, database: SqliteAgentDistributionDatabase) -> None:
        self._database = database

    def coordinate_physical_execution_reservation(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        physical_reserve: Callable[[], PhysicalExecutionReservation],
    ) -> ServingExecutionEligibilityEvidence:
        reservation: PhysicalExecutionReservation | None = None
        evidence: ServingExecutionEligibilityEvidence | None = None
        try:
            with _sqlite_admission_coordination(self._database) as state:
                evidence = verify_serving_execution_eligibility_from_state(state, request)
                reservation = physical_reserve()
            assert evidence is not None
            return evidence
        except Exception:
            _release_physical_reservation_once(reservation)
            raise

    def coordinate_unit_serving_admission_open(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
        expected_serving_pointer_revision: int | None = None,
    ) -> ServingExecutionEligibilityEvidence:
        physical_admission_affected = False
        evidence: ServingExecutionEligibilityEvidence | None = None
        with _sqlite_admission_domain_lease(self._database) as lease:
            try:
                evidence = _verify_unit_admission_evidence(
                    lease.state,
                    request,
                    expected_serving_pointer_revision=expected_serving_pointer_revision,
                )
                unit_admission_open.apply_coordinator_verified_open(evidence)
                physical_admission_affected = True
                lease.commit_durable()
                unit_admission_open.publish_coordinator_verified_open(evidence)
            except Exception as primary:
                if evidence is not None:
                    try:
                        _compensate_unit_open_after_coordination_failure(
                            unit_admission_open,
                            evidence,
                            physical_admission_affected=physical_admission_affected,
                        )
                    except Exception as compensation_error:
                        raise compensation_error from primary
                raise
        assert evidence is not None
        return evidence

    def coordinate_unit_serving_admission_reopen(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
        expected_serving_pointer_revision: int | None = None,
    ) -> ServingExecutionEligibilityEvidence:
        with _sqlite_admission_domain_lease(self._database) as lease:
            evidence = _verify_unit_admission_evidence(
                lease.state,
                request,
                expected_serving_pointer_revision=expected_serving_pointer_revision,
            )
            unit_admission_open.reconcile_coordinator_verified_existing_open(evidence)
        return evidence


def _shared_in_memory_state_from_stores(
    serving_store: InMemoryApplicationEnvironmentServingStore,
    deployment_instance_store: InMemoryDeploymentInstanceStore,
    revision_store: InMemoryRuntimeRevisionStore,
) -> AgentDistributionStoreState:
    state = serving_store.state
    if deployment_instance_store.state is not state:
        raise ActivationDomainCoordinatorError(
            "deployment instance store does not share serving store state"
        )
    if revision_store.state is not state:
        raise ActivationDomainCoordinatorError(
            "revision store does not share serving store state"
        )
    return state


def _shared_sqlite_database_from_stores(
    serving_store: SqliteApplicationEnvironmentServingStore,
    deployment_instance_store: SqliteDeploymentInstanceStore,
    revision_store: SqliteRuntimeRevisionStore,
) -> SqliteAgentDistributionDatabase:
    database = serving_store._db
    if deployment_instance_store._db is not database:
        raise ActivationDomainCoordinatorError(
            "deployment instance store does not share sqlite database"
        )
    if revision_store._db is not database:
        raise ActivationDomainCoordinatorError(
            "revision store does not share sqlite database"
        )
    return database


def build_activation_domain_coordinator_for_stores(
    *,
    serving_store: ApplicationEnvironmentServingStore,
    deployment_instance_store: DeploymentInstanceStore,
    revision_store: RuntimeRevisionStore,
) -> ActivationDomainCoordinator:
    """Construct coordinator backend matching the sanctioned store bundle topology."""
    if isinstance(serving_store, SqliteApplicationEnvironmentServingStore):
        if not isinstance(deployment_instance_store, SqliteDeploymentInstanceStore):
            raise ActivationDomainCoordinatorError(
                "sqlite stores must share one SqliteAgentDistributionDatabase"
            )
        if not isinstance(revision_store, SqliteRuntimeRevisionStore):
            raise ActivationDomainCoordinatorError(
                "sqlite stores must share one SqliteAgentDistributionDatabase"
            )
        database = _shared_sqlite_database_from_stores(
            serving_store,
            deployment_instance_store,
            revision_store,
        )
        return SqliteActivationDomainCoordinator(database)
    if isinstance(serving_store, InMemoryApplicationEnvironmentServingStore):
        if not isinstance(deployment_instance_store, InMemoryDeploymentInstanceStore):
            raise ActivationDomainCoordinatorError(
                "in-memory stores must share one AgentDistributionStoreState bundle"
            )
        if not isinstance(revision_store, InMemoryRuntimeRevisionStore):
            raise ActivationDomainCoordinatorError(
                "in-memory stores must share one AgentDistributionStoreState bundle"
            )
        state = _shared_in_memory_state_from_stores(
            serving_store,
            deployment_instance_store,
            revision_store,
        )
        return InMemoryActivationDomainCoordinator(state)
    raise ActivationDomainCoordinatorError(
        "unsupported serving store for activation domain coordinator"
    )


def build_activation_domain_coordinator_for_sqlite_bundle(
    bundle: SqliteAgentDistributionStoreBundle,
) -> ActivationDomainCoordinator:
    """Composition-owned SQLite bundle → activation domain coordinator."""
    return build_activation_domain_coordinator_for_stores(
        serving_store=bundle.serving_store,
        deployment_instance_store=bundle.deployment_instance_store,
        revision_store=bundle.revision_store,
    )


__all__ = [
    "ActivationDomainCoordinator",
    "ActivationDomainCoordinatorError",
    "CoordinatorVerifiedUnitServingAdmissionOpen",
    "InMemoryActivationDomainCoordinator",
    "PhysicalExecutionReservation",
    "ServingExecutionAdmissionRequest",
    "ServingExecutionEligibilityEvidence",
    "SqliteActivationDomainCoordinator",
    "build_activation_domain_coordinator_for_sqlite_bundle",
    "build_activation_domain_coordinator_for_stores",
    "verify_serving_execution_eligibility_from_state",
]
