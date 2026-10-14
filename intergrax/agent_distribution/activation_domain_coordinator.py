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
        """Open the bound unit for execution admission; evidence must match bound identity."""

    def compensate_failed_coordination_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Revert a coordination-time open after canonical persistence failed."""


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


def _acquire_sqlite_admission_transaction(
    db: SqliteAgentDistributionDatabase,
) -> tuple[sqlite3.Connection, AgentDistributionStoreState]:
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(1, _SQLITE_BUSY_MAX_ATTEMPTS + 1):
        with db._activation_lock:
            conn = db._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                state = db._load_state(conn)
                return conn, state
            except sqlite3.OperationalError as exc:
                conn.rollback()
                conn.close()
                if _sqlite_busy_operational_error(exc) and attempt < _SQLITE_BUSY_MAX_ATTEMPTS:
                    last_error = exc
                    time.sleep(_SQLITE_BUSY_BASE_DELAY_S * attempt)
                    continue
                raise ServingExecutionAdmissionError(
                    "sqlite admission coordination failed"
                ) from exc
            except Exception:
                conn.rollback()
                conn.close()
                raise
    if last_error is not None:
        raise ServingExecutionAdmissionError(
            "sqlite admission coordination exceeded busy retry budget"
        ) from last_error
    raise ServingExecutionAdmissionError("sqlite admission coordination failed")


@contextmanager
def _sqlite_admission_coordination(db: SqliteAgentDistributionDatabase):
    conn, state = _acquire_sqlite_admission_transaction(db)
    try:
        yield state
    except Exception:
        conn.rollback()
        raise
    else:
        try:
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.rollback()
            raise ServingExecutionAdmissionError(
                "sqlite admission coordination commit failed"
            ) from exc
    finally:
        conn.close()


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
    physical_opened: bool,
) -> None:
    if not physical_opened:
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
        physical_opened = False
        evidence: ServingExecutionEligibilityEvidence | None = None

        def _work() -> ServingExecutionEligibilityEvidence:
            nonlocal physical_opened, evidence
            evidence = verify_serving_execution_eligibility_from_state(
                self._state,
                request,
            )
            if expected_serving_pointer_revision is not None:
                if (
                    evidence.serving_pointer_revision
                    != expected_serving_pointer_revision
                ):
                    raise ServingExecutionAdmissionError(
                        "serving pointer revision mismatch for unit admission open"
                    )
            unit_admission_open.apply_coordinator_verified_open(evidence)
            physical_opened = True
            return evidence

        try:
            return run_under_activation_domain(self._state, _work)
        except Exception:
            if evidence is not None:
                _compensate_unit_open_after_coordination_failure(
                    unit_admission_open,
                    evidence,
                    physical_opened=physical_opened,
                )
            raise


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
        physical_opened = False
        evidence: ServingExecutionEligibilityEvidence | None = None
        try:
            with _sqlite_admission_coordination(self._database) as state:
                evidence = verify_serving_execution_eligibility_from_state(
                    state, request
                )
                if expected_serving_pointer_revision is not None:
                    if (
                        evidence.serving_pointer_revision
                        != expected_serving_pointer_revision
                    ):
                        raise ServingExecutionAdmissionError(
                            "serving pointer revision mismatch for unit admission open"
                        )
                unit_admission_open.apply_coordinator_verified_open(evidence)
                physical_opened = True
            assert evidence is not None
            return evidence
        except Exception:
            if evidence is not None:
                _compensate_unit_open_after_coordination_failure(
                    unit_admission_open,
                    evidence,
                    physical_opened=physical_opened,
                )
            raise


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
