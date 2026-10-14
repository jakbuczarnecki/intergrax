# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Transaction-coordinated serving execution admission (PROD-Q-R1-R7-D1-R6)."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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
    SqliteApplicationEnvironmentServingStore,
    SqliteDeploymentInstanceStore,
    SqliteRuntimeRevisionStore,
)

_NON_EMPTY = Field(min_length=1)
_SQLITE_BUSY_MAX_ATTEMPTS = 5
_SQLITE_BUSY_BASE_DELAY_S = 0.01
_COORDINATOR_UNIT_OPEN_DEPTH = threading.local()


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
    """Non-authoritative eligibility snapshot — not valid without coordinator seal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    serving_unit_ref: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    serving_pointer_revision: int = Field(ge=0)
    _coordinator_seal: str | None = PrivateAttr(default=None)


def _issue_eligibility_evidence(
    *,
    request: ServingExecutionAdmissionRequest,
    serving_pointer_revision: int,
) -> ServingExecutionEligibilityEvidence:
    evidence = ServingExecutionEligibilityEvidence(
        application_id=request.application_id,
        application_environment_id=request.application_environment_id,
        runtime_revision_id=request.runtime_revision_id,
        serving_unit_ref=request.serving_unit_ref,
        materialization_artifact_digest=request.materialization_artifact_digest,
        serving_pointer_revision=serving_pointer_revision,
    )
    object.__setattr__(evidence, "_coordinator_seal", "activation-domain-coordinator")
    return evidence


def assert_coordinator_issued_eligibility_evidence(
    evidence: ServingExecutionEligibilityEvidence,
) -> None:
    if evidence._coordinator_seal != "activation-domain-coordinator":
        raise ServingExecutionAdmissionError(
            "serving execution eligibility evidence was not issued by activation domain coordinator"
        )


def assert_within_coordinator_unit_admission_open() -> None:
    if getattr(_COORDINATOR_UNIT_OPEN_DEPTH, "depth", 0) < 1:
        raise ServingExecutionAdmissionError(
            "unit serving admission open requires activation domain coordinator"
        )


def _invoke_unit_admission_open(
    evidence: ServingExecutionEligibilityEvidence,
    open_unit_admission: Callable[[ServingExecutionEligibilityEvidence], None],
) -> None:
    prior_depth = getattr(_COORDINATOR_UNIT_OPEN_DEPTH, "depth", 0)
    _COORDINATOR_UNIT_OPEN_DEPTH.depth = prior_depth + 1
    try:
        open_unit_admission(evidence)
    finally:
        _COORDINATOR_UNIT_OPEN_DEPTH.depth = prior_depth


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

    return _issue_eligibility_evidence(
        request=request,
        serving_pointer_revision=serving.serving_pointer_revision,
    )


class ActivationDomainCoordinator(Protocol):
    """Provider-neutral boundary between canonical serving state and physical reservation."""

    def coordinate_physical_execution_reservation(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        physical_reserve: Callable[[], None],
        physical_release: Callable[[], None],
    ) -> ServingExecutionEligibilityEvidence: ...

    def coordinate_unit_serving_admission_open(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        open_unit_admission: Callable[[ServingExecutionEligibilityEvidence], None],
    ) -> ServingExecutionEligibilityEvidence: ...


@contextmanager
def _sqlite_admission_coordination(db: SqliteAgentDistributionDatabase):
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(1, _SQLITE_BUSY_MAX_ATTEMPTS + 1):
        with db._activation_lock:
            conn = db._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                state = db._load_state(conn)
                try:
                    yield state
                except Exception:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
                return
            except sqlite3.OperationalError as exc:
                conn.rollback()
                if "locked" in str(exc).lower() and attempt < _SQLITE_BUSY_MAX_ATTEMPTS:
                    last_error = exc
                    time.sleep(_SQLITE_BUSY_BASE_DELAY_S * attempt)
                    continue
                raise ServingExecutionAdmissionError(
                    "sqlite admission coordination failed"
                ) from exc
            finally:
                conn.close()
    if last_error is not None:
        raise ServingExecutionAdmissionError(
            "sqlite admission coordination exceeded busy retry budget"
        ) from last_error


class InMemoryActivationDomainCoordinator:
    """Shared in-memory Agent Distribution activation domain coordinator."""

    def __init__(self, state: AgentDistributionStoreState) -> None:
        self._state = state

    def coordinate_physical_execution_reservation(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        physical_reserve: Callable[[], None],
        physical_release: Callable[[], None],
    ) -> ServingExecutionEligibilityEvidence:
        reserved = False
        evidence: ServingExecutionEligibilityEvidence | None = None

        def _work() -> ServingExecutionEligibilityEvidence:
            nonlocal reserved, evidence
            evidence = verify_serving_execution_eligibility_from_state(
                self._state,
                request,
            )
            try:
                physical_reserve()
                reserved = True
            except Exception:
                raise
            return evidence

        try:
            return run_under_activation_domain(self._state, _work)
        except Exception:
            if reserved:
                physical_release()
            raise

    def coordinate_unit_serving_admission_open(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        open_unit_admission: Callable[[ServingExecutionEligibilityEvidence], None],
    ) -> ServingExecutionEligibilityEvidence:
        def _work() -> ServingExecutionEligibilityEvidence:
            evidence = verify_serving_execution_eligibility_from_state(
                self._state,
                request,
            )
            _invoke_unit_admission_open(evidence, open_unit_admission)
            return evidence

        return run_under_activation_domain(self._state, _work)


class SqliteActivationDomainCoordinator:
    """SQLite-backed activation domain coordinator (one IMMEDIATE snapshot per admission)."""

    def __init__(self, database: SqliteAgentDistributionDatabase) -> None:
        self._database = database

    def coordinate_physical_execution_reservation(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        physical_reserve: Callable[[], None],
        physical_release: Callable[[], None],
    ) -> ServingExecutionEligibilityEvidence:
        reserved = False
        evidence: ServingExecutionEligibilityEvidence | None = None
        try:
            with _sqlite_admission_coordination(self._database) as state:
                evidence = verify_serving_execution_eligibility_from_state(state, request)
                physical_reserve()
                reserved = True
            assert evidence is not None
            return evidence
        except Exception:
            if reserved:
                physical_release()
            raise

    def coordinate_unit_serving_admission_open(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        open_unit_admission: Callable[[ServingExecutionEligibilityEvidence], None],
    ) -> ServingExecutionEligibilityEvidence:
        with _sqlite_admission_coordination(self._database) as state:
            evidence = verify_serving_execution_eligibility_from_state(state, request)
            _invoke_unit_admission_open(evidence, open_unit_admission)
            return evidence


def _shared_in_memory_state_from_stores(
    serving_store: object,
    deployment_instance_store: object,
    revision_store: object,
) -> AgentDistributionStoreState:
    if not isinstance(serving_store, InMemoryApplicationEnvironmentServingStore):
        raise ActivationDomainCoordinatorError(
            "in-memory activation domain requires InMemoryApplicationEnvironmentServingStore"
        )
    state = serving_store.state
    if not isinstance(deployment_instance_store, InMemoryDeploymentInstanceStore):
        raise ActivationDomainCoordinatorError(
            "in-memory stores must share one AgentDistributionStoreState bundle"
        )
    if deployment_instance_store.state is not state:
        raise ActivationDomainCoordinatorError(
            "deployment instance store does not share serving store state"
        )
    if not isinstance(revision_store, InMemoryRuntimeRevisionStore):
        raise ActivationDomainCoordinatorError(
            "in-memory stores must share one AgentDistributionStoreState bundle"
        )
    if revision_store.state is not state:
        raise ActivationDomainCoordinatorError(
            "revision store does not share serving store state"
        )
    return state


def _shared_sqlite_database_from_stores(
    serving_store: object,
    deployment_instance_store: object,
    revision_store: object,
) -> SqliteAgentDistributionDatabase:
    if not isinstance(serving_store, SqliteApplicationEnvironmentServingStore):
        raise ActivationDomainCoordinatorError(
            "sqlite activation domain requires SqliteApplicationEnvironmentServingStore"
        )
    database = serving_store._db
    if not isinstance(deployment_instance_store, SqliteDeploymentInstanceStore):
        raise ActivationDomainCoordinatorError(
            "sqlite stores must share one SqliteAgentDistributionDatabase"
        )
    if deployment_instance_store._db is not database:
        raise ActivationDomainCoordinatorError(
            "deployment instance store does not share sqlite database"
        )
    if not isinstance(revision_store, SqliteRuntimeRevisionStore):
        raise ActivationDomainCoordinatorError(
            "sqlite stores must share one SqliteAgentDistributionDatabase"
        )
    if revision_store._db is not database:
        raise ActivationDomainCoordinatorError(
            "revision store does not share sqlite database"
        )
    return database


def build_activation_domain_coordinator_for_stores(
    *,
    serving_store: object,
    deployment_instance_store: object,
    revision_store: object,
) -> ActivationDomainCoordinator:
    """Construct coordinator backend matching the sanctioned store bundle topology."""
    if isinstance(serving_store, InMemoryApplicationEnvironmentServingStore):
        state = _shared_in_memory_state_from_stores(
            serving_store,
            deployment_instance_store,
            revision_store,
        )
        return InMemoryActivationDomainCoordinator(state)
    if isinstance(serving_store, SqliteApplicationEnvironmentServingStore):
        database = _shared_sqlite_database_from_stores(
            serving_store,
            deployment_instance_store,
            revision_store,
        )
        return SqliteActivationDomainCoordinator(database)
    raise ActivationDomainCoordinatorError(
        "unsupported serving store for activation domain coordinator"
    )


__all__ = [
    "ActivationDomainCoordinator",
    "ActivationDomainCoordinatorError",
    "InMemoryActivationDomainCoordinator",
    "ServingExecutionAdmissionRequest",
    "ServingExecutionEligibilityEvidence",
    "SqliteActivationDomainCoordinator",
    "assert_coordinator_issued_eligibility_evidence",
    "assert_within_coordinator_unit_admission_open",
    "build_activation_domain_coordinator_for_stores",
    "verify_serving_execution_eligibility_from_state",
]
