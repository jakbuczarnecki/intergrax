# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R4 — physical gate atomicity & concurrency qualification."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
    ServingExecutionEligibilityEvidence,
    _SqliteAdmissionDomainLease,
    _acquire_sqlite_admission_transaction,
    _release_sqlite_admission_transaction,
    _sqlite_admission_coordination,
    build_activation_domain_coordinator_for_sqlite_bundle,
    verify_serving_execution_eligibility_from_state,
)
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.in_memory_stores import AgentDistributionStoreState
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.sqlite_stores import build_sqlite_agent_distribution_store_bundle
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
)
from testing_support.agent_distribution.serving_execution_admission_fixtures import (
    seed_canonical_serving_execution_stores,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure import (
    _seeded_coordinator,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r3_coordinator_scoped_gate_closure import (
    _run_s2_concurrent_admission_wins,
)
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def _ready_unit_without_admission(tmp_path: Path):
    adapter, _, unit_ref, _, _, _, revision, digest = _minimal_projection_bundle(tmp_path)
    return adapter, unit_ref, digest, revision


def _unit_state(adapter: object, unit_ref: str) -> dict[str, object]:
    unit = adapter._units[unit_ref]
    return {
        "authorized": unit.serving_admission_authorized,
        "staged": unit.coordinator_admission_open_staged is not None,
        "gate_closed": unit.admission.admission_closed(),
        "in_flight": unit.admission.in_flight_count(),
        "lifecycle": unit.lifecycle_state,
    }


def test_staged_apply_does_not_authorize_reserve(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision = _ready_unit_without_admission(tmp_path)
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    evidence = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    port.apply_coordinator_verified_open(evidence)
    state = _unit_state(adapter, unit_ref)
    assert state["staged"] is True
    assert state["authorized"] is False
    assert state["gate_closed"] is True
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        adapter.reserve_unit_execution(unit_ref)


def test_publish_without_stage_fails_closed(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, *_ = _seeded_coordinator(tmp_path)
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    evidence = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not staged"):
        port.publish_coordinator_verified_open(evidence)


def test_sqlite_commit_failure_after_stage_compensates(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, serving_store, instance_store, revision_store, coordinator, request, _ = (
        _seeded_coordinator(tmp_path)
    )
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r4-commit-fail.db")
    activated_at = datetime.now(UTC)
    active_revision = revision.model_copy(
        update={
            "revision_state": RuntimeRevisionState.ACTIVE,
            "activated_at": activated_at,
        },
    )
    bundle.revision_store.persist_candidate_revision(active_revision)
    bundle.serving_store.atomic_swap_serving_revision(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=revision.runtime_revision_id,
        prior_revision_id=None,
        committed_at=activated_at,
    )
    from intergrax.agent_distribution.deployment import DeploymentInstanceRecord

    bundle.deployment_instance_store.persist_instance(
        DeploymentInstanceRecord(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            serving_unit_ref=unit_ref,
            instance_state=DeploymentInstanceState.SERVING,
            readiness_evidence_ref="evidence",
        ),
    )
    sqlite_coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()

    def _fail_commit(self: _SqliteAdmissionDomainLease) -> None:
        raise ServingExecutionAdmissionError("sqlite admission coordination commit failed")

    with mock.patch.object(_SqliteAdmissionDomainLease, "commit_durable", _fail_commit):
        with pytest.raises(ServingExecutionAdmissionError, match="commit failed"):
            sqlite_coordinator.coordinate_unit_serving_admission_open(
                request,
                unit_admission_open=port,
            )
    assert _unit_state(adapter, unit_ref)["staged"] is False
    assert _unit_state(adapter, unit_ref)["authorized"] is False


def test_sqlite_s2_admission_wins_concurrent(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, serving_store, instance_store, revision_store, coordinator, request, _ = (
        _seeded_coordinator(tmp_path)
    )
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r4-s2.db")
    activated_at = datetime.now(UTC)
    active_revision = revision.model_copy(
        update={
            "revision_state": RuntimeRevisionState.ACTIVE,
            "activated_at": activated_at,
        },
    )
    bundle.revision_store.persist_candidate_revision(active_revision)
    bundle.serving_store.atomic_swap_serving_revision(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=revision.runtime_revision_id,
        prior_revision_id=None,
        committed_at=activated_at,
    )
    from intergrax.agent_distribution.deployment import DeploymentInstanceRecord

    bundle.deployment_instance_store.persist_instance(
        DeploymentInstanceRecord(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            serving_unit_ref=unit_ref,
            instance_state=DeploymentInstanceState.SERVING,
            readiness_evidence_ref="evidence",
        ),
    )
    sqlite_coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    sqlite_coordinator.coordinate_unit_serving_admission_open(
        request,
        unit_admission_open=adapter.coordinator_verified_unit_serving_admission_open_port(
            unit_ref
        ),
    )

    def _reserve() -> object:
        return adapter.reserve_unit_execution(unit_ref)

    _run_s2_concurrent_admission_wins(
        serving_store=bundle.serving_store,
        coordinator=sqlite_coordinator,
        request=request,
        revision=revision,
        unit_ref=unit_ref,
        adapter=adapter,
        physical_reserve=_reserve,
    )


def test_publish_internal_failure_restores_safe_state(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision = _ready_unit_without_admission(tmp_path)
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    evidence = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    port.apply_coordinator_verified_open(evidence)
    with mock.patch(
        "intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter._UnitAdmissionGate.open_admission",
        side_effect=RuntimeError("open_admission failed"),
    ):
        with pytest.raises(RuntimeError, match="open_admission failed"):
            port.publish_coordinator_verified_open(evidence)
    state = _unit_state(adapter, unit_ref)
    assert state["authorized"] is False
    assert state["gate_closed"] is True


def test_sqlite_commit_failure_after_physical_reserve_compensates(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, *_ = _seeded_coordinator(tmp_path)
    request = ServingExecutionAdmissionRequest(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
    )
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r4-reserve-commit.db")
    activated_at = datetime.now(UTC)
    bundle.revision_store.persist_candidate_revision(
        revision.model_copy(
            update={
                "revision_state": RuntimeRevisionState.ACTIVE,
                "activated_at": activated_at,
            },
        ),
    )
    bundle.serving_store.atomic_swap_serving_revision(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=revision.runtime_revision_id,
        prior_revision_id=None,
        committed_at=activated_at,
    )
    from intergrax.agent_distribution.deployment import DeploymentInstanceRecord

    bundle.deployment_instance_store.persist_instance(
        DeploymentInstanceRecord(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            serving_unit_ref=unit_ref,
            instance_state=DeploymentInstanceState.SERVING,
            readiness_evidence_ref="evidence",
        ),
    )
    sqlite_coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    released = threading.Event()

    class _Token:
        def release(self) -> None:
            released.set()

    @contextmanager
    def _coordination_commit_fail(db):  # type: ignore[no-untyped-def]
        conn, state = _acquire_sqlite_admission_transaction(db)
        try:
            yield state
        except Exception:
            conn.rollback()
            raise
        else:
            conn.rollback()
            raise ServingExecutionAdmissionError("sqlite admission coordination commit failed")
        finally:
            _release_sqlite_admission_transaction(conn)

    with mock.patch(
        "intergrax.agent_distribution.activation_domain_coordinator._sqlite_admission_coordination",
        _coordination_commit_fail,
    ):
        with pytest.raises(ServingExecutionAdmissionError, match="commit failed"):
            sqlite_coordinator.coordinate_physical_execution_reservation(
                request,
                physical_reserve=lambda: _Token(),
            )
    assert released.is_set()
    assert _unit_state(adapter, unit_ref)["in_flight"] == 0


def test_concurrent_drain_clears_staged_open(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision = _ready_unit_without_admission(tmp_path)
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    evidence = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    port.apply_coordinator_verified_open(evidence)
    adapter.begin_drain(revision, serving_unit_ref=unit_ref)
    state = _unit_state(adapter, unit_ref)
    assert state["staged"] is False
    assert state["lifecycle"] is DeploymentInstanceState.DRAINING


def test_stop_invalidates_staged_and_authorized(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, *_ = _seeded_coordinator(tmp_path)
    unit = adapter._units[unit_ref]
    assert unit.serving_admission_authorized
    adapter.stop(revision, serving_unit_ref=unit_ref)
    assert unit_ref not in adapter._units or not unit.runtime_resource.is_live


def test_s2_admission_wins_in_memory_r4_regression(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, serving_store, instance_store, revision_store, coordinator, request, _ = (
        _seeded_coordinator(tmp_path)
    )

    def _reserve() -> object:
        return adapter.reserve_unit_execution(unit_ref)

    _run_s2_concurrent_admission_wins(
        serving_store=serving_store,
        coordinator=coordinator,
        request=request,
        revision=revision,
        unit_ref=unit_ref,
        adapter=adapter,
        physical_reserve=_reserve,
    )


def test_eligibility_failure_before_stage_leaves_unit_closed(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, serving_store, instance_store, revision_store, coordinator, request, _ = (
        _seeded_coordinator(tmp_path)
    )
    bad_request = request.model_copy(update={"application_environment_id": "wrong-env"})
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    with pytest.raises(ServingExecutionAdmissionError):
        coordinator.coordinate_unit_serving_admission_open(
            bad_request,
            unit_admission_open=port,
        )
    assert _unit_state(adapter, unit_ref)["staged"] is False
    assert _unit_state(adapter, unit_ref)["authorized"] is True  # prior authorize from fixture


def test_wrong_unit_release_is_safe(tmp_path: Path) -> None:
    adapter, unit_ref, *_ = _seeded_coordinator(tmp_path)
    reservation = adapter.reserve_unit_execution(unit_ref)
    reservation.release()
    with pytest.raises(VenvBundleColocatedDeploymentError, match="without matching"):
        reservation.release()


def test_cross_environment_evidence_rejected_on_stage(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision = _ready_unit_without_admission(tmp_path)
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    evidence = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id="other-env",
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="environment mismatch"):
        port.apply_coordinator_verified_open(evidence)
