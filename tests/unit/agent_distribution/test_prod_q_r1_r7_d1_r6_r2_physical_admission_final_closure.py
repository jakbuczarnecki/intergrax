# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R2 — physical admission bypass closure & SQLite concurrency evidence."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
    ServingExecutionEligibilityEvidence,
    _sqlite_admission_domain_lease,
    build_activation_domain_coordinator_for_sqlite_bundle,
    build_activation_domain_coordinator_for_stores,
)
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.in_memory_stores import AgentDistributionStoreState
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.sqlite_stores import build_sqlite_agent_distribution_store_bundle
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
)
from intergrax.applications._shared.colocated_serving_unit_execution_admission import (
    StoreBackedColocatedServingUnitExecutionAdmission,
)
from intergrax.applications._shared.serving_runtime_handoff import ServingRuntimeHandoffError
from intergrax.applications._shared.colocated_unit_physical_execution_gate import (
    as_colocated_unit_physical_gate,
)
from intergrax.applications._shared.serving_runtime_handoff import ServingUnitIdentity
from testing_support.agent_distribution.serving_execution_admission_fixtures import (
    seed_canonical_serving_execution_stores,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter import (
    _authorize_adapter_execution_gate,
)
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def _seeded_coordinator(tmp_path: Path):
    adapter, _, unit_ref, _, deployment_instance, _, revision, digest = (
        _minimal_projection_bundle(tmp_path)
    )
    state = AgentDistributionStoreState()
    serving_record = ApplicationEnvironmentServingRecord(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        traffic_serving_revision_id=revision.runtime_revision_id,
        serving_pointer_revision=1,
    )
    serving_store, instance_store, revision_store = seed_canonical_serving_execution_stores(
        state=state,
        revision=revision,
        serving_record=serving_record,
        deployment_instance=deployment_instance.model_copy(
            update={"instance_state": DeploymentInstanceState.SERVING},
        ),
    )
    coordinator = build_activation_domain_coordinator_for_stores(
        serving_store=serving_store,
        deployment_instance_store=instance_store,
        revision_store=revision_store,
    )
    _authorize_adapter_execution_gate(
        adapter,
        serving_store=serving_store,
        instance_store=instance_store,
        revision_store=revision_store,
        revision=revision,
        serving_unit_ref=unit_ref,
    )
    request = ServingExecutionAdmissionRequest(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
    )
    identity = ServingUnitIdentity(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        materialization_topology=revision.materialization_topology,
    )
    admission = StoreBackedColocatedServingUnitExecutionAdmission(
        physical_gate=as_colocated_unit_physical_gate(adapter),
        activation_domain_coordinator=coordinator,
        identity=identity,
    )
    return (
        adapter,
        unit_ref,
        digest,
        revision,
        serving_store,
        instance_store,
        revision_store,
        coordinator,
        request,
        admission,
    )


def test_operation_authority_type_removed_from_public_exports() -> None:
    import intergrax.agent_distribution.activation_domain_coordinator as module

    assert "ActivationDomainCoordinatorOperationAuthority" not in module.__all__


def test_adapter_has_no_global_admission_open_entrypoint(tmp_path: Path) -> None:
    adapter, unit_ref, *_rest = _seeded_coordinator(tmp_path)
    assert not hasattr(adapter, "open_unit_serving_admission_after_coordinator_verify")
    assert not hasattr(adapter, "_coordinator_synchronized_admission_open_depth")
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    assert port is not None


def test_identifier_only_reserve_without_traffic_authorization_fails(
    tmp_path: Path,
) -> None:
    adapter, _, unit_ref, *_rest = _minimal_projection_bundle(tmp_path)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        adapter.reserve_unit_execution(unit_ref)


def test_s1_pointer_swap_wins_before_admission_check(tmp_path: Path) -> None:
    (
        adapter,
        unit_ref,
        _digest,
        revision,
        serving_store,
        instance_store,
        revision_store,
        _coordinator,
        _request,
        admission,
    ) = _seeded_coordinator(tmp_path)
    scope = serving_store.get_serving_record(
        revision.application_id,
        revision.application_environment_id,
    )
    assert scope is not None
    serving_store.atomic_swap_serving_revision(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        expected_current_revision_id=scope.traffic_serving_revision_id,
        expected_pointer_revision=scope.serving_pointer_revision,
        new_revision_id="rev-race",
        prior_revision_id=revision.runtime_revision_id,
        committed_at=datetime.now(UTC),
    )
    with pytest.raises(ServingRuntimeHandoffError, match="does not target"):
        admission.admit_execution(unit_ref)
    unit = adapter._units[unit_ref]  # qualification: inspect in-flight accounting
    assert unit.admission.in_flight_count() == 0


def test_s2_admitted_in_flight_survives_post_coordination_pointer_swap(
    tmp_path: Path,
) -> None:
    (
        adapter,
        unit_ref,
        _digest,
        revision,
        serving_store,
        _instance_store,
        _revision_store,
        coordinator,
        request,
        admission,
    ) = _seeded_coordinator(tmp_path)
    held: list[object] = []

    def _reserve() -> object:
        token = adapter.reserve_unit_execution(unit_ref)
        held.append(token)
        return token

    evidence = coordinator.coordinate_physical_execution_reservation(
        request,
        physical_reserve=_reserve,
    )
    assert evidence.serving_unit_ref == unit_ref
    assert adapter._units[unit_ref].admission.in_flight_count() == 1
    scope = serving_store.get_serving_record(
        revision.application_id,
        revision.application_environment_id,
    )
    assert scope is not None
    serving_store.atomic_swap_serving_revision(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        expected_current_revision_id=scope.traffic_serving_revision_id,
        expected_pointer_revision=scope.serving_pointer_revision,
        new_revision_id="rev-race",
        prior_revision_id=revision.runtime_revision_id,
        committed_at=datetime.now(UTC),
    )
    assert adapter._units[unit_ref].admission.in_flight_count() == 1
    with pytest.raises(ServingRuntimeHandoffError):
        admission.admit_execution(unit_ref)
    held[0].release()


def test_s4_two_sqlite_connections_serialize_admission(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "dual.db")
    with _sqlite_admission_domain_lease(bundle.database) as lease_a:
        second_acquired = threading.Event()

        def _second_connection() -> None:
            with _sqlite_admission_domain_lease(bundle.database) as lease_b:
                lease_b.rollback()
            second_acquired.set()

        worker = threading.Thread(target=_second_connection)
        worker.start()
        assert not second_acquired.wait(timeout=0.2)
        lease_a.rollback()
    worker.join(timeout=5.0)
    assert second_acquired.is_set()


def test_s5_nested_immediate_transaction_fails_closed(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "nested.db")
    with _sqlite_admission_domain_lease(bundle.database) as lease:
        with pytest.raises(sqlite3.OperationalError):
            lease.conn.execute("BEGIN IMMEDIATE")
        lease.rollback()


def test_s6_sqlite_busy_retry_budget_exhausted(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "busy.db")
    calls = 0
    real_connect = bundle.database._connect

    class _BusyBeginConnection:
        __slots__ = ("_inner",)

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, statement, *args, **kwargs):
            nonlocal calls
            if "BEGIN IMMEDIATE" in str(statement):
                calls += 1
                raise sqlite3.OperationalError("database is locked")
            return self._inner.execute(statement, *args, **kwargs)

        def rollback(self) -> None:
            self._inner.rollback()

        def close(self) -> None:
            self._inner.close()

    def _connect_with_busy_begin() -> _BusyBeginConnection:
        return _BusyBeginConnection(real_connect())

    with mock.patch.object(bundle.database, "_connect", side_effect=_connect_with_busy_begin):
        with pytest.raises(ServingExecutionAdmissionError, match="sqlite admission coordination"):
            with _sqlite_admission_domain_lease(bundle.database):
                pass
    assert calls == 5


def test_double_physical_reservation_release_is_deterministic(tmp_path: Path) -> None:
    adapter, unit_ref, *_rest = _seeded_coordinator(tmp_path)
    reservation = adapter.reserve_unit_execution(unit_ref)
    reservation.release()
    with pytest.raises(VenvBundleColocatedDeploymentError, match="without matching"):
        reservation.release()
