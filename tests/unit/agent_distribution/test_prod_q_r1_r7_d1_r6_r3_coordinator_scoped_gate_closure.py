# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R3 — coordinator-scoped gate opening & deterministic race closure."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from intergrax.agent_distribution.activation_domain_coordinator import (
    ActivationDomainCoordinator,
    CoordinatorVerifiedUnitServingAdmissionOpen,
    InMemoryActivationDomainCoordinator,
    ServingExecutionAdmissionRequest,
    ServingExecutionEligibilityEvidence,
    build_activation_domain_coordinator_for_stores,
    verify_serving_execution_eligibility_from_state,
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
from intergrax.applications._shared.colocated_unit_physical_execution_gate import (
    coordinator_verified_unit_serving_admission_open_port,
)
from testing_support.agent_distribution.serving_execution_admission_fixtures import (
    seed_canonical_serving_execution_stores,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure import (
    _seeded_coordinator,
)
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


class _AltInMemoryCoordinator:
    """Minimal structural substitute for ``ActivationDomainCoordinator``."""

    def __init__(self, inner: InMemoryActivationDomainCoordinator) -> None:
        self._inner = inner
        self.calls = 0

    def coordinate_physical_execution_reservation(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        physical_reserve: Callable[[], object],
    ) -> ServingExecutionEligibilityEvidence:
        self.calls += 1
        return self._inner.coordinate_physical_execution_reservation(
            request,
            physical_reserve=physical_reserve,
        )

    def coordinate_unit_serving_admission_open(
        self,
        request: ServingExecutionAdmissionRequest,
        *,
        unit_admission_open: CoordinatorVerifiedUnitServingAdmissionOpen,
        expected_serving_pointer_revision: int | None = None,
    ) -> ServingExecutionEligibilityEvidence:
        self.calls += 1
        return self._inner.coordinate_unit_serving_admission_open(
            request,
            unit_admission_open=unit_admission_open,
            expected_serving_pointer_revision=expected_serving_pointer_revision,
        )


class _RecordingUnitOpenPort:
    def __init__(self, inner: CoordinatorVerifiedUnitServingAdmissionOpen) -> None:
        self._inner = inner
        self.applied = 0
        self.compensated = 0

    def apply_coordinator_verified_open(
        self, evidence: ServingExecutionEligibilityEvidence
    ) -> None:
        self.applied += 1
        self._inner.apply_coordinator_verified_open(evidence)

    def publish_coordinator_verified_open(
        self, evidence: ServingExecutionEligibilityEvidence
    ) -> None:
        self._inner.publish_coordinator_verified_open(evidence)

    def compensate_failed_coordination_open(
        self, evidence: ServingExecutionEligibilityEvidence
    ) -> None:
        self.compensated += 1
        self._inner.compensate_failed_coordination_open(evidence)


def test_coordinator_exports_exclude_depth_scope_protocols() -> None:
    import intergrax.agent_distribution.activation_domain_coordinator as module

    assert "CoordinatorSynchronizedColocatedServingUnitAdmissionPort" not in module.__all__
    assert "ColocatedServingUnitAdmissionPort" not in module.__all__
    assert "CoordinatorVerifiedUnitServingAdmissionOpen" in module.__all__


def test_structural_activation_domain_coordinator_substitution(
    tmp_path: Path,
) -> None:
    _, _, unit_ref, _, deployment_instance, _, revision, digest = _minimal_projection_bundle(
        tmp_path
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
    inner = build_activation_domain_coordinator_for_stores(
        serving_store=serving_store,
        deployment_instance_store=instance_store,
        revision_store=revision_store,
    )
    alt: ActivationDomainCoordinator = _AltInMemoryCoordinator(inner)
    request = ServingExecutionAdmissionRequest(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
    )

    class _Token:
        def release(self) -> None:
            return None

    alt.coordinate_physical_execution_reservation(
        request, physical_reserve=_Token
    )
    assert alt.calls == 1


def test_structural_unit_admission_open_port_substitution(tmp_path: Path) -> None:
    adapter, _accessor, unit_ref, _att, deployment_instance, _bundle, revision, digest = (
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
    request = ServingExecutionAdmissionRequest(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
    )
    inner = coordinator_verified_unit_serving_admission_open_port(adapter, unit_ref)
    port: CoordinatorVerifiedUnitServingAdmissionOpen = _RecordingUnitOpenPort(inner)
    coordinator.coordinate_unit_serving_admission_open(
        request,
        unit_admission_open=port,
    )
    assert isinstance(port, _RecordingUnitOpenPort)
    assert port.applied == 1
    unit = adapter._units[unit_ref]
    assert unit.serving_admission_authorized


def _run_s2_concurrent_admission_wins(
    *,
    serving_store: object,
    coordinator: object,
    request: ServingExecutionAdmissionRequest,
    revision: object,
    unit_ref: str,
    adapter: object,
    physical_reserve: Callable[[], object],
) -> None:
    t1_inside = threading.Event()
    t2_started_swap = threading.Event()
    reserve_done = threading.Event()
    swap_done = threading.Event()
    original_verify = verify_serving_execution_eligibility_from_state

    def _verify_hold(state, req):  # type: ignore[no-untyped-def]
        evidence = original_verify(state, req)
        t1_inside.set()
        assert t2_started_swap.wait(timeout=3.0)
        return evidence

    def _reserve() -> object:
        token = physical_reserve()
        reserve_done.set()
        return token

    def _t1() -> None:
        with mock.patch(
            "intergrax.agent_distribution.activation_domain_coordinator.verify_serving_execution_eligibility_from_state",
            side_effect=_verify_hold,
        ):
            coordinator.coordinate_physical_execution_reservation(
                request,
                physical_reserve=_reserve,
            )

    def _t2() -> None:
        assert t1_inside.wait(timeout=3.0)
        t2_started_swap.set()
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
            new_revision_id="rev-race-s2",
            prior_revision_id=revision.runtime_revision_id,
            committed_at=datetime.now(UTC),
        )
        swap_done.set()

    t1 = threading.Thread(target=_t1)
    t2 = threading.Thread(target=_t2)
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    assert reserve_done.is_set()
    assert swap_done.is_set()
    assert adapter._units[unit_ref].admission.in_flight_count() == 1


def test_s2_admission_wins_concurrent_in_memory(tmp_path: Path) -> None:
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


def test_coordination_failure_after_unit_open_compensates_gate(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, serving_store, instance_store, revision_store, coordinator, request, _ = (
        _seeded_coordinator(tmp_path)
    )
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.admission.begin_drain()
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)

    def _run_then_fail(state: AgentDistributionStoreState, fn):  # type: ignore[no-untyped-def]
        fn()
        raise RuntimeError("domain failure after unit open")

    with mock.patch(
        "intergrax.agent_distribution.activation_domain_coordinator.run_under_activation_domain",
        side_effect=_run_then_fail,
    ):
        with pytest.raises(RuntimeError, match="domain failure"):
            coordinator.coordinate_unit_serving_admission_open(
                request,
                unit_admission_open=port,
            )
    assert not unit.serving_admission_authorized
    assert unit.admission.admission_closed()


def test_bound_port_rejects_evidence_for_different_unit(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, *_ = _seeded_coordinator(tmp_path)
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    evidence = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref="venv-colocated:other-unit",
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="unit mismatch"):
        port.apply_coordinator_verified_open(evidence)
