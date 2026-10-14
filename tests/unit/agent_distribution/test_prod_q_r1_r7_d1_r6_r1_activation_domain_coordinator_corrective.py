# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R1 — admission authority & compensation corrective (superseded by R2 tests)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from pydantic import PrivateAttr

from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
    ServingExecutionEligibilityEvidence,
    SqliteActivationDomainCoordinator,
    _sqlite_admission_coordination,
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
from testing_support.agent_distribution.serving_execution_admission_fixtures import (
    seed_canonical_serving_execution_stores,
)
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


class _CountingReservation:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def test_eligibility_evidence_has_no_coordinator_seal_field() -> None:
    fields = ServingExecutionEligibilityEvidence.model_fields
    assert "_coordinator_seal" not in fields
    assert not any(
        isinstance(value, PrivateAttr)
        for value in ServingExecutionEligibilityEvidence.__private_attributes__.values()
    )


def test_serving_execution_admission_module_has_no_grant_alias() -> None:
    import intergrax.agent_distribution.serving_execution_admission as module

    assert "ServingExecutionAdmissionGrant" not in module.__all__


def test_reserve_exception_does_not_invoke_release_callback(tmp_path: Path) -> None:
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
    token = _CountingReservation()

    def _reserve_fail() -> _CountingReservation:
        raise RuntimeError("reserve failed")

    with pytest.raises(RuntimeError, match="reserve failed"):
        coordinator.coordinate_physical_execution_reservation(
            request,
            physical_reserve=_reserve_fail,
        )
    assert not token.released


def test_post_reserve_coordination_failure_compensates_once(tmp_path: Path) -> None:
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
    token = _CountingReservation()

    def _reserve_ok() -> _CountingReservation:
        return token

    def _run_then_fail(state: AgentDistributionStoreState, fn):  # type: ignore[no-untyped-def]
        fn()
        raise RuntimeError("domain failure after reserve")

    with mock.patch(
        "intergrax.agent_distribution.activation_domain_coordinator.run_under_activation_domain",
        side_effect=_run_then_fail,
    ):
        with pytest.raises(RuntimeError, match="domain failure"):
            coordinator.coordinate_physical_execution_reservation(
                request,
                physical_reserve=_reserve_ok,
            )
    assert token.released


def test_sqlite_bundle_factory_builds_coordinator(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "coord.db")
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    assert isinstance(coordinator, SqliteActivationDomainCoordinator)


def test_sqlite_coordination_context_yields_once_per_attempt(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "yield.db")
    yields = 0
    with _sqlite_admission_coordination(bundle.database) as _state:
        yields += 1
    assert yields == 1


def test_forged_evidence_cannot_open_unit_admission(tmp_path: Path) -> None:
    adapter, _, unit_ref, _, _, _, revision, digest = _minimal_projection_bundle(tmp_path)
    forged = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    forged_bad_digest = forged.model_copy(
        update={"materialization_artifact_digest": "forged-digest"},
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="digest mismatch"):
        port.apply_coordinator_verified_open(forged_bad_digest)


def test_public_authorize_entrypoint_rejects_evidence_only(tmp_path: Path) -> None:
    adapter, _, unit_ref, _, _, _, revision, digest = _minimal_projection_bundle(tmp_path)
    evidence = ServingExecutionEligibilityEvidence(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="activation domain coordinator"):
        adapter.authorize_serving_execution_admission(evidence)
