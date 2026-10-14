# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6 — ActivationDomainCoordinator qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
    build_activation_domain_coordinator_for_stores,
)
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.in_memory_stores import AgentDistributionStoreState
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.sqlite_stores import build_sqlite_agent_distribution_store_bundle
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.applications._shared.colocated_serving_runtime_composition import (
    ColocatedServingAdmissionTopologyError,
    validate_colocated_serving_admission_topology,
)
from testing_support.agent_distribution.serving_execution_admission_fixtures import (
    seed_canonical_serving_execution_stores,
)
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def _request_for_revision(revision, unit_ref: str, digest: str) -> ServingExecutionAdmissionRequest:
    return ServingExecutionAdmissionRequest(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
    )


def test_sqlite_coordinator_rejects_mismatched_database_objects(tmp_path: Path) -> None:
    bundle_a = build_sqlite_agent_distribution_store_bundle(tmp_path / "a.db")
    bundle_b = build_sqlite_agent_distribution_store_bundle(tmp_path / "b.db")
    with pytest.raises(ColocatedServingAdmissionTopologyError):
        validate_colocated_serving_admission_topology(
            serving_store=bundle_a.serving_store,
            deployment_instance_store=bundle_b.deployment_instance_store,
            revision_store=bundle_b.revision_store,
        )


def test_sequential_reservations_invoke_physical_hooks(tmp_path: Path) -> None:
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
    request = _request_for_revision(revision, unit_ref, digest)
    calls = 0

    def _reserve() -> None:
        nonlocal calls
        calls += 1

    for _ in range(2):
        coordinator.coordinate_physical_execution_reservation(
            request,
            physical_reserve=_reserve,
            physical_release=lambda: None,
        )
    assert calls == 2


def test_wrong_environment_identity_fails_closed(tmp_path: Path) -> None:
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
    bad_request = ServingExecutionAdmissionRequest(
        application_id=revision.application_id,
        application_environment_id="wrong-env",
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
    )
    with pytest.raises(ServingExecutionAdmissionError, match="missing"):
        coordinator.coordinate_physical_execution_reservation(
            bad_request,
            physical_reserve=lambda: None,
            physical_release=lambda: None,
        )


def test_in_memory_topology_validation_passes(tmp_path: Path) -> None:
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
        deployment_instance=deployment_instance,
    )
    coordinator = validate_colocated_serving_admission_topology(
        serving_store=serving_store,
        deployment_instance_store=instance_store,
        revision_store=revision_store,
    )
    assert coordinator is not None
