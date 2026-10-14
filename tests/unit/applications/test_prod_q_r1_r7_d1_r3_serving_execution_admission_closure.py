# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R3 — canonical serving-state execution admission closure."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.serving_execution_admission import ServingExecutionAdmissionError
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.colocated_serving_runtime_composition import (
    bind_colocated_serving_runtime_after_activation,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoffError,
    authorize_colocated_serving_execution_after_activation,
    establish_serving_runtime_handoff,
)
from testing_support.agent_distribution.serving_execution_admission_fixtures import (
    activation_domain_coordinator_for_stores,
)
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _canonical_commit_and_stores,
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def test_identifier_only_adapter_authorization_rejected(tmp_path: Path) -> None:
    adapter, _accessor, unit_ref, _att, _dep, _bundle, revision, _ = _minimal_projection_bundle(
        tmp_path
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="activation domain coordinator"):
        adapter.authorize_serving_execution_admission(unit_ref)  # type: ignore[arg-type]


def test_ready_deployment_instance_denied_at_authorize(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    composition = create_reference_production_process_composition()
    handoff, _ = establish_serving_runtime_handoff(
        composition=composition,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    commit, serving_store, instance_store, revision_store = _canonical_commit_and_stores(
        revision,
        deployment_instance,
    )
    ready_commit = commit.model_copy(
        update={
            "candidate_instance": commit.candidate_instance.model_copy(
                update={"instance_state": DeploymentInstanceState.READY},
            ),
        },
    )
    with pytest.raises(ServingRuntimeHandoffError, match="not serving"):
        authorize_colocated_serving_execution_after_activation(
            activation_commit=ready_commit,
            handoff=handoff,
            unit_admission_open=adapter.coordinator_verified_unit_serving_admission_open_port(
                unit_ref
            ),
            activation_domain_coordinator=activation_domain_coordinator_for_stores(
                serving_store,
                instance_store,
                revision_store,
            ),
            serving_store=serving_store,
            deployment_instance_store=instance_store,
            revision_store=revision_store,
        )


def test_stale_activation_pointer_denied(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    handoff, _ = establish_serving_runtime_handoff(
        composition=create_reference_production_process_composition(),
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    commit, serving_store, instance_store, revision_store = _canonical_commit_and_stores(
        revision,
        deployment_instance,
    )
    stale = commit.model_copy(
        update={
            "serving_record": commit.serving_record.model_copy(
                update={"serving_pointer_revision": 0},
            ),
        },
    )
    with pytest.raises(ServingRuntimeHandoffError, match="pointer is stale"):
        authorize_colocated_serving_execution_after_activation(
            activation_commit=stale,
            handoff=handoff,
            unit_admission_open=adapter.coordinator_verified_unit_serving_admission_open_port(
                unit_ref
            ),
            activation_domain_coordinator=activation_domain_coordinator_for_stores(
                serving_store,
                instance_store,
                revision_store,
            ),
            serving_store=serving_store,
            deployment_instance_store=instance_store,
            revision_store=revision_store,
        )


def test_valid_current_active_serving_admitted(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    commit, serving_store, instance_store, revision_store = _canonical_commit_and_stores(
        revision,
        deployment_instance,
    )
    updated = bind_colocated_serving_runtime_after_activation(
        composition=create_reference_production_process_composition(),
        activation_commit=commit,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        physical_adapter=adapter,
        serving_store=serving_store,
        deployment_instance_store=instance_store,
        revision_store=revision_store,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    handoff = updated.serving_runtime_handoff
    assert handoff is not None
    admission = handoff.colocated_execution_admission
    assert admission is not None
    admission.admit_execution(unit_ref)


def test_superseded_pointer_blocks_per_execution_admission(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    commit, serving_store, instance_store, revision_store = _canonical_commit_and_stores(
        revision,
        deployment_instance,
    )
    updated = bind_colocated_serving_runtime_after_activation(
        composition=create_reference_production_process_composition(),
        activation_commit=commit,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        physical_adapter=adapter,
        serving_store=serving_store,
        deployment_instance_store=instance_store,
        revision_store=revision_store,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    handoff = updated.serving_runtime_handoff
    assert handoff is not None
    scope_record = serving_store.get_serving_record(
        revision.application_id,
        revision.application_environment_id,
    )
    assert scope_record is not None
    serving_store.atomic_swap_serving_revision(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        expected_current_revision_id=scope_record.traffic_serving_revision_id,
        expected_pointer_revision=scope_record.serving_pointer_revision,
        new_revision_id="rev-superseded",
        prior_revision_id=revision.runtime_revision_id,
        committed_at=scope_record.committed_at,
    )
    admission = handoff.colocated_execution_admission
    assert admission is not None
    with pytest.raises(ServingRuntimeHandoffError, match="does not target"):
        admission.admit_execution(unit_ref)
