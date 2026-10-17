# © Artur Czarnecki. All rights reserved.

"""R7-Q §C DIM13 — recovery admission authority (no Applications private adapter inspection)."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.agent_distribution.serving_execution_admission_fixtures import (
    activation_domain_coordinator_for_stores,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
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
)

from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _canonical_commit_and_stores,
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def _bind_once(tmp_path: Path, composition):
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    commit, serving_store, instance_store, revision_store = _canonical_commit_and_stores(
        revision,
        deployment_instance,
    )
    updated = bind_colocated_serving_runtime_after_activation(
        composition=composition,
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
    return (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        revision,
        commit,
        serving_store,
        instance_store,
        revision_store,
        updated,
    )


def test_dim13_recovery_rebind_uses_reopen_not_duplicate_grant(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    first = _bind_once(tmp_path, composition)
    (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        revision,
        commit,
        serving_store,
        instance_store,
        revision_store,
        _,
    ) = first
    unit = adapter._units[unit_ref]
    pointer_before = unit.coordinator_verified_published_pointer_revision
    bind_colocated_serving_runtime_after_activation(
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
        admission_bind_mode="reopen",
    )
    assert unit.coordinator_verified_published_pointer_revision == pointer_before
    adapter.reserve_unit_execution(unit_ref).release()


def test_dim13_recovery_reopen_rejected_after_drain(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        revision,
        commit,
        serving_store,
        instance_store,
        revision_store,
        _,
    ) = _bind_once(tmp_path, composition)
    adapter.begin_drain(revision, serving_unit_ref=unit_ref)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="lifecycle"):
        bind_colocated_serving_runtime_after_activation(
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
            admission_bind_mode="reopen",
        )


def test_dim13_recovery_reopen_rejected_on_identity_mismatch(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        revision,
        commit,
        serving_store,
        instance_store,
        revision_store,
        _,
    ) = _bind_once(tmp_path, composition)
    bad_attestation = attestation.model_copy(
        update={"materialization_artifact_digest": "sha256:deadbeef"},
    )
    with pytest.raises(ServingRuntimeHandoffError, match="digest does not match handoff"):
        bind_colocated_serving_runtime_after_activation(
            composition=create_reference_production_process_composition(),
            activation_commit=commit,
            attestation=bad_attestation,
            deployment_instance=deployment_instance,
            projection_input=bundle,
            colocated_accessor=accessor,
            physical_adapter=adapter,
            serving_store=serving_store,
            deployment_instance_store=instance_store,
            revision_store=revision_store,
            topology=MaterializationTopology.VENV_BUNDLE,
            artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
            admission_bind_mode="reopen",
        )


def test_dim13_grant_after_physical_authorization_fails_closed(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        _revision,
        commit,
        serving_store,
        instance_store,
        revision_store,
        _,
    ) = _bind_once(tmp_path, composition)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="already authorized"):
        bind_colocated_serving_runtime_after_activation(
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
            admission_bind_mode="grant",
        )


def test_dim13_duplicate_explicit_authorize_still_rejected(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    (
        adapter,
        _accessor,
        unit_ref,
        _attestation,
        deployment_instance,
        _bundle,
        _revision,
        commit,
        serving_store,
        instance_store,
        revision_store,
        updated,
    ) = _bind_once(tmp_path, composition)
    handoff = updated.serving_runtime_handoff
    assert handoff is not None
    with pytest.raises(VenvBundleColocatedDeploymentError, match="already authorized"):
        authorize_colocated_serving_execution_after_activation(
            activation_commit=commit,
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
