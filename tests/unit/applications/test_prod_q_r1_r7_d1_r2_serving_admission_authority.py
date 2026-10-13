# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R2 — serving admission authority and runtime identity corrective."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from intergrax.agent_distribution.activation import (
    ActivationCommitResult,
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.colocated_serving_runtime_composition import (
    bind_colocated_serving_runtime_after_activation,
)
from intergrax.applications._shared.colocated_serving_unit_task_execution import (
    ColocatedServingUnitAdmissionTaskExecutionPort,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoffError,
    authorize_colocated_serving_execution_after_activation,
    establish_serving_runtime_handoff,
)
from intergrax.applications._shared.venv_bundle_colocated_serving_unit_runtime_accessor import (
    VenvBundleColocatedServingUnitRuntimeAccessor,
)
from intergrax.runtime.task.task import Task
from testing_support.agent_distribution.activation_materialization_fixtures import (
    persist_materialization_for_revision,
)
from testing_support.agent_distribution.venv_bundle_deployment_fixtures import (
    file_artifact_locator,
    write_minimal_venv_bundle,
)

from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def test_ready_handoff_keeps_execution_admission_closed(tmp_path: Path) -> None:
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
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    assert handoff is not None
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        adapter.admit_unit_execution(unit_ref)


def test_authorize_denied_without_activation_commit(tmp_path: Path) -> None:
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
    stale_commit = ActivationCommitResult(
        serving_record=ApplicationEnvironmentServingRecord(
            application_id="app_a",
            application_environment_id="env-prod",
            traffic_serving_revision_id="other-revision",
            serving_pointer_revision=1,
        ),
        activated_revision=revision,
        candidate_instance=deployment_instance,
    )
    with pytest.raises(ServingRuntimeHandoffError, match="serving pointer"):
        authorize_colocated_serving_execution_after_activation(
            activation_commit=stale_commit,
            handoff=handoff,
            admission_authority=adapter,
        )


def test_bind_after_activation_opens_admission(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    composition = create_reference_production_process_composition()
    commit = ActivationCommitResult(
        serving_record=ApplicationEnvironmentServingRecord(
            application_id="app_a",
            application_environment_id="env-prod",
            traffic_serving_revision_id=revision.runtime_revision_id,
            serving_pointer_revision=1,
        ),
        activated_revision=revision,
        candidate_instance=deployment_instance,
    )
    updated = bind_colocated_serving_runtime_after_activation(
        composition=composition,
        activation_commit=commit,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        admission_authority=adapter,
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    assert updated.serving_runtime_handoff is not None
    adapter.admit_unit_execution(unit_ref)
    adapter.release_unit_execution(unit_ref)


def test_duplicate_admission_authorization_rejected(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    commit = _commit_for(revision, deployment_instance)
    updated = bind_colocated_serving_runtime_after_activation(
        composition=create_reference_production_process_composition(),
        activation_commit=commit,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        admission_authority=adapter,
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    handoff = updated.serving_runtime_handoff
    assert handoff is not None
    with pytest.raises(VenvBundleColocatedDeploymentError, match="already authorized"):
        authorize_colocated_serving_execution_after_activation(
            activation_commit=commit,
            handoff=handoff,
            admission_authority=adapter,
        )


def test_accessor_requires_physical_runtime_binding(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    digest, lock = write_minimal_venv_bundle(root)
    locator = file_artifact_locator(root)
    from intergrax.agent_distribution.runtime_revision import RuntimeRevision

    revision = RuntimeRevision(
        runtime_revision_id="rev-bind",
        application_id="app_a",
        application_environment_id="env-prod",
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster",
        materialized_runtime_lock_id=lock.lock_id,
        materialized_runtime_lock_digest=lock.lock_digest,
        runtime_graph_digest="graph",
        materialization_artifact_digest=digest,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.VALIDATED,
    )
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    prepared = adapter.prepare(revision, artifact_locator=locator)
    accessor = VenvBundleColocatedServingUnitRuntimeAccessor(adapter)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="invalid"):
        accessor.runtime_agent_factory_resolver_for_unit(
            serving_unit_ref=prepared.serving_unit_ref,
            revision=revision,
            artifact_root=adapter.artifact_root_for_serving_unit(prepared.serving_unit_ref),
        )


def test_physical_binding_warmup_evidence_reaches_accessor_cache(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, _att, _dep, bundle, revision, _ = _minimal_projection_bundle(
        tmp_path
    )
    binding = adapter.colocated_unit_runtime_binding(unit_ref)
    resolver = accessor.runtime_agent_factory_resolver_for_unit(
        serving_unit_ref=unit_ref,
        revision=revision,
        artifact_root=binding.artifact_root,
    )
    assert resolver is bundle.factory_resolver
    assert "venv-colocated-runtime:" in binding.physical_runtime_warmup_evidence


def _commit_for(revision, deployment_instance) -> ActivationCommitResult:
    return ActivationCommitResult(
        serving_record=ApplicationEnvironmentServingRecord(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            traffic_serving_revision_id=revision.runtime_revision_id,
            serving_pointer_revision=1,
        ),
        activated_revision=revision,
        candidate_instance=deployment_instance,
    )


@pytest.mark.asyncio
async def test_gca_admission_wrapper_blocked_under_drain(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    composition = bind_colocated_serving_runtime_after_activation(
        composition=create_reference_production_process_composition(),
        activation_commit=_commit_for(revision, deployment_instance),
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        admission_authority=adapter,
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    handoff = composition.serving_runtime_handoff
    assert handoff is not None
    adapter.begin_drain(revision, serving_unit_ref=unit_ref)
    inner = AsyncMock()
    port = ColocatedServingUnitAdmissionTaskExecutionPort(inner, handoff)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        await port.execute(Task(tenant_id="t", user_id="u", message="x"))


@pytest.mark.asyncio
async def test_gca_admission_wrapper_releases_on_exception(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    composition = bind_colocated_serving_runtime_after_activation(
        composition=create_reference_production_process_composition(),
        activation_commit=_commit_for(revision, deployment_instance),
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        admission_authority=adapter,
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    handoff = composition.serving_runtime_handoff
    assert handoff is not None
    inner = AsyncMock()
    inner.execute = AsyncMock(side_effect=RuntimeError("task failed"))
    port = ColocatedServingUnitAdmissionTaskExecutionPort(inner, handoff)
    with pytest.raises(RuntimeError, match="task failed"):
        await port.execute(Task(tenant_id="t", user_id="u", message="x"))
    adapter.admit_unit_execution(unit_ref)
    adapter.release_unit_execution(unit_ref)


def test_activation_commit_then_bind_end_to_end(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, digest = (
        _minimal_projection_bundle(tmp_path)
    )
    locator = file_artifact_locator(tmp_path / "bundle")
    state = AgentDistributionStoreState()
    revision_store = InMemoryRuntimeRevisionStore(state)
    revision_service = RuntimeRevisionService(revision_store)
    candidate = revision.model_copy(update={"revision_state": RuntimeRevisionState.CANDIDATE})
    revision_service.persist_candidate_revision(candidate)
    revision_service.mark_validated(
        revision.runtime_revision_id,
        validated_revision=revision,
    )
    materialization_store = InMemoryRuntimeMaterializationStore(state)
    persist_materialization_for_revision(
        materialization_store,
        revision,
        artifact_locator=locator,
    )
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=InMemoryDeploymentInstanceStore(state),
        serving_store=InMemoryApplicationEnvironmentServingStore(state),
        activation_store=InMemoryApplicationEnvironmentActivationStore(state),
        deployment_adapter=adapter,
        projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        materialization_store=materialization_store,
    )
    activation.prepare_candidate(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        artifact_locator=locator,
    )
    committed = activation.commit_activation(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=digest,
    )
    composition = bind_colocated_serving_runtime_after_activation(
        composition=create_reference_production_process_composition(),
        activation_commit=committed.value,
        attestation=attestation,
        deployment_instance=committed.value.candidate_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        admission_authority=adapter,
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    assert composition.serving_runtime_handoff is not None
    adapter.admit_unit_execution(unit_ref)


def test_handoff_failure_leaves_composition_unchanged(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    wrong_bundle = replace(bundle, factory_resolver=None)
    composition = create_reference_production_process_composition()
    with pytest.raises(ServingRuntimeHandoffError, match="factory_resolver"):
        establish_serving_runtime_handoff(
            composition=composition,
            attestation=attestation,
            deployment_instance=deployment_instance,
            projection_input=wrong_bundle,
            colocated_accessor=accessor,
            topology=MaterializationTopology.VENV_BUNDLE,
            artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
        )
    assert composition.serving_runtime_handoff is None
