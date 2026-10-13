# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1b — ServingRuntimeHandoff identity continuity and GCA admission."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoffError,
    authorize_colocated_serving_execution_after_activation,
    establish_serving_runtime_handoff,
)
from intergrax.agent_distribution.activation import ActivationCommitResult
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.venv_bundle_colocated_serving_unit_runtime_accessor import (
    VenvBundleColocatedServingUnitRuntimeAccessor,
)
from testing_support.agent_distribution.venv_bundle_deployment_fixtures import (
    file_artifact_locator,
    write_minimal_venv_bundle,
)
pytestmark = [pytest.mark.unit]


def _minimal_projection_bundle(tmp_path: Path):
    root = tmp_path / "bundle"
    root.mkdir()
    digest, lock = write_minimal_venv_bundle(root)
    locator = file_artifact_locator(root)
    from intergrax.agent_distribution.runtime_revision import (
        RuntimeRevision,
        RuntimeRevisionState,
    )
    from intergrax.agent_distribution.roster import EffectiveRoster
    from intergrax.applications.contracts.build_context import ApplicationBuildContext
    from intergrax.applications.contracts.manifest import ApplicationManifest

    revision = RuntimeRevision(
        runtime_revision_id="rev-handoff",
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
    adapter.check_readiness(revision, serving_unit_ref=prepared.serving_unit_ref)
    accessor = VenvBundleColocatedServingUnitRuntimeAccessor(adapter)
    resolver = accessor.runtime_agent_factory_resolver_for_unit(
        serving_unit_ref=prepared.serving_unit_ref,
        revision=revision,
        artifact_root=adapter.artifact_root_for_serving_unit(prepared.serving_unit_ref),
    )
    manifest = ApplicationManifest.lab(app_id="app_a", name="App A", agents=())
    bundle = RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=EffectiveRoster(
            application_id="app_a",
            application_environment_id="env-prod",
            manifest_release_id="rel-1",
            entries=(),
        ),
        manifest=manifest,
        build_context=ApplicationBuildContext(manifest=manifest),
        factory_resolver=resolver,
        materialization_artifact_digest=digest,
    )
    attestation = adapter.attest_serving_unit(
        revision,
        request=ServingUnitAttestationRequest(
            application_id="app_a",
            application_environment_id="env-prod",
            runtime_revision_id="rev-handoff",
            serving_unit_ref=prepared.serving_unit_ref,
            materialization_artifact_digest=digest,
            artifact_locator=locator,
        ),
    )
    deployment_instance = DeploymentInstanceRecord(
        application_id="app_a",
        application_environment_id="env-prod",
        runtime_revision_id="rev-handoff",
        serving_unit_ref=prepared.serving_unit_ref,
        instance_state=DeploymentInstanceState.READY,
        readiness_evidence_ref="evidence",
    )
    return (
        adapter,
        accessor,
        prepared.serving_unit_ref,
        attestation,
        deployment_instance,
        bundle,
        revision,
        digest,
    )


def test_handoff_binds_same_resolver_instance(tmp_path: Path) -> None:
    (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        _revision,
        _digest,
    ) = _minimal_projection_bundle(tmp_path)
    composition = create_reference_production_process_composition()
    handoff, updated = establish_serving_runtime_handoff(
        composition=composition,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    assert handoff.runtime_agent_factory_resolver is bundle.factory_resolver
    assert updated.serving_runtime_handoff is handoff
    with pytest.raises(VenvBundleColocatedDeploymentError):
        adapter.admit_unit_execution(unit_ref)
    commit = ActivationCommitResult(
        serving_record=ApplicationEnvironmentServingRecord(
            application_id="app_a",
            application_environment_id="env-prod",
            traffic_serving_revision_id="rev-handoff",
            serving_pointer_revision=1,
        ),
        activated_revision=_revision,
        candidate_instance=deployment_instance,
    )
    authorize_colocated_serving_execution_after_activation(
        activation_commit=commit,
        handoff=handoff,
        admission_authority=adapter,
    )
    adapter.admit_unit_execution(unit_ref)
    adapter.release_unit_execution(unit_ref)


def test_handoff_rejects_resolver_instance_mismatch(tmp_path: Path) -> None:
    (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        revision,
        _digest,
    ) = _minimal_projection_bundle(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    _other_digest, other_lock = write_minimal_venv_bundle(other_root)
    from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
        build_production_runtime_agent_factory_resolver,
    )

    wrong_resolver = build_production_runtime_agent_factory_resolver(
        runtime_revision=revision,
        artifact_root=other_root,
        expected_artifact_digest=_other_digest,
    )
    assert wrong_resolver is not bundle.factory_resolver
    wrong_bundle = replace(bundle, factory_resolver=wrong_resolver)
    composition = create_reference_production_process_composition()
    with pytest.raises(ServingRuntimeHandoffError, match="not the authoritative"):
        establish_serving_runtime_handoff(
            composition=composition,
            attestation=attestation,
            deployment_instance=deployment_instance,
            projection_input=wrong_bundle,
            colocated_accessor=accessor,
            topology=MaterializationTopology.VENV_BUNDLE,
            artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
        )


@pytest.mark.asyncio
async def test_colocated_admission_wrapper_accounts_in_flight(tmp_path: Path) -> None:
    from intergrax.applications._shared.colocated_serving_unit_task_execution import (
        ColocatedServingUnitAdmissionTaskExecutionPort,
    )
    from intergrax.runtime.task.task import Task

    (
        adapter,
        accessor,
        unit_ref,
        attestation,
        deployment_instance,
        bundle,
        _revision,
        _digest,
    ) = _minimal_projection_bundle(tmp_path)
    composition = create_reference_production_process_composition()
    handoff, _updated = establish_serving_runtime_handoff(
        composition=composition,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=bundle,
        colocated_accessor=accessor,
        colocated_execution_admission=accessor,
        topology=MaterializationTopology.VENV_BUNDLE,
        artifact_root=adapter.artifact_root_for_serving_unit(unit_ref),
    )
    commit = ActivationCommitResult(
        serving_record=ApplicationEnvironmentServingRecord(
            application_id="app_a",
            application_environment_id="env-prod",
            traffic_serving_revision_id="rev-handoff",
            serving_pointer_revision=1,
        ),
        activated_revision=_revision,
        candidate_instance=deployment_instance,
    )
    authorize_colocated_serving_execution_after_activation(
        activation_commit=commit,
        handoff=handoff,
        admission_authority=adapter,
    )
    inner = AsyncMock()
    inner.execute = AsyncMock(return_value=object())
    port = ColocatedServingUnitAdmissionTaskExecutionPort(inner, handoff)
    await port.execute(Task(tenant_id="tenant-1", user_id="user-1", message="ping"))
    inner.execute.assert_awaited_once()
