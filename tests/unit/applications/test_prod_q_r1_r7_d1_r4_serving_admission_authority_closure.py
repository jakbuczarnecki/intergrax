# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R4 — atomic serving admission authority closure."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
    ServingExecutionAdmissionGrant,
    run_serving_execution_admission_critical_section,
    verify_current_serving_execution_admission_eligibility,
)
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
from intergrax.applications._shared.serving_runtime_handoff import ServingRuntimeHandoffError
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _canonical_commit_and_stores,
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def _forged_matching_grant(
    *,
    application_id: str,
    application_environment_id: str,
    runtime_revision_id: str,
    serving_unit_ref: str,
    materialization_artifact_digest: str,
    serving_pointer_revision: int,
) -> ServingExecutionAdmissionGrant:
    return ServingExecutionAdmissionGrant(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=runtime_revision_id,
        serving_unit_ref=serving_unit_ref,
        materialization_artifact_digest=materialization_artifact_digest,
        serving_pointer_revision=serving_pointer_revision,
    )


def test_forged_grant_cannot_open_physical_admission(tmp_path: Path) -> None:
    adapter, _, unit_ref, _, deployment_instance, _, revision, digest = _minimal_projection_bundle(
        tmp_path
    )
    forged = _forged_matching_grant(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=digest,
        serving_pointer_revision=1,
    )
    with pytest.raises(ServingExecutionAdmissionError, match="not issued by canonical"):
        adapter.authorize_serving_execution_admission(forged)


def test_canonical_grant_cannot_be_replayed(tmp_path: Path) -> None:
    adapter, _, unit_ref, _, deployment_instance, _, revision, digest = _minimal_projection_bundle(
        tmp_path
    )
    commit, serving_store, instance_store, revision_store = _canonical_commit_and_stores(
        revision,
        deployment_instance,
    )
    def _authorize_once() -> None:
        grant = verify_current_serving_execution_admission_eligibility(
            serving_store=serving_store,
            deployment_instance_store=instance_store,
            revision_store=revision_store,
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            serving_unit_ref=unit_ref,
            materialization_artifact_digest=digest,
        )
        adapter.authorize_serving_execution_admission(grant)

    run_serving_execution_admission_critical_section(serving_store, _authorize_once)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="already authorized"):
        run_serving_execution_admission_critical_section(
            serving_store,
            lambda: adapter.authorize_serving_execution_admission(
                verify_current_serving_execution_admission_eligibility(
                    serving_store=serving_store,
                    deployment_instance_store=instance_store,
                    revision_store=revision_store,
                    application_id=revision.application_id,
                    application_environment_id=revision.application_environment_id,
                    runtime_revision_id=revision.runtime_revision_id,
                    serving_unit_ref=unit_ref,
                    materialization_artifact_digest=digest,
                )
            ),
        )


def test_verify_then_pointer_swap_before_authorize_fails(tmp_path: Path) -> None:
    adapter, accessor, unit_ref, attestation, deployment_instance, bundle, revision, _ = (
        _minimal_projection_bundle(tmp_path)
    )
    commit, serving_store, instance_store, revision_store = _canonical_commit_and_stores(
        revision,
        deployment_instance,
    )
    grant = verify_current_serving_execution_admission_eligibility(
        serving_store=serving_store,
        deployment_instance_store=instance_store,
        revision_store=revision_store,
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        serving_unit_ref=unit_ref,
        materialization_artifact_digest=revision.materialization_artifact_digest or "",
    )
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
    with pytest.raises(ServingExecutionAdmissionError, match="activation critical section"):
        adapter.authorize_serving_execution_admission(grant)


def test_pointer_swap_before_per_execution_admit_rejected(tmp_path: Path) -> None:
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
        admission_authority=adapter,
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
        new_revision_id="rev-race",
        prior_revision_id=revision.runtime_revision_id,
        committed_at=datetime.now(UTC),
    )
    with pytest.raises(ServingRuntimeHandoffError, match="does not target"):
        admission.admit_execution(unit_ref)


def test_direct_adapter_admit_without_authorize_still_closed(tmp_path: Path) -> None:
    adapter, _, unit_ref, _, _, _, _, _ = _minimal_projection_bundle(tmp_path)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        adapter.admit_unit_execution(unit_ref)
