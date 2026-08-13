# © Artur Czarnecki. All rights reserved.

"""AP-9 rollback orchestration tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.agent_distribution.activation import ActivationService, ArtifactRevalidationHook
from intergrax.agent_distribution.deployment import DeploymentInstanceState, FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.errors import (
    RuntimeActivationConflict,
    RuntimeRollbackError,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.activation import FakeRuntimeServingProjectionCoordinator

_APP = "app-a"
_ENV = "env-prod"
_ARTIFACT_N = "sha256:" + ("a" * 64)
_ARTIFACT_N1 = "sha256:" + ("b" * 64)


def _revision(
    revision_id: str,
    *,
    state: RuntimeRevisionState,
    artifact: str,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id=f"lock-{revision_id}",
        materialized_runtime_lock_digest=f"lock-digest-{revision_id}",
        runtime_graph_digest=f"graph-digest-{revision_id}",
        materialization_artifact_digest=artifact,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=datetime.now(UTC) if state is RuntimeRevisionState.ACTIVE else None,
    )


def _bootstrap_active_pair() -> tuple[ActivationService, RuntimeRevisionService, FakeInMemoryRuntimeDeploymentAdapter]:
    state = AgentDistributionStoreState()
    revision_store = InMemoryRuntimeRevisionStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
    projection = FakeRuntimeServingProjectionCoordinator()
    revision_service = RuntimeRevisionService(revision_store)
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_store,
        serving_store=serving_store,
        deployment_adapter=deployment_adapter,
        projection_coordinator=projection,
    )

    for revision_id, artifact in (("rev-n", _ARTIFACT_N), ("rev-n1", _ARTIFACT_N1)):
        candidate = _revision(revision_id, state=RuntimeRevisionState.CANDIDATE, artifact=artifact)
        revision_service.persist_candidate_revision(candidate)
        validated = _revision(revision_id, state=RuntimeRevisionState.VALIDATED, artifact=artifact)
        revision_service.mark_validated(revision_id, validated_revision=validated)

    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n",
        artifact_locator=f"artifact://{_ARTIFACT_N}",
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=_ARTIFACT_N,
    )
    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        artifact_locator=f"artifact://{_ARTIFACT_N1}",
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        expected_prior_traffic_revision_id="rev-n",
        expected_serving_pointer_revision=1,
        expected_artifact_digest=_ARTIFACT_N1,
    )
    return activation, revision_service, deployment_adapter


def test_rollback_uses_existing_prior_artifact_without_rebuild() -> None:
    activation, revision_service, deployment_adapter = _bootstrap_active_pair()
    prepare_before = deployment_adapter.prepare_count.get("rev-n", 0)
    rolled = activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    assert rolled.value.serving_record.traffic_serving_revision_id == "rev-n"
    assert rolled.value.restored_revision.materialization_artifact_digest == _ARTIFACT_N
    assert deployment_adapter.prepare_count.get("rev-n", 0) == prepare_before


def test_rollback_cas_swaps_serving_pointer() -> None:
    activation, _, _ = _bootstrap_active_pair()
    rolled = activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    assert rolled.value.serving_record.prior_traffic_revision_id == "rev-n1"
    assert rolled.value.serving_record.serving_pointer_revision == 3


def test_rollback_restored_revision_serving_and_prior_draining() -> None:
    activation, revision_service, _ = _bootstrap_active_pair()
    activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    assert revision_service.get_active_revision(_ENV).runtime_revision_id == "rev-n"
    n1_instance = activation._deployment_instance_store.get_instance(_ENV, "rev-n1")
    assert n1_instance is not None
    assert n1_instance.instance_state is DeploymentInstanceState.DRAINING


def test_rollback_missing_prior_artifact_fails_closed() -> None:
    activation, _, _ = _bootstrap_active_pair()
    activation._artifact_revalidation = ArtifactRevalidationHook(
        validate=lambda _revision: (_ for _ in ()).throw(RuntimeRollbackError("artifact missing"))
    )
    with pytest.raises(RuntimeRollbackError):
        activation.rollback(
            application_id=_APP,
            application_environment_id=_ENV,
            expected_current_traffic_revision_id="rev-n1",
            expected_serving_pointer_revision=2,
        )


def test_stale_rollback_pointer_conflict() -> None:
    activation, _, _ = _bootstrap_active_pair()
    with pytest.raises(RuntimeActivationConflict):
        activation.rollback(
            application_id=_APP,
            application_environment_id=_ENV,
            expected_current_traffic_revision_id="rev-n",
            expected_serving_pointer_revision=2,
        )


def test_rollback_retry_idempotent() -> None:
    activation, _, _ = _bootstrap_active_pair()
    first = activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    second = activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    assert first.value.serving_record.traffic_serving_revision_id == second.value.serving_record.traffic_serving_revision_id


def test_rollback_failure_retains_authoritative_serving_pointer() -> None:
    activation, _, _ = _bootstrap_active_pair()
    activation._artifact_revalidation = ArtifactRevalidationHook(
        validate=lambda _revision: (_ for _ in ()).throw(RuntimeRollbackError("trust invalid"))
    )
    with pytest.raises(RuntimeRollbackError):
        activation.rollback(
            application_id=_APP,
            application_environment_id=_ENV,
            expected_current_traffic_revision_id="rev-n1",
            expected_serving_pointer_revision=2,
        )
    serving = activation._serving_store.get_serving_record(_ENV)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-n1"


def test_post_cutover_failure_triggers_rollback() -> None:
    activation, revision_service, _ = _bootstrap_active_pair()
    result = activation.mark_post_cutover_failure(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        failure_evidence_ref="health:failed",
    )
    assert result.value is not None
    assert result.value.serving_record.traffic_serving_revision_id == "rev-n"
    assert revision_service.get_active_revision(_ENV).runtime_revision_id == "rev-n"


def test_rollback_reuses_draining_instance_without_prepare() -> None:
    activation, _, deployment_adapter = _bootstrap_active_pair()
    n_instance = activation._deployment_instance_store.get_instance(_ENV, "rev-n")
    assert n_instance is not None
    draining = n_instance.model_copy(
        update={
            "instance_state": DeploymentInstanceState.DRAINING,
            "record_revision": n_instance.record_revision + 1,
        }
    )
    activation._deployment_instance_store.update_instance(
        draining,
        expected_state=n_instance.instance_state,
        expected_record_revision=n_instance.record_revision,
    )
    prepare_before = deployment_adapter.prepare_count.get("rev-n", 0)
    activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    assert deployment_adapter.prepare_count.get("rev-n", 0) == prepare_before
