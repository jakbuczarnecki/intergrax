# © Artur Czarnecki. All rights reserved.

"""AP-9 activation orchestration tests — PREPARE, READY, COMMIT, DRAIN, projection."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    DrainActionOnTimeout,
    DrainPolicy,
    FakeInMemoryRuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.errors import (
    RuntimeActivationConflict,
    RuntimeActivationError,
    RuntimeDrainError,
    RuntimeReadinessError,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentActivationStore,
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

_APP = "app-a"
_ENV_PROD = "env-prod"
_ENV_STAGING = "env-staging"
_ARTIFACT = "sha256:" + ("d" * 64)
_LOCATOR = f"artifact://{_ARTIFACT}"


def _revision(
    revision_id: str,
    *,
    environment: str = _ENV_PROD,
    state: RuntimeRevisionState = RuntimeRevisionState.VALIDATED,
    artifact_digest: str = _ARTIFACT,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_environment_id=environment,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="lock-digest",
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest=artifact_digest,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=datetime.now(UTC) if state is RuntimeRevisionState.ACTIVE else None,
    )


def _persist_validated(
    revision_service: RuntimeRevisionService,
    revision_id: str,
    *,
    environment: str = _ENV_PROD,
    artifact_digest: str = _ARTIFACT,
) -> RuntimeRevision:
    candidate = _revision(
        revision_id,
        environment=environment,
        state=RuntimeRevisionState.CANDIDATE,
        artifact_digest=artifact_digest,
    )
    revision_service.persist_candidate_revision(candidate)
    validated = _revision(revision_id, environment=environment, artifact_digest=artifact_digest)
    revision_service.mark_validated(revision_id, validated_revision=validated)
    return validated


class ActivationHarness:
    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self.state = state or AgentDistributionStoreState()
        self.revision_store = InMemoryRuntimeRevisionStore(self.state)
        self.deployment_store = InMemoryDeploymentInstanceStore(self.state)
        self.serving_store = InMemoryApplicationEnvironmentServingStore(self.state)
        self.activation_store = InMemoryApplicationEnvironmentActivationStore(self.state)
        self.deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
        self.projection = FakeRuntimeServingProjectionCoordinator()
        self.revision_service = RuntimeRevisionService(self.revision_store)
        self.activation = ActivationService(
            revision_store=self.revision_store,
            deployment_instance_store=self.deployment_store,
            serving_store=self.serving_store,
            activation_store=self.activation_store,
            deployment_adapter=self.deployment_adapter,
            projection_coordinator=self.projection,
        )

    def seed_validated(
        self,
        revision_id: str,
        *,
        environment: str = _ENV_PROD,
        artifact_digest: str = _ARTIFACT,
    ) -> RuntimeRevision:
        return _persist_validated(
            self.revision_service,
            revision_id,
            environment=environment,
            artifact_digest=artifact_digest,
        )

    def prepare(self, revision_id: str, *, environment: str = _ENV_PROD):
        return self.activation.prepare_candidate(
            application_id=_APP,
            application_environment_id=environment,
            runtime_revision_id=revision_id,
            artifact_locator=_LOCATOR,
        )

    def commit(
        self,
        revision_id: str,
        *,
        environment: str = _ENV_PROD,
        expected_prior: str | None = None,
        pointer: int = 0,
    ):
        return self.activation.commit_activation(
            application_id=_APP,
            application_environment_id=environment,
            runtime_revision_id=revision_id,
            expected_prior_traffic_revision_id=expected_prior,
            expected_serving_pointer_revision=pointer,
            expected_artifact_digest=_ARTIFACT,
        )


def test_prepare_validated_candidate_becomes_ready() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    prepared = harness.prepare("rev-1")
    assert prepared.value.instance_state is DeploymentInstanceState.READY
    assert prepared.value.readiness_evidence_ref is not None
    assert prepared.value.serving_unit_ref is not None


def test_prepare_rejects_non_validated_revision() -> None:
    harness = ActivationHarness()
    harness.revision_service.persist_candidate_revision(
        _revision("rev-1", state=RuntimeRevisionState.CANDIDATE)
    )
    with pytest.raises(RuntimeActivationError):
        harness.prepare("rev-1")


def test_readiness_failure_leaves_serving_pointer_unchanged() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.deployment_adapter.fail_readiness("rev-1")
    with pytest.raises(RuntimeReadinessError):
        harness.prepare("rev-1")
    assert harness.serving_store.get_serving_record(_ENV_PROD) is None
    instance = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert instance is not None
    assert instance.instance_state is DeploymentInstanceState.FAILED


def test_deploy_failure_leaves_active_untouched() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.deployment_adapter.fail_prepare("rev-2")
    with pytest.raises(RuntimeError):
        harness.prepare("rev-2")
    serving = harness.serving_store.get_serving_record(_ENV_PROD)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-1"


def test_ready_instance_bound_to_exact_revision_and_env() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    ready = harness.prepare("rev-1").value
    assert ready.runtime_revision_id == "rev-1"
    assert ready.application_environment_id == _ENV_PROD


def test_artifact_identity_mismatch_rejected_at_commit() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    with pytest.raises(RuntimeActivationError):
        harness.activation.commit_activation(
            application_id=_APP,
            application_environment_id=_ENV_PROD,
            runtime_revision_id="rev-1",
            expected_prior_traffic_revision_id=None,
            expected_serving_pointer_revision=0,
            expected_artifact_digest="sha256:" + ("e" * 64),
        )


def test_ready_candidate_activates_traffic_pointer_and_states() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    committed = harness.commit("rev-1")
    serving = committed.value.serving_record
    assert serving.traffic_serving_revision_id == "rev-1"
    assert serving.serving_pointer_revision == 1
    assert harness.state.revisions["rev-1"].revision_state is RuntimeRevisionState.ACTIVE
    instance = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert instance is not None
    assert instance.instance_state is DeploymentInstanceState.SERVING


def test_second_activation_supersedes_prior_and_drains() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    committed = harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    assert committed.value.serving_record.traffic_serving_revision_id == "rev-2"
    assert committed.value.serving_record.prior_traffic_revision_id == "rev-1"
    assert harness.state.revisions["rev-1"].revision_state is RuntimeRevisionState.SUPERSEDED
    prior_instance = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert prior_instance is not None
    assert prior_instance.instance_state is DeploymentInstanceState.DRAINING


def test_stale_serving_pointer_cas_conflict() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    with pytest.raises(RuntimeActivationConflict):
        harness.commit("rev-2", expected_prior="rev-missing", pointer=1)


def test_two_concurrent_activation_attempts_one_wins() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.seed_validated("rev-3")
    harness.prepare("rev-2")
    harness.prepare("rev-3")
    results: list[str] = []
    errors: list[Exception] = []

    def attempt(revision_id: str) -> None:
        try:
            harness.commit(revision_id, expected_prior="rev-1", pointer=1)
            results.append(revision_id)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=attempt, args=("rev-2",))
    t2 = threading.Thread(target=attempt, args=("rev-3",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeActivationConflict)


def test_commit_rejects_candidate_not_ready() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-1")


def test_failure_before_commit_leaves_prior_serving() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.deployment_adapter.fail_readiness("rev-2")
    with pytest.raises(RuntimeReadinessError):
        harness.prepare("rev-2")
    serving = harness.serving_store.get_serving_record(_ENV_PROD)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-1"


def test_draining_prior_completes_to_stopped() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    prior = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert prior is not None
    harness.deployment_adapter.complete_drain(prior.serving_unit_ref or "")
    completed = harness.activation.complete_drain(
        application_environment_id=_ENV_PROD,
        runtime_revision_id="rev-1",
        policy=DrainPolicy(timeout_seconds=30.0),
    )
    assert completed.value.instance.instance_state is DeploymentInstanceState.STOPPED


def test_drain_timeout_returns_typed_outcome() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    prior = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert prior is not None
    harness.deployment_adapter.force_drain_timeout(prior.serving_unit_ref or "")
    with pytest.raises(RuntimeDrainError):
        harness.activation.complete_drain(
            application_environment_id=_ENV_PROD,
            runtime_revision_id="rev-1",
            policy=DrainPolicy(
                timeout_seconds=1.0,
                action_on_timeout=DrainActionOnTimeout.MARK_FAILED,
            ),
        )


def test_drain_timeout_does_not_route_traffic_back_to_old_revision() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    prior = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert prior is not None
    harness.deployment_adapter.force_drain_timeout(prior.serving_unit_ref or "")
    with pytest.raises(RuntimeDrainError):
        harness.activation.complete_drain(
            application_environment_id=_ENV_PROD,
            runtime_revision_id="rev-1",
            policy=DrainPolicy(timeout_seconds=1.0),
        )
    serving = harness.serving_store.get_serving_record(_ENV_PROD)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-2"


def test_projection_prepared_before_traffic_commit() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    assert "rev-1" in harness.projection.prepared
    assert harness.projection.ready_tokens["rev-1"] == "projection-ready:rev-1"


def test_traffic_switch_blocked_when_projection_prepare_fails() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.projection.fail_prepare("rev-1")
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-1")
    assert harness.serving_store.get_serving_record(_ENV_PROD) is None


def test_post_commit_projection_ready_without_fallible_publication() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    assert "rev-1" in harness.projection.ready_tokens


def test_activation_prod_leaves_staging_untouched() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-prod", environment=_ENV_PROD)
    harness.seed_validated("rev-staging", environment=_ENV_STAGING)
    harness.prepare("rev-prod", environment=_ENV_PROD)
    harness.commit("rev-prod", environment=_ENV_PROD)
    assert harness.serving_store.get_serving_record(_ENV_STAGING) is None


def test_activation_app_a_leaves_app_b_untouched() -> None:
    state = AgentDistributionStoreState()
    harness_a = ActivationHarness(state)
    harness_b = ActivationHarness(state)
    harness_a.seed_validated("rev-a")
    harness_b.seed_validated("rev-b", environment="env-b")
    harness_a.prepare("rev-a")
    harness_a.commit("rev-a")
    assert harness_b.serving_store.get_serving_record("env-b") is None


def test_prepare_idempotent_for_same_ready_candidate() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    first = harness.prepare("rev-1")
    second = harness.prepare("rev-1")
    assert first.value.serving_unit_ref == second.value.serving_unit_ref
    assert harness.deployment_adapter.prepare_count.get("rev-1", 0) == 1


def test_commit_idempotent_when_already_active() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    first = harness.commit("rev-1")
    second = harness.commit("rev-1")
    assert second.value.serving_record.traffic_serving_revision_id == "rev-1"
    assert first.value.serving_record.serving_pointer_revision == second.value.serving_record.serving_pointer_revision


def test_no_live_mutation_boundaries_enforced_by_orchestration_only() -> None:
    """AP-9 orchestrates immutable revisions — no roster/install/binding mutation APIs."""
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    assert not hasattr(harness.activation, "mutate_effective_roster")
    assert not hasattr(harness.activation, "mutate_installation")


def _snapshot_activation_state(harness: ActivationHarness) -> dict[str, object]:
    serving = harness.serving_store.get_serving_record(_ENV_PROD)
    return {
        "serving": serving.model_dump() if serving else None,
        "revisions": {k: v.model_dump() for k, v in harness.state.revisions.items()},
        "instances": {str(k): v.model_dump() for k, v in harness.state.deployment_instances.items()},
    }


def test_atomic_commit_revision_precondition_failure_leaves_pointer_unchanged() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    harness.state.revisions["rev-2"] = harness.state.revisions["rev-2"].model_copy(
        update={"revision_state": RuntimeRevisionState.CANDIDATE}
    )
    before = _snapshot_activation_state(harness)
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    assert _snapshot_activation_state(harness) == before


def test_atomic_commit_candidate_deployment_conflict_leaves_pointer_unchanged() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    instance = harness.deployment_store.get_instance(_ENV_PROD, "rev-2")
    assert instance is not None
    harness.deployment_store.persist_instance(
        instance.model_copy(update={"instance_state": DeploymentInstanceState.PREPARING})
    )
    before = _snapshot_activation_state(harness)
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    assert _snapshot_activation_state(harness) == before


def test_atomic_commit_prior_deployment_conflict_leaves_pointer_unchanged() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    prior = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert prior is not None
    harness.deployment_store.persist_instance(
        prior.model_copy(update={"instance_state": DeploymentInstanceState.READY})
    )
    before = _snapshot_activation_state(harness)
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    assert _snapshot_activation_state(harness) == before


def test_stale_pointer_cas_leaves_revision_and_deployment_unmutated() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    before = _snapshot_activation_state(harness)
    with pytest.raises(RuntimeActivationConflict):
        harness.commit("rev-2", expected_prior="rev-1", pointer=0)
    assert _snapshot_activation_state(harness) == before


def test_projection_prepare_failure_leaves_no_durable_mutations() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.projection.fail_prepare("rev-1")
    before = _snapshot_activation_state(harness)
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-1")
    assert _snapshot_activation_state(harness) == before


def test_atomic_store_internal_precondition_failure_leaves_no_partial_writes() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    before = _snapshot_activation_state(harness)
    harness.activation_store._fail_atomic_commit_activation = True
    with pytest.raises(RuntimeActivationConflict):
        harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    assert _snapshot_activation_state(harness) == before


def test_successful_atomic_commit_sets_consistent_authority_bundle() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    serving = harness.serving_store.get_serving_record(_ENV_PROD)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-2"
    assert harness.state.revisions["rev-2"].revision_state is RuntimeRevisionState.ACTIVE
    assert harness.state.revisions["rev-1"].revision_state is RuntimeRevisionState.SUPERSEDED
    candidate = harness.deployment_store.get_instance(_ENV_PROD, "rev-2")
    prior = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert candidate is not None and candidate.instance_state is DeploymentInstanceState.SERVING
    assert prior is not None and prior.instance_state is DeploymentInstanceState.DRAINING


def test_begin_drain_failure_after_commit_retains_n_plus_one_authority() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    prior = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert prior is not None
    harness.deployment_adapter.fail_begin_drain(prior.serving_unit_ref or "")
    with pytest.raises(RuntimeDrainError):
        harness.commit("rev-2", expected_prior="rev-1", pointer=1)
    serving = harness.serving_store.get_serving_record(_ENV_PROD)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-2"
    assert harness.state.revisions["rev-2"].revision_state is RuntimeRevisionState.ACTIVE
    candidate = harness.deployment_store.get_instance(_ENV_PROD, "rev-2")
    assert candidate is not None and candidate.instance_state is DeploymentInstanceState.SERVING
    drained_prior = harness.deployment_store.get_instance(_ENV_PROD, "rev-1")
    assert drained_prior is not None
    assert drained_prior.instance_state is DeploymentInstanceState.DRAINING
    assert drained_prior.failure_evidence_ref is not None
