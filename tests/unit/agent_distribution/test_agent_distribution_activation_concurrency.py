# © Artur Czarnecki. All rights reserved.

"""AP-9-FIX-2 cross-service activation concurrency tests."""

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
    FakeInMemoryRuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.errors import (
    RuntimeActivationConflict,
    RuntimeRevisionConflict,
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
from intergrax.agent_distribution.application_environment_identity import (
    ApplicationEnvironmentIdentity,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService

_APP = "app-a"
_ENV = "env-prod"
_ARTIFACT = "sha256:" + ("d" * 64)
_LOCATOR = f"artifact://{_ARTIFACT}"


def _scope() -> ApplicationEnvironmentIdentity:
    return ApplicationEnvironmentIdentity(
        application_id=_APP,
        application_environment_id=_ENV,
    )


def _revision(
    revision_id: str,
    *,
    state: RuntimeRevisionState = RuntimeRevisionState.VALIDATED,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="lock-digest",
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest=_ARTIFACT,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=datetime.now(UTC)
        if state is RuntimeRevisionState.ACTIVE
        else None,
    )


class SharedActivationHarness:
    """Two independent store/service instance pairs over one shared state."""

    def __init__(self) -> None:
        self.state = AgentDistributionStoreState()
        self.revision_store_a = InMemoryRuntimeRevisionStore(self.state)
        self.deployment_store_a = InMemoryDeploymentInstanceStore(self.state)
        self.serving_store_a = InMemoryApplicationEnvironmentServingStore(self.state)
        self.activation_store_a = InMemoryApplicationEnvironmentActivationStore(
            self.state
        )
        self.revision_store_b = InMemoryRuntimeRevisionStore(self.state)
        self.deployment_store_b = InMemoryDeploymentInstanceStore(self.state)
        self.serving_store_b = InMemoryApplicationEnvironmentServingStore(self.state)
        self.activation_store_b = InMemoryApplicationEnvironmentActivationStore(
            self.state
        )
        self.deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
        self.projection = FakeRuntimeServingProjectionCoordinator()
        self.revision_service_a = RuntimeRevisionService(self.revision_store_a)
        self.revision_service_b = RuntimeRevisionService(self.revision_store_b)
        self.activation_a = ActivationService(
            revision_store=self.revision_store_a,
            deployment_instance_store=self.deployment_store_a,
            serving_store=self.serving_store_a,
            activation_store=self.activation_store_a,
            deployment_adapter=self.deployment_adapter,
            projection_coordinator=self.projection,
        )
        self.activation_b = ActivationService(
            revision_store=self.revision_store_b,
            deployment_instance_store=self.deployment_store_b,
            serving_store=self.serving_store_b,
            activation_store=self.activation_store_b,
            deployment_adapter=self.deployment_adapter,
            projection_coordinator=self.projection,
        )

    def seed_validated(self, revision_id: str) -> RuntimeRevision:
        candidate = _revision(revision_id, state=RuntimeRevisionState.CANDIDATE)
        self.revision_service_a.persist_candidate_revision(candidate)
        validated = _revision(revision_id)
        self.revision_service_a.mark_validated(
            revision_id, validated_revision=validated
        )
        return validated

    def prepare(self, revision_id: str) -> None:
        self.activation_a.prepare_candidate(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision_id,
            artifact_locator=_LOCATOR,
        )

    def commit_a(
        self,
        revision_id: str,
        *,
        expected_prior: str | None = None,
        pointer: int = 0,
    ) -> None:
        self.activation_a.commit_activation(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision_id,
            expected_prior_traffic_revision_id=expected_prior,
            expected_serving_pointer_revision=pointer,
            expected_artifact_digest=_ARTIFACT,
        )

    def commit_b(
        self,
        revision_id: str,
        *,
        expected_prior: str | None = None,
        pointer: int = 0,
    ) -> None:
        self.activation_b.commit_activation(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision_id,
            expected_prior_traffic_revision_id=expected_prior,
            expected_serving_pointer_revision=pointer,
            expected_artifact_digest=_ARTIFACT,
        )


def _snapshot_state(state: AgentDistributionStoreState) -> dict[str, object]:
    serving = state.serving_records.get(_scope())
    return {
        "serving": serving.model_dump() if serving else None,
        "revisions": {k: v.model_dump() for k, v in state.revisions.items()},
        "instances": {
            str(k): v.model_dump() for k, v in state.deployment_instances.items()
        },
        "active": dict(state.active_revision_by_scope),
    }


def _bootstrap_active(
    harness: SharedActivationHarness, revision_id: str = "rev-1"
) -> None:
    harness.seed_validated(revision_id)
    harness.prepare(revision_id)
    harness.commit_a(revision_id)


def _pause_at_validation(
    activation_store: InMemoryApplicationEnvironmentActivationStore,
) -> tuple[threading.Event, threading.Event]:
    at_validation = threading.Event()
    continue_commit = threading.Event()

    def sync_hook() -> None:
        at_validation.set()
        continue_commit.wait(timeout=5.0)

    activation_store._sync_after_validation = sync_hook
    return at_validation, continue_commit


def test_two_activation_services_one_conflicting_activation_wins() -> None:
    harness = SharedActivationHarness()
    _bootstrap_active(harness)
    harness.seed_validated("rev-2")
    harness.seed_validated("rev-3")
    harness.prepare("rev-2")
    harness.prepare("rev-3")
    winners: list[str] = []
    errors: list[Exception] = []

    def attempt(service: str, revision_id: str) -> None:
        try:
            if service == "a":
                harness.commit_a(revision_id, expected_prior="rev-1", pointer=1)
            else:
                harness.commit_b(revision_id, expected_prior="rev-1", pointer=1)
            winners.append(revision_id)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=attempt, args=("a", "rev-2"))
    t2 = threading.Thread(target=attempt, args=("b", "rev-3"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(winners) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeActivationConflict)
    serving = harness.state.serving_records[_scope()]
    assert serving.traffic_serving_revision_id == winners[0]


def test_deployment_mutation_blocked_during_atomic_commit() -> None:
    harness = SharedActivationHarness()
    _bootstrap_active(harness)
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    at_validation, continue_commit = _pause_at_validation(harness.activation_store_a)
    mutation_started = threading.Event()
    mutation_finished = threading.Event()
    commit_error: list[Exception] = []

    def commit() -> None:
        try:
            harness.commit_a("rev-2", expected_prior="rev-1", pointer=1)
        except Exception as exc:
            commit_error.append(exc)

    def mutate_deployment() -> None:
        at_validation.wait(timeout=5.0)
        mutation_started.set()
        instance = harness.deployment_store_b.get_instance(_APP, _ENV, "rev-2")
        assert instance is not None
        with pytest.raises(RuntimeActivationConflict):
            harness.deployment_store_b.update_instance(
                instance.model_copy(
                    update={"instance_state": DeploymentInstanceState.PREPARING}
                ),
                expected_state=DeploymentInstanceState.SERVING,
                expected_record_revision=instance.record_revision,
            )
        mutation_finished.set()

    t_commit = threading.Thread(target=commit)
    t_mutate = threading.Thread(target=mutate_deployment)
    t_commit.start()
    t_mutate.start()
    at_validation.wait(timeout=5.0)
    mutation_started.wait(timeout=5.0)
    assert not mutation_finished.is_set()
    continue_commit.set()
    t_commit.join(timeout=5.0)
    t_mutate.join(timeout=5.0)
    assert mutation_finished.is_set()
    assert not commit_error
    assert (
        harness.state.deployment_instances[(_scope(), "rev-2")].instance_state
        is DeploymentInstanceState.SERVING
    )


def test_runtime_revision_mutation_blocked_during_atomic_commit() -> None:
    harness = SharedActivationHarness()
    _bootstrap_active(harness)
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    at_validation, continue_commit = _pause_at_validation(harness.activation_store_a)
    mutation_started = threading.Event()
    mutation_finished = threading.Event()

    def commit() -> None:
        harness.commit_a("rev-2", expected_prior="rev-1", pointer=1)

    def mutate_revision() -> None:
        at_validation.wait(timeout=5.0)
        mutation_started.set()
        demoted = harness.state.revisions["rev-2"].model_copy(
            update={"revision_state": RuntimeRevisionState.CANDIDATE}
        )
        with pytest.raises(RuntimeRevisionConflict):
            harness.revision_store_b.persist_candidate_revision(
                demoted,
                expected_revision_state=RuntimeRevisionState.VALIDATED,
            )
        mutation_finished.set()

    t_commit = threading.Thread(target=commit)
    t_mutate = threading.Thread(target=mutate_revision)
    t_commit.start()
    t_mutate.start()
    at_validation.wait(timeout=5.0)
    mutation_started.wait(timeout=5.0)
    assert not mutation_finished.is_set()
    continue_commit.set()
    t_commit.join(timeout=5.0)
    t_mutate.join(timeout=5.0)
    assert mutation_finished.is_set()
    assert (
        harness.state.revisions["rev-2"].revision_state is RuntimeRevisionState.ACTIVE
    )


def test_serving_pointer_mutation_blocked_during_atomic_commit() -> None:
    harness = SharedActivationHarness()
    _bootstrap_active(harness)
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    at_validation, continue_commit = _pause_at_validation(harness.activation_store_a)
    mutation_started = threading.Event()
    mutation_finished = threading.Event()
    swap_error: list[Exception] = []

    def commit() -> None:
        harness.commit_a("rev-2", expected_prior="rev-1", pointer=1)

    def swap_serving() -> None:
        at_validation.wait(timeout=5.0)
        mutation_started.set()
        try:
            harness.serving_store_b.atomic_swap_serving_revision(
                application_id=_APP,
                application_environment_id=_ENV,
                expected_current_revision_id="rev-1",
                expected_pointer_revision=1,
                new_revision_id="rev-2",
                prior_revision_id="rev-1",
                committed_at=datetime.now(UTC),
            )
        except Exception as exc:
            swap_error.append(exc)
        mutation_finished.set()

    t_commit = threading.Thread(target=commit)
    t_swap = threading.Thread(target=swap_serving)
    t_commit.start()
    t_swap.start()
    at_validation.wait(timeout=5.0)
    mutation_started.wait(timeout=5.0)
    assert not mutation_finished.is_set()
    continue_commit.set()
    t_commit.join(timeout=5.0)
    t_swap.join(timeout=5.0)
    assert mutation_finished.is_set()
    assert swap_error and isinstance(swap_error[0], RuntimeActivationConflict)
    assert harness.state.serving_records[_scope()].traffic_serving_revision_id == "rev-2"


def test_failed_concurrent_mutation_leaves_activation_state_unchanged() -> None:
    harness = SharedActivationHarness()
    _bootstrap_active(harness)
    harness.seed_validated("rev-2")
    harness.prepare("rev-2")
    before = _snapshot_state(harness.state)
    harness.activation_store_a._fail_atomic_commit_activation = True
    at_validation, continue_commit = _pause_at_validation(harness.activation_store_a)
    mutation_started = threading.Event()
    mutation_finished = threading.Event()

    def commit() -> None:
        with pytest.raises(RuntimeActivationConflict):
            harness.commit_a("rev-2", expected_prior="rev-1", pointer=1)

    def stale_mutate() -> None:
        at_validation.wait(timeout=5.0)
        mutation_started.set()
        instance = harness.deployment_store_b.get_instance(_APP, _ENV, "rev-2")
        assert instance is not None
        with pytest.raises(RuntimeActivationConflict):
            harness.deployment_store_b.update_instance(
                instance.model_copy(
                    update={"instance_state": DeploymentInstanceState.PREPARING}
                ),
                expected_state=DeploymentInstanceState.SERVING,
                expected_record_revision=instance.record_revision,
            )
        mutation_finished.set()

    t_commit = threading.Thread(target=commit)
    t_mutate = threading.Thread(target=stale_mutate)
    t_commit.start()
    t_mutate.start()
    at_validation.wait(timeout=5.0)
    mutation_started.wait(timeout=5.0)
    assert not mutation_finished.is_set()
    continue_commit.set()
    t_commit.join(timeout=5.0)
    t_mutate.join(timeout=5.0)
    assert mutation_finished.is_set()
    assert _snapshot_state(harness.state) == before


def test_rollback_vs_activation_from_different_services_one_coherent_result() -> None:
    harness = SharedActivationHarness()
    _bootstrap_active(harness, "rev-n")
    harness.seed_validated("rev-n1")
    harness.prepare("rev-n1")
    harness.commit_a("rev-n1", expected_prior="rev-n", pointer=1)
    harness.seed_validated("rev-n2")
    harness.prepare("rev-n2")
    outcomes: list[str] = []
    errors: list[Exception] = []

    def activate() -> None:
        try:
            harness.commit_b("rev-n2", expected_prior="rev-n1", pointer=2)
            outcomes.append("activate")
        except Exception as exc:
            errors.append(exc)

    def rollback() -> None:
        try:
            harness.activation_a.rollback(
                application_id=_APP,
                application_environment_id=_ENV,
                expected_current_traffic_revision_id="rev-n1",
                expected_serving_pointer_revision=2,
            )
            outcomes.append("rollback")
        except Exception as exc:
            errors.append(exc)

    t_activate = threading.Thread(target=activate)
    t_rollback = threading.Thread(target=rollback)
    t_activate.start()
    t_rollback.start()
    t_activate.join()
    t_rollback.join()
    assert len(outcomes) == 1
    serving = harness.state.serving_records[_scope()]
    if outcomes[0] == "activate":
        assert serving.traffic_serving_revision_id == "rev-n2"
        assert (
            harness.state.revisions["rev-n2"].revision_state
            is RuntimeRevisionState.ACTIVE
        )
    else:
        assert serving.traffic_serving_revision_id == "rev-n"
        assert (
            harness.state.revisions["rev-n"].revision_state
            is RuntimeRevisionState.ACTIVE
        )
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeActivationConflict)


def test_cross_service_idempotent_retry_remains_correct() -> None:
    harness = SharedActivationHarness()
    _bootstrap_active(harness)
    harness.commit_a("rev-1")
    first_pointer = harness.state.serving_records[_scope()].serving_pointer_revision
    second = harness.activation_b.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-1",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=_ARTIFACT,
    )
    assert second.value.serving_record.serving_pointer_revision == first_pointer
    assert (
        harness.state.revisions["rev-1"].revision_state is RuntimeRevisionState.ACTIVE
    )
