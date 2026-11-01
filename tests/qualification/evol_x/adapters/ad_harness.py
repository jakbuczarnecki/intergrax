# © Artur Czarnecki. All rights reserved.

"""agent_distribution harness adapter (S7): drives the real in-memory ActivationService rollback (observer)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.activation import ActivationService, FakeRuntimeServingProjectionCoordinator
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
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
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from testing_support.agent_distribution.activation_materialization_fixtures import persist_materialization_for_revision

_APP = "evolx-app"
_ENV = "env-s7"
_ARTIFACT = "sha256:" + ("e" * 64)
_LOCATOR = f"artifact://{_ARTIFACT}"


@dataclass(frozen=True)
class AdRollbackObservation:
    """What the AD layer alone does for a rollback (no durable-schema knowledge): outcome + serving pointer."""

    succeeded: bool
    traffic_revision_after: str | None
    error_type: str | None


def _revision(revision_id: str, state: RuntimeRevisionState) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="sha256:" + ("2" * 64),
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest=_ARTIFACT,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
    )


def observe_ad_rollback() -> AdRollbackObservation:
    """Seed rev-1 -> rev-2 (both ACTIVE once), then roll back rev-2 -> rev-1 through the production service."""
    state = AgentDistributionStoreState()
    revision_store = InMemoryRuntimeRevisionStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    materialization_store = InMemoryRuntimeMaterializationStore(state)
    service = RuntimeRevisionService(revision_store)
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=InMemoryDeploymentInstanceStore(state),
        serving_store=serving_store,
        activation_store=InMemoryApplicationEnvironmentActivationStore(state),
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        materialization_store=materialization_store,
    )
    for revision_id in ("rev-1", "rev-2"):
        service.persist_candidate_revision(_revision(revision_id, RuntimeRevisionState.CANDIDATE))
        validated = _revision(revision_id, RuntimeRevisionState.VALIDATED)
        service.mark_validated(revision_id, validated_revision=validated)
        persist_materialization_for_revision(materialization_store, validated, artifact_locator=_LOCATOR)
    prior: str | None = None
    for pointer, revision_id in enumerate(("rev-1", "rev-2")):
        activation.prepare_candidate(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision_id,
            artifact_locator=_LOCATOR,
        )
        activation.commit_activation(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision_id,
            expected_prior_traffic_revision_id=prior,
            expected_serving_pointer_revision=pointer,
            expected_artifact_digest=_ARTIFACT,
        )
        prior = revision_id
    try:
        activation.rollback(
            application_id=_APP,
            application_environment_id=_ENV,
            expected_current_traffic_revision_id="rev-2",
            expected_serving_pointer_revision=2,
        )
    except Exception as exc:  # noqa: BLE001 - observation boundary: record, never raise
        record = serving_store.get_serving_record(_APP, _ENV)
        return AdRollbackObservation(False, record.traffic_serving_revision_id if record else None, type(exc).__name__)
    record = serving_store.get_serving_record(_APP, _ENV)
    return AdRollbackObservation(True, record.traffic_serving_revision_id if record else None, None)
