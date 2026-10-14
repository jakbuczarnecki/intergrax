# © Artur Czarnecki. All rights reserved.

"""Fixtures for canonical serving execution admission tests (PROD-Q-R1-R7-D1-R3)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.agent_distribution.deployment import DeploymentInstanceRecord, DeploymentInstanceState
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
    _application_environment_scope,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevision, RuntimeRevisionState
from intergrax.agent_distribution.activation_domain_coordinator import (
    ActivationDomainCoordinator,
    build_activation_domain_coordinator_for_stores,
)
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord


def activation_domain_coordinator_for_stores(
    serving_store: InMemoryApplicationEnvironmentServingStore,
    deployment_instance_store: InMemoryDeploymentInstanceStore,
    revision_store: InMemoryRuntimeRevisionStore,
) -> ActivationDomainCoordinator:
    return build_activation_domain_coordinator_for_stores(
        serving_store=serving_store,
        deployment_instance_store=deployment_instance_store,
        revision_store=revision_store,
    )


def seed_canonical_serving_execution_stores(
    *,
    state: AgentDistributionStoreState,
    revision: RuntimeRevision,
    serving_record: ApplicationEnvironmentServingRecord,
    deployment_instance: DeploymentInstanceRecord,
) -> tuple[
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
]:
    """Populate in-memory stores for ACTIVE/SERVING execution admission qualification."""
    active_revision = revision.model_copy(
        update={
            "revision_state": RuntimeRevisionState.ACTIVE,
            "activated_at": revision.activated_at or datetime.now(UTC),
        },
    )
    scope = _application_environment_scope(
        serving_record.application_id,
        serving_record.application_environment_id,
    )
    state.revisions[revision.runtime_revision_id] = active_revision
    state.active_revision_by_scope[scope] = revision.runtime_revision_id

    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    state.serving_records[scope] = serving_record

    instance_store = InMemoryDeploymentInstanceStore(state)
    serving_instance = deployment_instance.model_copy(
        update={"instance_state": DeploymentInstanceState.SERVING},
    )
    instance_store.persist_instance(serving_instance)

    revision_store = InMemoryRuntimeRevisionStore(state)
    return serving_store, instance_store, revision_store


__all__ = [
    "activation_domain_coordinator_for_stores",
    "seed_canonical_serving_execution_stores",
]
