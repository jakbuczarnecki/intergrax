# © Artur Czarnecki. All rights reserved.

"""AP-4 runtime revision domain service tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.agent_distribution.errors import RuntimeRevisionConflict, RuntimeRevisionLifecycleError
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology, RuntimeRevision, RuntimeRevisionState
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService


def _validated_revision(
    revision_id: str,
    *,
    state: RuntimeRevisionState = RuntimeRevisionState.VALIDATED,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id="app-a",
        application_environment_id="env-prod",
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="lock-digest",
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest="artifact-digest",
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=datetime.now(UTC) if state is RuntimeRevisionState.ACTIVE else None,
    )


def _service(state: AgentDistributionStoreState | None = None) -> tuple[RuntimeRevisionService, AgentDistributionStoreState]:
    backing = state or AgentDistributionStoreState()
    store = InMemoryRuntimeRevisionStore(backing)
    return RuntimeRevisionService(store), backing


def test_runtime_revision_candidate_to_validated_to_active() -> None:
    service, state = _service()
    candidate = _validated_revision("rev-1", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(candidate)
    validated = _validated_revision("rev-1")
    service.mark_validated("rev-1", validated_revision=validated)
    activated = service.activate_revision("rev-1")
    assert activated.value.revision_state is RuntimeRevisionState.ACTIVE
    assert activated.value.rollback_target_revision_id is None
    assert any(event.event_type == "runtime_revision.activated" for event in activated.events)
    from intergrax.agent_distribution.application_environment_identity import (
        ApplicationEnvironmentIdentity,
    )

    scope = ApplicationEnvironmentIdentity(
        application_id="app-a",
        application_environment_id="env-prod",
    )
    assert state.active_revision_by_scope[scope] == "rev-1"


def test_runtime_revision_rejects_identity_mutation_on_validation() -> None:
    service, _ = _service()
    candidate = _validated_revision("rev-1", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(candidate)
    mutated = _validated_revision("rev-1").model_copy(
        update={"application_id": "app-other"},
    )
    with pytest.raises(RuntimeRevisionLifecycleError, match="application_id"):
        service.mark_validated("rev-1", validated_revision=mutated)


def test_runtime_revision_prior_active_becomes_superseded_with_rollback_pointer() -> None:
    service, state = _service()
    first_candidate = _validated_revision("rev-1", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(first_candidate)
    service.mark_validated("rev-1", validated_revision=_validated_revision("rev-1"))
    service.activate_revision("rev-1")
    second_candidate = _validated_revision("rev-2", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(second_candidate)
    service.mark_validated("rev-2", validated_revision=_validated_revision("rev-2"))
    activated = service.activate_revision("rev-2", expected_prior_active_revision_id="rev-1")
    assert activated.value.rollback_target_revision_id == "rev-1"
    prior = state.revisions["rev-1"]
    assert prior.revision_state is RuntimeRevisionState.SUPERSEDED


def test_runtime_revision_stale_active_swap_rejected() -> None:
    service, _ = _service()
    candidate = _validated_revision("rev-1", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(candidate)
    service.mark_validated("rev-1", validated_revision=_validated_revision("rev-1"))
    service.activate_revision("rev-1")
    second = _validated_revision("rev-2", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(second)
    service.mark_validated("rev-2", validated_revision=_validated_revision("rev-2"))
    with pytest.raises(RuntimeRevisionConflict):
        service.activate_revision("rev-2", expected_prior_active_revision_id="rev-missing")


def test_runtime_revision_restart_reads_durable_active_revision() -> None:
    state = AgentDistributionStoreState()
    service, _ = _service(state)
    candidate = _validated_revision("rev-1", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(candidate)
    service.mark_validated("rev-1", validated_revision=_validated_revision("rev-1"))
    service.activate_revision("rev-1")
    restarted_store = InMemoryRuntimeRevisionStore(state)
    restarted_service = RuntimeRevisionService(restarted_store)
    active = restarted_service.get_active_revision("app-a", "env-prod")
    assert active is not None
    assert active.runtime_revision_id == "rev-1"
    assert active.revision_state is RuntimeRevisionState.ACTIVE


def test_runtime_revision_invalid_transition_rejected() -> None:
    service, _ = _service()
    candidate = _validated_revision("rev-1", state=RuntimeRevisionState.CANDIDATE)
    service.persist_candidate_revision(candidate)
    with pytest.raises(RuntimeRevisionLifecycleError):
        service.activate_revision("rev-1")
