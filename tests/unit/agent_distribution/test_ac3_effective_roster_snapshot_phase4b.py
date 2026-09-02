# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-007 Phase 4B — canonical effective roster snapshot persistence in build flow."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.agent_distribution.admin_models import AgentPlatformAdminBlockedError
from intergrax.agent_distribution.errors import (
    EffectiveRosterSnapshotConflict,
    RuntimeRevisionLifecycleError,
)
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryEffectiveRosterSnapshotStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    _APP,
    _ARTIFACT,
    _ENV,
    _build_request,
    _build_revision,
    _install_bind,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_KNOWN_LOCATOR = "test://canonical-effective-roster-snapshot"


class _KnownLocatorAdapter:
    topology = MaterializationTopology.OCI_IMAGE
    materializer_id = "intergrax.ac3-phase4b"
    materializer_version = "1.0.0"

    def materialize(self, materialization_input: object) -> MaterializationOutput:
        del materialization_input
        return MaterializationOutput(
            materialization_artifact_digest=_ARTIFACT,
            artifact_locator=_KNOWN_LOCATOR,
            health_check_evidence_ref="test://health",
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=self.topology,
        )


@dataclass
class _RecordingEffectiveRosterSnapshotStore:
    delegate: InMemoryEffectiveRosterSnapshotStore
    events: list[str] = field(default_factory=list)

    def get_by_revision(
        self,
        effective_roster_revision_id: str,
    ) -> EffectiveRoster | None:
        return self.delegate.get_by_revision(effective_roster_revision_id)

    def persist(self, roster: EffectiveRoster) -> EffectiveRoster:
        self.events.append("persist")
        return self.delegate.persist(roster)


class _RecordingRevisionService(RuntimeRevisionService):
    def __init__(self, store: InMemoryRuntimeRevisionStore) -> None:
        super().__init__(store)
        self.persist_events: list[str] = []

    def persist_candidate_revision(self, candidate):
        self.persist_events.append("persist_candidate")
        return super().persist_candidate_revision(candidate)


class _FailingEffectiveRosterSnapshotStore:
    def get_by_revision(
        self,
        effective_roster_revision_id: str,
    ) -> EffectiveRoster | None:
        del effective_roster_revision_id
        return None

    def persist(self, roster: EffectiveRoster) -> EffectiveRoster:
        del roster
        raise EffectiveRosterSnapshotConflict(
            "forced effective roster snapshot persist failure"
        )


class _FailingMarkValidatedRevisionService(RuntimeRevisionService):
    def __init__(
        self,
        store: InMemoryRuntimeRevisionStore,
        *,
        fail_once: bool = True,
    ) -> None:
        super().__init__(store)
        self._fail_once = fail_once
        self.mark_validated_calls = 0

    def mark_validated(self, runtime_revision_id: str, *, validated_revision):
        self.mark_validated_calls += 1
        if self._fail_once:
            self._fail_once = False
            raise RuntimeRevisionLifecycleError("forced mark_validated failure")
        return super().mark_validated(
            runtime_revision_id,
            validated_revision=validated_revision,
        )


def _stack_with_known_locator() -> AdminStack:
    stack = build_admin_stack()
    stack.service._materialization_service = RuntimeMaterializationService(
        {MaterializationTopology.OCI_IMAGE: _KnownLocatorAdapter()}
    )
    return stack


def _enable_binding(stack: AdminStack) -> None:
    from intergrax.agent_distribution.admin_models import SetAgentEnablementRequest

    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable", expected_revision=0
        ),
        principal=admin_test_principal(),
    )


def test_production_runtime_effective_roster_snapshot_store_shares_distribution_state() -> (
    None
):
    runtime = build_production_agent_platform_runtime()
    store = runtime.stores.effective_roster_snapshot_store
    assert isinstance(store, InMemoryEffectiveRosterSnapshotStore)
    assert store.state is runtime.distribution_state
    assert (
        runtime.stores.revision_store.state  # type: ignore[attr-defined]
        is runtime.distribution_state
    )
    assert (
        runtime.stores.lock_store.state  # type: ignore[attr-defined]
        is runtime.distribution_state
    )
    assert (
        runtime.stores.materialization_store.state  # type: ignore[attr-defined]
        is runtime.distribution_state
    )

    roster = EffectiveRoster(
        application_id="app-a",
        application_environment_id="prod",
        manifest_release_id="rel-1",
        entries=(),
    ).with_revision_id()
    runtime.stores.effective_roster_snapshot_store.persist(roster)
    reader = InMemoryEffectiveRosterSnapshotStore(runtime.distribution_state)
    loaded = reader.get_by_revision(roster.effective_roster_revision_id)
    assert loaded == roster


def test_successful_build_persists_canonical_effective_roster_snapshot() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    result = _build_revision(stack, "rev-phase4b-persist")
    revision = stack.service._revision_store.get_revision("rev-phase4b-persist")
    assert revision is not None
    snapshot = stack.effective_roster_snapshot_store.get_by_revision(
        revision.effective_roster_revision_id
    )
    assert snapshot is not None
    assert (
        snapshot.effective_roster_revision_id
        == revision.effective_roster_revision_id
        == result.effective_roster_revision_id
    )
    assert snapshot.compute_revision_id() == revision.effective_roster_revision_id
    assert snapshot.application_id == _APP
    assert snapshot.application_environment_id == _ENV
    assert snapshot.manifest_release_id == revision.application_release_id
    built_roster = stack.service._build_roster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id=revision.application_release_id,
    )
    assert snapshot == built_roster


def test_snapshot_persist_occurs_before_candidate_revision_persist() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    recording_snapshot = _RecordingEffectiveRosterSnapshotStore(
        stack.effective_roster_snapshot_store
    )
    recording_revision = _RecordingRevisionService(stack.service._revision_store)
    stack.service._effective_roster_snapshot_store = recording_snapshot
    stack.service._revision_service = recording_revision
    _build_revision(stack, "rev-phase4b-order")
    assert recording_snapshot.events == ["persist"]
    assert recording_revision.persist_events == ["persist_candidate"]
    assert recording_snapshot.events[0] == "persist"
    assert recording_revision.persist_events[0] == "persist_candidate"


def test_snapshot_persist_failure_prevents_candidate_creation() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    stack.service._effective_roster_snapshot_store = (
        _FailingEffectiveRosterSnapshotStore()
    )
    with pytest.raises(EffectiveRosterSnapshotConflict):
        _build_revision(stack, "rev-phase4b-snapshot-fail")
    revision = stack.service._revision_store.get_revision("rev-phase4b-snapshot-fail")
    assert revision is None
    assert not stack.state.locks
    assert not stack.state.materializations


def test_validated_replay_missing_snapshot_fails_closed() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    revision_id = "rev-phase4b-validated-missing"
    request = _build_request(revision_id, mutation_id="mut-phase4b-validated-missing")
    _build_revision(stack, revision_id)
    stack.state.effective_roster_snapshots.clear()
    with pytest.raises(AgentPlatformAdminBlockedError) as exc_info:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=admin_test_principal(),
        )
    assert (
        exc_info.value.blocker_code
        == "AP-11_BLOCKED_BY_MISSING_EFFECTIVE_ROSTER_SNAPSHOT"
    )


def test_candidate_replay_missing_snapshot_fails_closed() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    revision_id = "rev-phase4b-candidate-missing"
    request = _build_request(revision_id, mutation_id="mut-phase4b-candidate-missing")
    stack.service._revision_service = _FailingMarkValidatedRevisionService(
        stack.service._revision_store,
        fail_once=True,
    )
    with pytest.raises(RuntimeRevisionLifecycleError, match="forced mark_validated"):
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=admin_test_principal(),
        )
    stack.state.effective_roster_snapshots.clear()
    with pytest.raises(AgentPlatformAdminBlockedError) as exc_info:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=admin_test_principal(),
        )
    assert (
        exc_info.value.blocker_code
        == "AP-11_BLOCKED_BY_MISSING_EFFECTIVE_ROSTER_SNAPSHOT"
    )
    revision = stack.service._revision_store.get_revision(revision_id)
    assert revision is not None
    assert revision.revision_state is RuntimeRevisionState.CANDIDATE


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("application_id", "other-app"),
        ("application_environment_id", "other-env"),
        ("manifest_release_id", "other-release"),
    ],
)
def test_replay_scope_mismatch_fails_closed(
    field_name: str,
    field_value: str,
) -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    revision_id = f"rev-phase4b-scope-{field_name}"
    _build_revision(
        stack,
        revision_id,
    )
    revision = stack.service._revision_store.get_revision(revision_id)
    assert revision is not None
    snapshot = stack.effective_roster_snapshot_store.get_by_revision(
        revision.effective_roster_revision_id
    )
    assert snapshot is not None
    alternate = snapshot.model_copy(update={field_name: field_value}).with_revision_id()
    stack.state.effective_roster_snapshots[alternate.effective_roster_revision_id] = (
        alternate
    )
    mismatched_revision = revision.model_copy(
        update={"effective_roster_revision_id": alternate.effective_roster_revision_id}
    )
    with pytest.raises(AgentPlatformAdminBlockedError) as exc_info:
        stack.service._require_historical_effective_roster_snapshot(mismatched_revision)
    assert (
        exc_info.value.blocker_code
        == "AP-11_BLOCKED_BY_EFFECTIVE_ROSTER_SNAPSHOT_AUTHORITY"
    )


def test_current_desired_state_does_not_repair_missing_snapshot_history() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    revision_id = "rev-phase4b-no-repair"
    request = _build_request(revision_id, mutation_id="mut-phase4b-no-repair")
    _build_revision(stack, revision_id)
    revision = stack.service._revision_store.get_revision(revision_id)
    assert revision is not None
    roster_revision_id = revision.effective_roster_revision_id
    stack.state.effective_roster_snapshots.clear()
    current_roster = stack.service._build_roster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id=revision.application_release_id,
    )
    assert current_roster.effective_roster_revision_id == roster_revision_id
    with pytest.raises(AgentPlatformAdminBlockedError) as exc_info:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=admin_test_principal(),
        )
    assert (
        exc_info.value.blocker_code
        == "AP-11_BLOCKED_BY_MISSING_EFFECTIVE_ROSTER_SNAPSHOT"
    )
    assert roster_revision_id not in stack.state.effective_roster_snapshots


def test_idempotent_snapshot_reused_across_distinct_runtime_revisions() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    first = _build_revision(stack, "rev-phase4b-idem-a")
    second = _build_revision(stack, "rev-phase4b-idem-b")
    assert first.effective_roster_revision_id == second.effective_roster_revision_id
    assert len(stack.state.effective_roster_snapshots) == 1
    snapshot = stack.effective_roster_snapshot_store.get_by_revision(
        first.effective_roster_revision_id
    )
    assert snapshot is not None
