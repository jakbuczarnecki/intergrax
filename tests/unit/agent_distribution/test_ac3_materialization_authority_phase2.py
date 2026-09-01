# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-006 Phase 2 — canonical runtime materialization authority in build flow."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.agent_distribution.admin_models import AgentPlatformAdminBlockedError
from intergrax.agent_distribution.errors import (
    RuntimeMaterializationConflict,
    RuntimeRevisionLifecycleError,
)
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    _ARTIFACT,
    _APP,
    _ENV,
    _build_request,
    _build_revision,
    _install_bind,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_KNOWN_LOCATOR = "test://canonical-materialization-authority"


class _KnownLocatorAdapter:
    topology = MaterializationTopology.OCI_IMAGE
    materializer_id = "intergrax.ac3-phase2"
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
class _RecordingMaterializationStore:
    delegate: InMemoryRuntimeMaterializationStore
    events: list[str] = field(default_factory=list)

    def get_by_revision(
        self, runtime_revision_id: str
    ) -> RuntimeMaterializationRecord | None:
        return self.delegate.get_by_revision(runtime_revision_id)

    def persist(
        self, record: RuntimeMaterializationRecord
    ) -> RuntimeMaterializationRecord:
        self.events.append("persist")
        return self.delegate.persist(record)


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

    def mark_validated(
        self, runtime_revision_id: str, *, validated_revision: RuntimeRevision
    ):
        self.mark_validated_calls += 1
        if self._fail_once:
            self._fail_once = False
            raise RuntimeRevisionLifecycleError("forced mark_validated failure")
        return super().mark_validated(
            runtime_revision_id,
            validated_revision=validated_revision,
        )


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


def _stack_with_known_locator() -> AdminStack:
    stack = build_admin_stack()
    stack.service._materialization_service = RuntimeMaterializationService(
        {MaterializationTopology.OCI_IMAGE: _KnownLocatorAdapter()}
    )
    return stack


def test_production_runtime_materialization_store_shares_distribution_state() -> None:
    runtime = build_production_agent_platform_runtime()
    store = runtime.stores.materialization_store
    assert isinstance(store, InMemoryRuntimeMaterializationStore)
    assert store.state is runtime.distribution_state

    record = RuntimeMaterializationRecord(
        runtime_revision_id="rev-prod-shared",
        application_id="app-a",
        application_environment_id="prod",
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        artifact_locator="file:///artifact/prod",
        materialization_artifact_digest=_ARTIFACT,
        materialized_runtime_lock_id="sha256:" + ("c" * 64),
        materialized_runtime_lock_digest="sha256:" + ("c" * 64),
    )
    runtime.stores.materialization_store.persist(record)
    reader = InMemoryRuntimeMaterializationStore(runtime.distribution_state)
    loaded = reader.get_by_revision("rev-prod-shared")
    assert loaded == record


def test_successful_build_persists_canonical_materialization_record() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    result = _build_revision(stack, "rev-ac3-persist")
    record = stack.materialization_store.get_by_revision("rev-ac3-persist")
    assert record is not None
    assert record.runtime_revision_id == "rev-ac3-persist"
    assert record.application_id == _APP
    assert record.application_environment_id == _ENV
    assert record.materialization_topology == result.materialization_topology
    assert record.artifact_locator == result.artifact_locator
    assert (
        record.materialization_artifact_digest == result.materialization_artifact_digest
    )
    assert record.materialized_runtime_lock_id == result.materialized_runtime_lock_id
    assert (
        record.materialized_runtime_lock_digest
        == result.materialized_runtime_lock_digest
    )
    revision = stack.service.inspect_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-ac3-persist",
    )
    assert revision.revision_state is RuntimeRevisionState.VALIDATED


def test_build_result_artifact_locator_is_canonical_echo() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    result = _build_revision(stack, "rev-ac3-echo")
    record = stack.materialization_store.get_by_revision("rev-ac3-echo")
    assert record is not None
    assert result.artifact_locator == _KNOWN_LOCATOR
    assert result.artifact_locator == record.artifact_locator


def test_persist_occurs_before_mark_validated() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    recording_store = _RecordingMaterializationStore(stack.materialization_store)
    stack.service._materialization_store = recording_store
    stack.service._revision_service = _FailingMarkValidatedRevisionService(
        stack.service._revision_store,
        fail_once=True,
    )
    with pytest.raises(RuntimeRevisionLifecycleError, match="forced mark_validated"):
        _build_revision(stack, "rev-ac3-order")
    assert recording_store.events == ["persist"]
    assert stack.service._revision_service.mark_validated_calls == 1
    record = stack.materialization_store.get_by_revision("rev-ac3-order")
    assert record is not None
    revision = stack.service._revision_store.get_revision("rev-ac3-order")
    assert revision is not None
    assert revision.revision_state is RuntimeRevisionState.CANDIDATE


def test_materialization_persist_conflict_fails_closed() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    revision_id = "rev-ac3-conflict"
    stack.materialization_store.persist(
        RuntimeMaterializationRecord(
            runtime_revision_id=revision_id,
            application_id=_APP,
            application_environment_id=_ENV,
            materialization_topology=MaterializationTopology.OCI_IMAGE,
            artifact_locator="test://conflicting-authority",
            materialization_artifact_digest=_ARTIFACT,
            materialized_runtime_lock_id="sha256:" + ("1" * 64),
            materialized_runtime_lock_digest="sha256:" + ("2" * 64),
        )
    )
    with pytest.raises(RuntimeMaterializationConflict):
        _build_revision(stack, revision_id)
    revision = stack.service._revision_store.get_revision(revision_id)
    if revision is not None:
        assert revision.revision_state is not RuntimeRevisionState.VALIDATED
    existing = stack.materialization_store.get_by_revision(revision_id)
    assert existing is not None
    assert existing.artifact_locator == "test://conflicting-authority"


def test_idempotent_validated_replay_uses_canonical_record() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    request = _build_request("rev-ac3-replay", mutation_id="mut-ac3-replay")
    first = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=admin_test_principal(),
    )
    second = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=admin_test_principal(),
    )
    record = stack.materialization_store.get_by_revision("rev-ac3-replay")
    assert record is not None
    assert second.artifact_locator == first.artifact_locator == record.artifact_locator
    assert (
        second.materialization_artifact_digest
        == first.materialization_artifact_digest
        == record.materialization_artifact_digest
    )


def test_validated_revision_without_materialization_record_fails_closed() -> None:
    stack = build_admin_stack()
    _enable_binding(stack)
    built = _build_revision(stack, "rev-ac3-missing-record")
    revision = stack.service._revision_store.get_revision(built.runtime_revision_id)
    assert revision is not None
    stack.state.materializations.clear()
    request = _build_request(
        "rev-ac3-missing-record", mutation_id="mut-ac3-missing-record"
    )
    with pytest.raises(AgentPlatformAdminBlockedError) as exc_info:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=admin_test_principal(),
        )
    assert (
        exc_info.value.blocker_code
        == "AP-11_BLOCKED_BY_MISSING_MATERIALIZATION_AUTHORITY"
    )


def test_candidate_replay_completes_after_mark_validated_failure() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    revision_id = "rev-ac3-candidate-complete"
    request = _build_request(revision_id, mutation_id="mut-ac3-candidate-complete")
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
    result = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=admin_test_principal(),
    )
    assert result.revision_state is RuntimeRevisionState.VALIDATED
    assert result.artifact_locator == _KNOWN_LOCATOR
    revision = stack.service.inspect_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=revision_id,
    )
    assert revision.revision_state is RuntimeRevisionState.VALIDATED


def test_cross_state_isolation_between_admin_stacks() -> None:
    stack_a = _stack_with_known_locator()
    stack_b = build_admin_stack()
    _enable_binding(stack_a)
    _build_revision(stack_a, "rev-ac3-isolated")
    assert stack_b.materialization_store.get_by_revision("rev-ac3-isolated") is None


def test_candidate_without_materialization_record_fails_closed_on_replay() -> None:
    stack = _stack_with_known_locator()
    _enable_binding(stack)
    revision_id = "rev-ac3-candidate-gap"
    request = _build_request(revision_id, mutation_id="mut-ac3-candidate-gap")
    _build_revision(stack, revision_id)
    revision = stack.service._revision_store.get_revision(revision_id)
    assert revision is not None
    stack.state.materializations.clear()
    stack.service._revision_store.persist_candidate_revision(
        revision.model_copy(update={"revision_state": RuntimeRevisionState.CANDIDATE})
    )
    with pytest.raises(AgentPlatformAdminBlockedError) as exc_info:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=admin_test_principal(),
        )
    assert exc_info.value.blocker_code == "AP-11_BLOCKED_BY_INCOMPLETE_BUILD_REPLAY"
