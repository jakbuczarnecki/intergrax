# © Artur Czarnecki. All rights reserved.

"""Tests for Workspace Knowledge Configuration mutation engine."""

from __future__ import annotations

import inspect
import threading
from datetime import UTC, datetime
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeExistingResult,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
    WorkspaceKnowledgeMutationExecutionResult,
    WorkspaceKnowledgeMutationRecoveryDispositionV1,
    WorkspaceKnowledgeStageInspection,
    WorkspaceKnowledgeStageStateV1,
    WorkspaceKnowledgeStagedResult,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import (
    ManagedWorkspaceRepository,
    WorkspaceKnowledgeConfigurationRepositoryError,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_SHA256 = "a" * 64
_SHA256_B = "b" * 64
_SHA256_C = "c" * 64
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-1"
_WORKSPACE_B = "workspace-2"
_OPERATION = WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION
_RESULT_TYPE = "connection_attachment"
_RESULT_ID = "att-staged-1"


def _workspace(**overrides: object) -> Workspace:
    payload = {
        "workspace_id": _WORKSPACE,
        "tenant_id": _TENANT,
        "name": "Workspace",
        "status": WorkspaceStatus.ACTIVE,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return Workspace(**payload)


class _FakeWorkspaceLookup:
    def __init__(self, workspaces: dict[tuple[str, str], Workspace]) -> None:
        self._workspaces = workspaces

    def require_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> Workspace | None:
        workspace = self._workspaces.get((tenant_id, workspace_id))
        if workspace is None:
            return None
        if workspace.tenant_id != tenant_id or workspace.workspace_id != workspace_id:
            return None
        return workspace


class _FakeHandler:
    operation = _OPERATION

    def __init__(self) -> None:
        self.existing_result: WorkspaceKnowledgeExistingResult | None = None
        self.stage_calls = 0
        self.cleanup_calls = 0
        self.stage_target_revision_at_call: int | None = None
        self.inspection_state = WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
        self.inspection_type: str | None = _RESULT_TYPE
        self.inspection_id: str | None = _RESULT_ID
        self.cleanup_returns = True
        self.staged_rows: list[WorkspaceConnectionAttachment] = []
        self.projection_revision_sequence: list[int] | None = None
        self._projection_reads = 0

    def find_existing_result(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        intent: object,
    ) -> WorkspaceKnowledgeExistingResult | None:
        if self.projection_revision_sequence is not None:
            index = min(self._projection_reads, len(self.projection_revision_sequence) - 1)
            self._projection_reads += 1
            if configuration.configuration_revision != self.projection_revision_sequence[index]:
                return None
        return self.existing_result

    def stage(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        target_revision: int,
        intent: object,
        now: datetime,
    ) -> WorkspaceKnowledgeStagedResult:
        self.stage_calls += 1
        self.stage_target_revision_at_call = mutation.target_revision
        attachment = WorkspaceConnectionAttachment(
            attachment_id=_RESULT_ID,
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
            connection_ref="conn.primary",
            safe_display_label="Primary",
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id=mutation.mutation_id,
            effective_revision=target_revision,
            created_at=now,
            updated_at=now,
        )
        repository.put_knowledge_connection_attachment_version_if_absent(attachment)
        self.staged_rows.append(attachment)
        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_RESULT_TYPE,
            result_entity_id=_RESULT_ID,
        )

    def inspect_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeStageInspection:
        if self.inspection_state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )
        if not self.staged_rows:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.ABSENT,
            )
        if self.inspection_state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return WorkspaceKnowledgeStageInspection(
                state=self.inspection_state,
                result_entity_type=self.inspection_type,
                result_entity_id=self.inspection_id,
            )
        return WorkspaceKnowledgeStageInspection(state=self.inspection_state)

    def cleanup_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        self.cleanup_calls += 1
        if not self.cleanup_returns:
            return False
        for row in list(self.staged_rows):
            if not repository.delete_knowledge_connection_attachment_version_if_match(row):
                return False
            self.staged_rows.remove(row)
        return True


class _ControlledClock:
    def __init__(self, start: datetime) -> None:
        self._current = start

    def now(self) -> datetime:
        return self._current

    def advance(self, seconds: int = 1) -> None:
        self._current = self._current.replace(second=self._current.second + seconds)


class _ControlledMutationIds:
    def __init__(self, ids: list[str]) -> None:
        self._ids = list(ids)
        self._index = 0

    def next_id(self) -> str:
        if self._index >= len(self._ids):
            raise RuntimeError("mutation id exhausted")
        value = self._ids[self._index]
        self._index += 1
        return value


class _FailingHeadPublishStore(InMemoryDocumentStore):
    def __init__(self, *, fail_remaining: int) -> None:
        super().__init__()
        self._fail_remaining = fail_remaining

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_head" in expected.partition_key
            and expected.data.get("pending_mutation_id")
            and not replacement.data.get("pending_mutation_id")
        ):
            if self._fail_remaining > 0:
                self._fail_remaining -= 1
                return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _FailingMutationFinalizeStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__()
        self._head_published = False
        self._finalize_failures_remaining = 1

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_head" in expected.partition_key
            and expected.data.get("pending_mutation_id")
            and not replacement.data.get("pending_mutation_id")
        ):
            self._head_published = True
            return super().replace_if_match(expected=expected, replacement=replacement)
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and self._head_published
            and replacement.data.get("status") == "committed"
            and expected.data.get("status") != "committed"
        ):
            if self._finalize_failures_remaining > 0:
                self._finalize_failures_remaining -= 1
                return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _RevisionRaceBeforeWriterSlotCASStore(InMemoryDocumentStore):
    """Bumps committed revision immediately before writer-slot CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_head" in expected.partition_key
            and not expected.data.get("pending_mutation_id")
            and replacement.data.get("pending_mutation_id")
            and expected.data.get("committed_revision") == 4
        ):
            bumped = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "committed_revision": 5,
                        "updated_at": replacement.data.get("updated_at"),
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=bumped)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _FailingHeadReleaseStore(InMemoryDocumentStore):
    """Returns False for pending-head release CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_head" in expected.partition_key
            and expected.data.get("pending_mutation_id")
            and not replacement.data.get("pending_mutation_id")
        ):
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _AbortedRestartCommittedRaceStore(InMemoryDocumentStore):
    """Replaces ABORTED mutation with COMMITTED before restart CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("status") == "aborted"
            and replacement.data.get("status") == "reserved"
        ):
            committed = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "status": "committed",
                        "outcome": "applied",
                        "target_revision": 1,
                        "committed_revision": 1,
                        "result_entity_type": _RESULT_TYPE,
                        "result_entity_id": _RESULT_ID,
                        "committed_at": _NOW.isoformat(),
                        "error_code": None,
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=committed)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _AbortedRestartReservedRaceStore(InMemoryDocumentStore):
    """Replaces ABORTED mutation with RESERVED before restart CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("status") == "aborted"
            and replacement.data.get("status") == "reserved"
        ):
            reserved = replacement.model_copy(
                update={
                    "data": {
                        **replacement.data,
                        "mutation_id": "mutation-race-winner",
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=reserved)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _PreparedCASConcurrentPublicationStore(InMemoryDocumentStore):
    """Another executor completes during local RESERVED→PREPARED CAS."""

    def __init__(self, *, finalize: bool = True) -> None:
        super().__init__()
        self._finalize = finalize
        self._intercepted = False

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("status") == "reserved"
            and replacement.data.get("status") == "prepared"
            and not self._intercepted
        ):
            self._intercepted = True
            if not super().replace_if_match(expected=expected, replacement=replacement):
                return False
            tenant_id = expected.data["tenant_id"]
            workspace_id = expected.data["workspace_id"]
            head_pk = f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_head"
            head_doc = self.get(head_pk, workspace_id)
            if head_doc is None:
                return False
            target_revision = replacement.data.get("target_revision")
            published = head_doc.model_copy(
                update={
                    "data": {
                        **head_doc.data,
                        "committed_revision": target_revision,
                        "pending_revision": None,
                        "pending_mutation_id": None,
                        "last_committed_mutation_id": expected.data["mutation_id"],
                    }
                }
            )
            super().replace_if_match(expected=head_doc, replacement=published)
            if self._finalize:
                prepared_doc = self.get(expected.partition_key, expected.row_key)
                if prepared_doc is None:
                    return False
                committed_doc = prepared_doc.model_copy(
                    update={
                        "data": {
                            **prepared_doc.data,
                            "status": "committed",
                            "outcome": "applied",
                            "committed_revision": target_revision,
                            "committed_at": replacement.data.get("updated_at"),
                        }
                    }
                )
                super().replace_if_match(expected=prepared_doc, replacement=committed_doc)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


def _build_engine(
  *,
    store: InMemoryDocumentStore | None = None,
    handler: _FakeHandler | None = None,
    workspaces: dict[tuple[str, str], Workspace] | None = None,
    mutation_ids: list[str] | None = None,
    clock: _ControlledClock | None = None,
    handlers: dict[WorkspaceKnowledgeMutationOperationV1, _FakeHandler] | None = None,
) -> tuple[WorkspaceKnowledgeConfigurationMutationEngine, ManagedWorkspaceRepository, _FakeHandler]:
    store = store or InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    lookup = _FakeWorkspaceLookup(
        workspaces if workspaces is not None else {(_TENANT, _WORKSPACE): _workspace()}
    )
    config_service = WorkspaceKnowledgeConfigurationService(repo, lookup)
    fake_handler = handler or _FakeHandler()
    handler_map = handlers if handlers is not None else {_OPERATION: fake_handler}
    controlled_clock = clock or _ControlledClock(_NOW)
    id_factory = _ControlledMutationIds(
        mutation_ids or ["mutation-1", "mutation-2", "mutation-3", "mutation-4", "mutation-5"]
    )
    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config_service,
        handler_map,
        clock=controlled_clock.now,
        mutation_id_factory=id_factory.next_id,
    )
    return engine, repo, fake_handler


def _execute(
    engine: WorkspaceKnowledgeConfigurationMutationEngine,
    *,
    expected_revision: int = 0,
    idempotency_key_hash: str = _SHA256,
    normalized_request_hash: str = _SHA256,
    semantic_identity_hash: str | None = None,
    intent: object = object(),
) -> Any:
    return engine.execute(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        expected_revision=expected_revision,
        idempotency_key_hash=idempotency_key_hash,
        normalized_request_hash=normalized_request_hash,
        semantic_identity_hash=semantic_identity_hash,
        intent=intent,
    )


def _mutation_record(**overrides: object) -> WorkspaceKnowledgeMutationRecord:
    payload = {
        "mutation_id": "mutation-seed",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation": _OPERATION,
        "idempotency_key_hash": _SHA256,
        "normalized_request_hash": _SHA256,
        "status": WorkspaceKnowledgeMutationStatusV1.RESERVED,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceKnowledgeMutationRecord(**payload)


# --- Input validation ---


@pytest.mark.parametrize(
    ("kwargs", "error_code"),
    [
        ({"expected_revision": -1}, "knowledge_configuration_expected_revision_invalid"),
        ({"idempotency_key_hash": "A" * 64}, "knowledge_configuration_idempotency_hash_invalid"),
        ({"normalized_request_hash": "not-hex"}, "knowledge_configuration_request_hash_invalid"),
        ({"semantic_identity_hash": "B" * 64}, "knowledge_configuration_semantic_identity_hash_invalid"),
    ],
)
def test_invalid_inputs_raise_before_persistence(kwargs: dict[str, object], error_code: str) -> None:
    engine, repo, _handler = _build_engine()
    params = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation": _OPERATION,
        "expected_revision": 0,
        "idempotency_key_hash": _SHA256,
        "normalized_request_hash": _SHA256,
        "semantic_identity_hash": None,
        "intent": object(),
    }
    params.update(kwargs)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.execute(**params)
    assert exc.value.error_code == error_code
    assert repo.list_knowledge_configuration_mutations(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    ) == []


# --- Workspace validation ---


def test_unknown_workspace_raises_without_mutation_row() -> None:
    engine, repo, _handler = _build_engine(workspaces={})
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "workspace_not_found"
    assert repo.list_knowledge_configuration_mutations(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    ) == []


def test_cross_tenant_workspace_raises_without_mutation_row() -> None:
    engine, repo, _handler = _build_engine(
        workspaces={( _TENANT_B, _WORKSPACE): _workspace(tenant_id=_TENANT_B)}
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "workspace_not_found"
    assert repo.list_knowledge_configuration_mutations(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    ) == []


# --- Applied mutation ---


def test_new_applied_mutation_full_protocol() -> None:
    engine, repo, handler = _build_engine()
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert result.configuration_revision == 1
    assert result.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert result.mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == 1
    assert head.pending_mutation_id is None
    assert handler.stage_calls == 1
    staged = repo.get_knowledge_connection_attachment_version(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        attachment_id=_RESULT_ID,
        effective_revision=1,
    )
    assert staged is not None


def test_staged_record_hidden_before_publication() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(handler=handler)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
        _execute(engine)
    config_service = WorkspaceKnowledgeConfigurationService(
        repo,
        _FakeWorkspaceLookup({(_TENANT, _WORKSPACE): _workspace()}),
    )
    config = config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.configuration_revision == 0
    assert config.connection_attachments == ()


# --- Concurrent first mutation ---


def test_first_concurrent_mutation_race() -> None:
    store = InMemoryDocumentStore()
    handler_a = _FakeHandler()
    handler_b = _FakeHandler()
    engine_a, repo, _ = _build_engine(
        store=store,
        handler=handler_a,
        mutation_ids=["mutation-a", "mutation-b", "mutation-c"],
    )
    engine_b, _, _ = _build_engine(
        store=store,
        handler=handler_b,
        mutation_ids=["mutation-b", "mutation-c", "mutation-d"],
    )

    results: list[Any] = []
    errors: list[BaseException] = []

    def run(engine: WorkspaceKnowledgeConfigurationMutationEngine) -> None:
        try:
            results.append(_execute(engine, idempotency_key_hash=_SHA256_B))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = threading.Thread(target=run, args=(engine_a,))
    thread_b = threading.Thread(target=run, args=(engine_b,))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    applied = [
        item
        for item in results
        if item.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    ]
    assert len(applied) == 1
    assert applied[0].configuration_revision == 1
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == 1
    assert handler_a.stage_calls + handler_b.stage_calls == 1


# --- Committed replay ---


def test_committed_replay_returns_previous_result() -> None:
    engine, repo, handler = _build_engine()
    first = _execute(engine)
    handler.stage_calls = 0
    second = _execute(engine)
    assert first.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert second.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert second.result_entity_id == first.result_entity_id
    assert second.configuration_revision == first.configuration_revision
    assert handler.stage_calls == 0


# --- Idempotency conflict ---


@pytest.mark.parametrize(
    "status",
    [
        WorkspaceKnowledgeMutationStatusV1.RESERVED,
        WorkspaceKnowledgeMutationStatusV1.PREPARED,
        WorkspaceKnowledgeMutationStatusV1.COMMITTED,
        WorkspaceKnowledgeMutationStatusV1.ABORTED,
        WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
    ],
)
def test_idempotency_conflict_across_statuses(status: WorkspaceKnowledgeMutationStatusV1) -> None:
    engine, repo, _handler = _build_engine()
    overrides: dict[str, object] = {
        "status": status,
        "normalized_request_hash": _SHA256,
    }
    if status is WorkspaceKnowledgeMutationStatusV1.PREPARED:
        overrides.update(
            {
                "target_revision": 1,
                "result_entity_type": _RESULT_TYPE,
                "result_entity_id": _RESULT_ID,
            }
        )
    if status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
        overrides.update(
            {
                "target_revision": 1,
                "committed_revision": 1,
                "outcome": WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
                "result_entity_type": _RESULT_TYPE,
                "result_entity_id": _RESULT_ID,
                "committed_at": _NOW,
            }
        )
    if status is WorkspaceKnowledgeMutationStatusV1.ABORTED:
        overrides["error_code"] = "configuration_revision_conflict"
    if status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED:
        overrides.update({"target_revision": 1, "error_code": "writer_slot_stale"})
    repo.put_knowledge_configuration_mutation_if_absent(_mutation_record(**overrides))
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine, normalized_request_hash=_SHA256_B)
    assert exc.value.error_code == "configuration_idempotency_conflict"


# --- ABORTED restart ---


def test_aborted_restart_preserves_row_identity_and_clears_state() -> None:
    engine, repo, _handler = _build_engine(mutation_ids=["mutation-retry", "mutation-new"])
    aborted = _mutation_record(
        status=WorkspaceKnowledgeMutationStatusV1.ABORTED,
        mutation_id="mutation-old",
        target_revision=2,
        error_code="configuration_revision_conflict",
    )
    repo.put_knowledge_configuration_mutation_if_absent(aborted)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-new"
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert loaded.error_code is None
    assert loaded.target_revision == 1


# --- Semantic no-op ---


def test_semantic_no_op_commits_existing_result_without_writer_slot() -> None:
    handler = _FakeHandler()
    handler.existing_result = WorkspaceKnowledgeExistingResult(
        result_entity_type=_RESULT_TYPE,
        result_entity_id="att-existing",
    )
    engine, repo, _handler = _build_engine(handler=handler)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert result.configuration_revision == 0
    assert result.mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT
    assert result.mutation.target_revision is None
    assert repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE) is None
    assert handler.stage_calls == 0


def test_semantic_no_op_projection_race_retries_once() -> None:
    handler = _FakeHandler()
    handler.existing_result = WorkspaceKnowledgeExistingResult(
        result_entity_type=_RESULT_TYPE,
        result_entity_id="att-existing",
    )
    handler.projection_revision_sequence = [0, 0]
    engine, _, _handler = _build_engine(handler=handler)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT


def test_semantic_no_op_projection_unstable_on_repeated_change() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    lookup = _FakeWorkspaceLookup({(_TENANT, _WORKSPACE): _workspace()})
    handler = _FakeHandler()
    handler.existing_result = WorkspaceKnowledgeExistingResult(
        result_entity_type=_RESULT_TYPE,
        result_entity_id="att-existing",
    )

    class _UnstableConfigurationService(WorkspaceKnowledgeConfigurationService):
        def __init__(self) -> None:
            super().__init__(repo, lookup)
            self._reads = 0

        def get_configuration(
            self,
            *,
            tenant_id: str,
            workspace_id: str,
        ) -> WorkspaceKnowledgeConfigurationV1 | None:
            self._reads += 1
            config = super().get_configuration(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if config is None:
                return None
            head = repo.get_knowledge_configuration_head(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if head is None:
                repo.put_knowledge_configuration_head_if_absent(
                    WorkspaceKnowledgeConfigurationHead(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        committed_revision=self._reads,
                        updated_at=_NOW,
                    )
                )
            else:
                bumped = head.model_copy(
                    update={
                        "committed_revision": head.committed_revision + 1,
                        "updated_at": _NOW,
                    }
                )
                repo.replace_knowledge_configuration_head_if_match(
                    expected=head,
                    replacement=bumped,
                )
            return config.model_copy(
                update={"configuration_revision": self._reads - 1}
            )

    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        _UnstableConfigurationService(),
        {_OPERATION: handler},
        mutation_id_factory=_ControlledMutationIds(["mutation-unstable"]).next_id,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_projection_unstable"


# --- Revision conflict ---


def test_expected_revision_conflict_aborts_without_staging() -> None:
    engine, repo, handler = _build_engine()
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=2,
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine, expected_revision=0)
    assert exc.value.error_code == "configuration_revision_conflict"
    assert handler.stage_calls == 0
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert loaded.target_revision is None


# --- Writer-slot conflict ---


def test_pending_competing_mutation_raises_recovery_required() -> None:
    engine, repo, handler = _build_engine()
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="other-mutation",
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_recovery_required"
    assert handler.stage_calls == 0
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id == "other-mutation"


# --- Target assignment ordering ---


def test_handler_observes_assigned_target_revision_before_stage() -> None:
    engine, _, handler = _build_engine()
    _execute(engine)
    assert handler.stage_target_revision_at_call == 1


# --- Stage validation failure ---


def test_stage_validation_failure_cleans_up_and_aborts() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(handler=handler)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_mutation_stage_failed"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id is None
    assert handler.cleanup_calls == 1


# --- Cleanup failure ---


def test_cleanup_failure_leaves_pending_head_and_recovery_required() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    handler.cleanup_returns = False
    engine, repo, _handler = _build_engine(handler=handler)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_mutation_cleanup_failed"
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id is not None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED


# --- Publication failure ---


def test_publication_failure_raises_unstable_without_unsafe_cleanup() -> None:
    store = _FailingHeadPublishStore(fail_remaining=3)
    handler = _FakeHandler()
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_mutation_publication_unstable"
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id is not None
    assert len(handler.staged_rows) == 1


# --- Finalization recovery ---


def test_finalization_failure_after_publication_is_repaired() -> None:
    store = _FailingMutationFinalizeStore()
    engine, repo, _handler = _build_engine(store=store)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == 1
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


# --- Recovery paths ---


def test_recovery_reserved_complete_staged_state_commits() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-recover"])
    mutation = _mutation_record(
        mutation_id="mutation-recover",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-recover",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=mutation,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    assert recovery.mutation is not None
    assert recovery.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_recovery_prepared_complete_staged_state_commits() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-prepared"])
    prepared = _mutation_record(
        mutation_id="mutation-prepared",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(prepared)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-prepared",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=prepared,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED


def test_recovery_incomplete_owned_state_aborts() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-incomplete"])
    mutation = _mutation_record(
        mutation_id="mutation-incomplete",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-incomplete",
            updated_at=_NOW,
        )
    )
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    handler.stage(
        repository=repo,
        mutation=mutation,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id is None


def test_recovery_ownership_conflict_raises() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-owner"])
    mutation = _mutation_record(
        mutation_id="mutation-owner",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-owner",
            updated_at=_NOW,
        )
    )
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id == "mutation-owner"


def test_post_publication_recovery_finalizes_prepared_mutation() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-post"])
    prepared = _mutation_record(
        mutation_id="mutation-post",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(prepared)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=1,
            last_committed_mutation_id="other",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=prepared,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == 1


# --- Missing handler ---


def test_missing_handler_on_execute() -> None:
    engine, _, _handler = _build_engine(handlers={})
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "knowledge_configuration_handler_not_registered"


def test_missing_handler_on_recovery() -> None:
    engine, repo, _handler = _build_engine(handlers={})
    mutation = _mutation_record(
        mutation_id="mutation-no-handler",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-no-handler",
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code == "knowledge_configuration_handler_not_registered"


# --- Conditional store ---


def test_conditional_store_required() -> None:
    from typing import Optional

    from intergrax.integrations.contracts.document_store import DocumentQueryResult

    class _PlainDocumentStore:
        def __init__(self) -> None:
            self._rows: dict[tuple[str, str], DocumentRecord] = {}

        def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
            return self._rows.get((partition_key, row_key))

        def put(self, document: DocumentRecord) -> None:
            self._rows[(document.partition_key, document.row_key)] = document

        def delete(self, partition_key: str, row_key: str) -> None:
            self._rows.pop((partition_key, row_key), None)

        def query(
            self,
            partition_key: str,
            *,
            limit: int = 100,
            row_key_prefix: Optional[str] = None,
        ) -> DocumentQueryResult:
            rows: list[DocumentRecord] = []
            for (pk, rk), doc in self._rows.items():
                if pk != partition_key:
                    continue
                if row_key_prefix is not None and not rk.startswith(row_key_prefix):
                    continue
                rows.append(doc)
            rows.sort(key=lambda doc: doc.row_key)
            return DocumentQueryResult(documents=rows[:limit], total=len(rows[:limit]))

        def close(self) -> None:
            self._rows.clear()

    repo = ManagedWorkspaceRepository(_PlainDocumentStore())
    lookup = _FakeWorkspaceLookup({(_TENANT, _WORKSPACE): _workspace()})
    config_service = WorkspaceKnowledgeConfigurationService(repo, lookup)
    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config_service,
        {_OPERATION: _FakeHandler()},
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationRepositoryError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_conditional_store_required"


# --- Raw transport safety ---


def test_execute_signature_rejects_raw_transport_inputs() -> None:
    signature = inspect.signature(WorkspaceKnowledgeConfigurationMutationEngine.execute)
    forbidden = {
        "idempotency_key",
        "raw_idempotency_key",
        "if_match",
        "http_request",
    }
    assert forbidden.isdisjoint(signature.parameters)


def test_module_has_no_raw_transport_names() -> None:
    import local_workspace_application.workspaces.knowledge_configuration_mutation_engine as module

    source = inspect.getsource(module)
    for token in ("Idempotency-Key", "If-Match", "FastAPI", "HTTPException"):
        assert token not in source


# --- CAS atomicity and recovery classification ---


def test_revision_changes_between_validation_and_writer_slot_cas() -> None:
    store = _RevisionRaceBeforeWriterSlotCASStore()
    engine, repo, handler = _build_engine(store=store)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=4,
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine, expected_revision=4)
    assert exc.value.error_code == "configuration_revision_conflict"
    assert handler.stage_calls == 0
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == 5
    assert head.pending_mutation_id is None
    assert head.pending_revision is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert loaded.target_revision is None


def test_pre_publication_rollback_fails_when_head_release_cas_fails() -> None:
    store = _FailingHeadReleaseStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_mutation_cleanup_failed"
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id is not None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED


def test_recovery_head_release_cas_failure_does_not_report_aborted() -> None:
    store = _FailingHeadReleaseStore()
    engine, repo, handler = _build_engine(
        store=store,
        mutation_ids=["mutation-incomplete-release"],
    )
    mutation = _mutation_record(
        mutation_id="mutation-incomplete-release",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-incomplete-release",
            updated_at=_NOW,
        )
    )
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    handler.stage(
        repository=repo,
        mutation=mutation,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code in {
        "configuration_recovery_required",
        "configuration_mutation_cleanup_failed",
    }
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.pending_mutation_id == "mutation-incomplete-release"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED


def test_recovery_commits_other_mutation_does_not_replay_current_row() -> None:
    engine, repo, handler = _build_engine(
        mutation_ids=["mutation-a", "mutation-b", "mutation-c"],
    )
    mutation_a = _mutation_record(
        mutation_id="mutation-a",
        status=WorkspaceKnowledgeMutationStatusV1.RESERVED,
        target_revision=None,
        idempotency_key_hash=_SHA256,
    )
    mutation_b = _mutation_record(
        mutation_id="mutation-b",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        idempotency_key_hash=_SHA256_B,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation_a)
    repo.put_knowledge_configuration_mutation_if_absent(mutation_b)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=1,
            last_committed_mutation_id="other",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=mutation_b,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine, idempotency_key_hash=_SHA256)
    assert exc.value.error_code in {
        "configuration_revision_conflict",
        "configuration_recovery_required",
        "configuration_mutation_state_conflict",
    }
    loaded_a = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded_a is not None
    assert loaded_a.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED
    loaded_b = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256_B,
    )
    assert loaded_b is not None
    assert loaded_b.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_aborted_restart_race_won_by_committed_record() -> None:
    store = _AbortedRestartCommittedRaceStore()
    engine, repo, handler = _build_engine(
        store=store,
        mutation_ids=["mutation-restart-race", "mutation-unused"],
    )
    aborted = _mutation_record(
        mutation_id="mutation-old",
        status=WorkspaceKnowledgeMutationStatusV1.ABORTED,
        error_code="configuration_revision_conflict",
    )
    repo.put_knowledge_configuration_mutation_if_absent(aborted)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert result.configuration_revision == 1
    assert result.result_entity_id == _RESULT_ID
    assert handler.stage_calls == 0


def test_aborted_restart_race_won_by_reserved_record() -> None:
    store = _AbortedRestartReservedRaceStore()
    engine, repo, handler = _build_engine(
        store=store,
        mutation_ids=["mutation-restart-race", "mutation-unused"],
    )
    aborted = _mutation_record(
        mutation_id="mutation-old",
        status=WorkspaceKnowledgeMutationStatusV1.ABORTED,
        error_code="configuration_revision_conflict",
    )
    repo.put_knowledge_configuration_mutation_if_absent(aborted)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-race-winner"
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


# --- Publication-aware pre-cleanup guard ---


def test_concurrent_publication_and_finalization_during_prepared_cas() -> None:
    store = _PreparedCASConcurrentPublicationStore(finalize=True)
    engine, repo, handler = _build_engine(store=store)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert result.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert result.mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED
    assert result.configuration_revision == 1
    assert result.result_entity_type == _RESULT_TYPE
    assert result.result_entity_id == _RESULT_ID
    assert handler.cleanup_calls == 0
    assert len(handler.staged_rows) == 1
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == 1
    assert head.pending_mutation_id is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED


def test_concurrent_publication_without_finalization_returns_committed_replay() -> None:
    store = _PreparedCASConcurrentPublicationStore(finalize=False)
    engine, repo, handler = _build_engine(store=store)
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert handler.cleanup_calls == 0
    assert len(handler.staged_rows) == 1
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == 1
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert loaded.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED


def test_ownership_lost_without_publication_raises_recovery_required() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-owner-lost"])
    mutation = _mutation_record(
        mutation_id="mutation-owner-lost",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="other-mutation",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=mutation,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    stale = mutation.model_copy(update={"status": WorkspaceKnowledgeMutationStatusV1.RESERVED})
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._handle_pre_publication_failure(
            mutation=stale,
            handler=handler,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=_OPERATION,
            idempotency_key_hash=_SHA256,
            error_code="configuration_mutation_stage_failed",
        )
    assert exc.value.error_code == "configuration_recovery_required"
    assert handler.cleanup_calls == 0
    assert len(handler.staged_rows) == 1
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED


def test_idle_head_without_publication_raises_recovery_required() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-idle-head"])
    mutation = _mutation_record(
        mutation_id="mutation-idle-head",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=mutation,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    stale = mutation.model_copy(update={"status": WorkspaceKnowledgeMutationStatusV1.RESERVED})
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._handle_pre_publication_failure(
            mutation=stale,
            handler=handler,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=_OPERATION,
            idempotency_key_hash=_SHA256,
            error_code="configuration_mutation_stage_failed",
        )
    assert exc.value.error_code == "configuration_recovery_required"
    assert handler.cleanup_calls == 0
    assert len(handler.staged_rows) == 1
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED


def test_exact_committed_row_short_circuits_cleanup() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-committed"])
    committed = _mutation_record(
        mutation_id="mutation-committed",
        status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
        outcome=WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
        target_revision=1,
        committed_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
        committed_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(committed)
    handler.stage(
        repository=repo,
        mutation=committed,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    stale = committed.model_copy(
        update={"status": WorkspaceKnowledgeMutationStatusV1.RESERVED}
    )
    result = engine._handle_pre_publication_failure(
        mutation=stale,
        handler=handler,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code="configuration_mutation_stage_failed",
    )
    assert result is not None
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert handler.cleanup_calls == 0
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


# --- R3 durable cleanup fence ---


class _StalePublisherDuringCleanupHandler(_FakeHandler):
    def __init__(self) -> None:
        super().__init__()
        self.engine: WorkspaceKnowledgeConfigurationMutationEngine | None = None
        self.stale_mutation: WorkspaceKnowledgeMutationRecord | None = None
        self.stale_head: WorkspaceKnowledgeConfigurationHead | None = None
        self.stale_publisher_error: str | None = None

    def cleanup_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        assert self.engine is not None
        assert self.stale_mutation is not None
        assert self.stale_head is not None
        try:
            self.engine._publish_head(
                head=self.stale_head,
                mutation=self.stale_mutation,
                now=_NOW,
            )
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            self.stale_publisher_error = exc.error_code
        return super().cleanup_staged(
            repository=repository,
            mutation=mutation,
            inspection=inspection,
        )


def _is_cleanup_head_fence_cas(
    expected: DocumentRecord,
    replacement: DocumentRecord,
) -> bool:
    if "knowledge_configuration_head" not in expected.partition_key:
        return False
    if not expected.data.get("pending_mutation_id"):
        return False
    if replacement.data.get("pending_mutation_id") != expected.data.get("pending_mutation_id"):
        return False
    if replacement.data.get("pending_revision") != expected.data.get("pending_revision"):
        return False
    if replacement.data.get("committed_revision") != expected.data.get("committed_revision"):
        return False
    return replacement.data.get("updated_at") != expected.data.get("updated_at")


def _is_publication_head_cas(
    expected: DocumentRecord,
    replacement: DocumentRecord,
) -> bool:
    return (
        "knowledge_configuration_head" in expected.partition_key
        and expected.data.get("pending_mutation_id")
        and not replacement.data.get("pending_mutation_id")
    )


class _PublisherWinsHeadFenceRaceStore(InMemoryDocumentStore):
    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if _is_cleanup_head_fence_cas(expected, replacement):
            tenant_id = expected.data["tenant_id"]
            workspace_id = expected.data["workspace_id"]
            target_revision = expected.data["pending_revision"]
            mutation_id = expected.data["pending_mutation_id"]
            published = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "committed_revision": target_revision,
                        "pending_revision": None,
                        "pending_mutation_id": None,
                        "last_committed_mutation_id": mutation_id,
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=published)
            mutation_pk = (
                f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_mutation"
            )
            mutation_row_key = (
                f"{workspace_id}:{_OPERATION.value}:{_SHA256}"
            )
            mutation_doc = self.get(mutation_pk, mutation_row_key)
            if mutation_doc is not None:
                committed_doc = mutation_doc.model_copy(
                    update={
                        "data": {
                            **mutation_doc.data,
                            "status": "committed",
                            "outcome": "applied",
                            "committed_revision": target_revision,
                            "result_entity_type": _RESULT_TYPE,
                            "result_entity_id": _RESULT_ID,
                            "committed_at": replacement.data.get("updated_at"),
                            "error_code": None,
                        }
                    }
                )
                super().replace_if_match(expected=mutation_doc, replacement=committed_doc)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _PublisherRetryRevalidationStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__()
        self.publication_cas_attempts = 0
        self._fence_applied = False

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and replacement.data.get("status") == "recovery_required"
            and replacement.data.get("error_code") == "configuration_mutation_cleanup_fenced"
        ):
            self._fence_applied = True
        if _is_publication_head_cas(expected, replacement):
            self.publication_cas_attempts += 1
            if self.publication_cas_attempts == 1 and not self._fence_applied:
                fenced_head = expected.model_copy(
                    update={
                        "data": {
                            **expected.data,
                            "updated_at": replacement.data.get("updated_at"),
                        }
                    }
                )
                super().replace_if_match(expected=expected, replacement=fenced_head)
                tenant_id = expected.data["tenant_id"]
                workspace_id = expected.data["workspace_id"]
                mutation_pk = (
                    f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_mutation"
                )
                mutation_row_key = (
                    f"{workspace_id}:{_OPERATION.value}:{_SHA256}"
                )
                mutation_doc = self.get(mutation_pk, mutation_row_key)
                if mutation_doc is not None and mutation_doc.data.get("status") == "prepared":
                    fenced_mutation = mutation_doc.model_copy(
                        update={
                            "data": {
                                **mutation_doc.data,
                                "status": "recovery_required",
                                "error_code": "configuration_mutation_cleanup_fenced",
                            }
                        }
                    )
                    super().replace_if_match(
                        expected=mutation_doc,
                        replacement=fenced_mutation,
                    )
                    self._fence_applied = True
                return False
        return super().replace_if_match(expected=expected, replacement=replacement)


def test_cleanup_fence_wins_against_stale_publisher() -> None:
    handler = _StalePublisherDuringCleanupHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(handler=handler)
    prepared = _mutation_record(
        mutation_id="mutation-stale-pub",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(prepared)
    head = WorkspaceKnowledgeConfigurationHead(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        committed_revision=0,
        pending_revision=1,
        pending_mutation_id="mutation-stale-pub",
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_head_if_absent(head)
    handler.stage(
        repository=repo,
        mutation=prepared,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    handler.engine = engine
    handler.stale_mutation = prepared
    handler.stale_head = head
    result = engine._handle_pre_publication_failure(
        mutation=prepared,
        handler=handler,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code="configuration_mutation_stage_failed",
    )
    assert result is None
    assert handler.stale_publisher_error == "configuration_recovery_required"
    assert handler.cleanup_calls == 1
    assert not handler.staged_rows
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 0
    assert loaded_head.pending_mutation_id is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED


def test_publisher_wins_head_fence_cas_race() -> None:
    store = _PublisherWinsHeadFenceRaceStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    prepared = _mutation_record(
        mutation_id="mutation-pub-wins",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(prepared)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-pub-wins",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=prepared,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    result = engine._handle_pre_publication_failure(
        mutation=prepared,
        handler=handler,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code="configuration_mutation_stage_failed",
    )
    assert result is not None
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert handler.cleanup_calls == 0
    assert len(handler.staged_rows) == 1
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 1
    assert loaded_head.pending_mutation_id is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_publisher_retry_revalidates_mutation() -> None:
    store = _PublisherRetryRevalidationStore()
    engine, repo, handler = _build_engine(store=store)
    prepared = _mutation_record(
        mutation_id="mutation-retry-reval",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(prepared)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-retry-reval",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=prepared,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert head is not None
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._publish_head(head=head, mutation=prepared, now=_NOW)
    assert exc.value.error_code == "configuration_recovery_required"
    assert store.publication_cas_attempts == 1


def test_recovery_resumes_abandoned_cleanup_fence() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-abandoned-fence"])
    fenced = _mutation_record(
        mutation_id="mutation-abandoned-fence",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code="configuration_mutation_cleanup_fenced",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-abandoned-fence",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=fenced,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 0
    assert loaded_head.pending_mutation_id is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert not handler.staged_rows


def test_fixed_clock_still_invalidates_stale_head() -> None:
    clock = _ControlledClock(_NOW)
    engine, repo, _handler = _build_engine(clock=clock)
    mutation = _mutation_record(
        mutation_id="mutation-fixed-clock",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    head = WorkspaceKnowledgeConfigurationHead(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        committed_revision=0,
        pending_revision=1,
        pending_mutation_id="mutation-fixed-clock",
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_head_if_absent(head)
    fence_or_replay = engine._acquire_staged_cleanup_fence(
        staged_mutation=mutation,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        now=clock.now(),
    )
    assert not isinstance(fence_or_replay, WorkspaceKnowledgeMutationExecutionResult)
    assert fence_or_replay.mutation.updated_at > mutation.updated_at
    assert fence_or_replay.head.updated_at > head.updated_at
    stale_publish_head = head.model_copy(
        update={
            "committed_revision": 1,
            "pending_revision": None,
            "pending_mutation_id": None,
            "last_committed_mutation_id": mutation.mutation_id,
            "updated_at": _NOW,
        }
    )
    assert not repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=stale_publish_head,
    )


def test_generic_recovery_required_remains_recoverable() -> None:
    engine, repo, handler = _build_engine(mutation_ids=["mutation-generic-recovery"])
    recovery_required = _mutation_record(
        mutation_id="mutation-generic-recovery",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code="configuration_mutation_stage_failed",
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(recovery_required)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-generic-recovery",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=recovery_required,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    assert recovery.mutation is not None
    assert recovery.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_normal_valid_rollback_under_cleanup_fence() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(handler=handler)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_mutation_stage_failed"
    assert handler.cleanup_calls == 1
    assert not handler.staged_rows
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.pending_mutation_id is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED


def test_cleanup_failure_after_fence_leaves_fenced_pending_state() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    handler.cleanup_returns = False
    engine, repo, _handler = _build_engine(handler=handler)
    prepared = _mutation_record(
        mutation_id="mutation-cleanup-fail-fence",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(prepared)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-cleanup-fail-fence",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=prepared,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._handle_pre_publication_failure(
            mutation=prepared,
            handler=handler,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=_OPERATION,
            idempotency_key_hash=_SHA256,
            error_code="configuration_mutation_stage_failed",
        )
    assert exc.value.error_code == "configuration_mutation_cleanup_failed"
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.pending_mutation_id is not None
    assert loaded_head.committed_revision == 0
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED
