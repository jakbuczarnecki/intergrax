# © Artur Czarnecki. All rights reserved.

"""Tests for Workspace Knowledge Configuration mutation engine."""

from __future__ import annotations

import inspect
import threading
from enum import IntEnum
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
        self.cleanup_raises: Exception | None = None
        self.post_cleanup_state: WorkspaceKnowledgeStageStateV1 | None = None
        self.staged_rows: list[WorkspaceConnectionAttachment] = []
        self.projection_revision_sequence: list[int] | None = None
        self._projection_reads = 0
        self.stage_entered = threading.Event()
        self.stage_release = threading.Event()
        self.stage_release.set()
        self._stage_lock = threading.Lock()

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
        with self._stage_lock:
            self.stage_calls += 1
        self.stage_target_revision_at_call = mutation.target_revision
        self.stage_entered.set()
        if not self.stage_release.is_set():
            self.stage_release.wait(timeout=5)
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
        if self.cleanup_calls > 0 and self.post_cleanup_state is not None:
            if self.post_cleanup_state is WorkspaceKnowledgeStageStateV1.ABSENT:
                return WorkspaceKnowledgeStageInspection(
                    state=WorkspaceKnowledgeStageStateV1.ABSENT,
                )
            if self.post_cleanup_state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
                return WorkspaceKnowledgeStageInspection(
                    state=self.post_cleanup_state,
                    result_entity_type=self.inspection_type,
                    result_entity_id=self.inspection_id,
                )
            return WorkspaceKnowledgeStageInspection(state=self.post_cleanup_state)
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
        if self.cleanup_raises is not None:
            raise self.cleanup_raises
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


class _FailingAbortedFinalizeAfterReleaseStore(InMemoryDocumentStore):
    """Head release succeeds; ABORTED mutation CAS fails for bounded attempts."""

    def __init__(self, *, fail_remaining: int = 3) -> None:
        super().__init__()
        self._fail_remaining = fail_remaining
        self._head_released = False

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
            self._head_released = True
            return super().replace_if_match(expected=expected, replacement=replacement)
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and self._head_released
            and expected.data.get("status") == "recovery_required"
            and replacement.data.get("status") == "aborted"
        ):
            if self._fail_remaining > 0:
                self._fail_remaining -= 1
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
    claim_ids: list[str] | None = None,
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
    claim_factory = _ControlledMutationIds(
        claim_ids or ["claim-1", "claim-2", "claim-3", "claim-4", "claim-5", "claim-6"]
    )
    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config_service,
        handler_map,
        clock=controlled_clock.now,
        mutation_id_factory=id_factory.next_id,
        stage_claim_id_factory=claim_factory.next_id,
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
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.error_code == "configuration_recovery_required"
    assert handler.cleanup_calls == 0
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


def _is_strict_publication_head_cas(
    expected: DocumentRecord,
    replacement: DocumentRecord,
) -> bool:
    return (
        _is_publication_head_cas(expected, replacement)
        and replacement.data.get("committed_revision", 0)
        > expected.data.get("committed_revision", 0)
    )


def _is_head_release_cas(
    expected: DocumentRecord,
    replacement: DocumentRecord,
) -> bool:
    return (
        _is_publication_head_cas(expected, replacement)
        and replacement.data.get("committed_revision")
        == expected.data.get("committed_revision")
    )


_CLEANUP_FENCE_ERROR = "configuration_mutation_cleanup_fenced"


def _is_cleanup_fenced_mutation_doc(doc: DocumentRecord) -> bool:
    return (
        doc.data.get("status") == "recovery_required"
        and doc.data.get("error_code") == _CLEANUP_FENCE_ERROR
    )


class _TransitionRecordingStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__()
        self.mutation_transitions: list[
            tuple[str | None, str | None, str | None, str | None]
        ] = []
        self.mutation_cas_attempts: list[
            tuple[str | None, str | None, str | None, str | None]
        ] = []
        self.cleanup_fenced_to_prepared_attempts = 0
        self.cleanup_fenced_to_generic_recovery_attempts = 0
        self.cleanup_fenced_to_committed_attempts = 0
        self.publication_head_cas_attempts = 0
        self.head_fence_cas_attempts = 0
        self.head_release_cas_attempts = 0

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if "knowledge_configuration_mutation" in expected.partition_key:
            src_status = expected.data.get("status")
            src_error = expected.data.get("error_code")
            dst_status = replacement.data.get("status")
            dst_error = replacement.data.get("error_code")
            self.mutation_cas_attempts.append(
                (
                    expected.data.get("mutation_id"),
                    replacement.data.get("mutation_id"),
                    src_status,
                    dst_status,
                )
            )
            if _is_cleanup_fenced_mutation_doc(expected):
                self.mutation_transitions.append(
                    (src_status, src_error, dst_status, dst_error)
                )
                if dst_status == "prepared":
                    self.cleanup_fenced_to_prepared_attempts += 1
                if (
                    dst_status == "recovery_required"
                    and dst_error != _CLEANUP_FENCE_ERROR
                ):
                    self.cleanup_fenced_to_generic_recovery_attempts += 1
                if dst_status == "committed":
                    self.cleanup_fenced_to_committed_attempts += 1
        if _is_cleanup_head_fence_cas(expected, replacement):
            self.head_fence_cas_attempts += 1
        if _is_strict_publication_head_cas(expected, replacement):
            self.publication_head_cas_attempts += 1
        elif _is_head_release_cas(expected, replacement):
            self.head_release_cas_attempts += 1
        return super().replace_if_match(expected=expected, replacement=replacement)


class _TransitionRecordingFailingHeadReleaseStore(_TransitionRecordingStore):
    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if _is_head_release_cas(expected, replacement):
            self.head_release_cas_attempts += 1
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


def _assert_cleanup_fence_transitions_monotonic(
    store: _TransitionRecordingStore,
) -> None:
    for src_status, src_error, dst_status, dst_error in store.mutation_transitions:
        if src_error != _CLEANUP_FENCE_ERROR:
            continue
        allowed_refresh = (
            src_status == "recovery_required"
            and dst_status == "recovery_required"
            and dst_error == _CLEANUP_FENCE_ERROR
        )
        allowed_abort = (
            src_status == "recovery_required" and dst_status == "aborted"
        )
        if allowed_refresh or allowed_abort:
            continue
        forbidden = (
            dst_status in {"reserved", "prepared", "committed"}
            or (
                dst_status == "recovery_required"
                and dst_error != _CLEANUP_FENCE_ERROR
            )
        )
        assert not forbidden, (
            f"illegal cleanup-fenced transition: "
            f"{src_status}/{src_error} -> {dst_status}/{dst_error}"
        )


def _seed_cleanup_fenced_pending(
    *,
    repo: ManagedWorkspaceRepository,
    mutation_id: str = "mutation-cleanup-fence",
    target_revision: int = 1,
) -> WorkspaceKnowledgeMutationRecord:
    fenced = _mutation_record(
        mutation_id=mutation_id,
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code=_CLEANUP_FENCE_ERROR,
        target_revision=target_revision,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=target_revision,
            pending_mutation_id=mutation_id,
            updated_at=_NOW,
        )
    )
    return fenced


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
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 0
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.error_code == _CLEANUP_FENCE_ERROR


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
    store = _TransitionRecordingStore()
    engine, repo, handler = _build_engine(
        store=store,
        mutation_ids=["mutation-generic-recovery"],
    )
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
    assert recovery.mutation.error_code != _CLEANUP_FENCE_ERROR
    assert handler.cleanup_calls == 0
    assert store.publication_head_cas_attempts == 1
    assert store.cleanup_fenced_to_prepared_attempts == 0
    assert store.cleanup_fenced_to_committed_attempts == 0


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


def test_cleanup_failure_preserves_fence_and_recovery_resumes() -> None:
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
    assert loaded_head.pending_mutation_id == "mutation-cleanup-fail-fence"
    assert loaded_head.pending_revision == 1
    assert loaded_head.committed_revision == 0
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.error_code == "configuration_mutation_cleanup_fenced"
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED
    handler.cleanup_returns = True
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert not handler.staged_rows
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.pending_mutation_id is None
    assert loaded_head.pending_revision is None
    assert loaded_head.committed_revision == 0
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_complete_valid_cleanup_failure_does_not_publish_after_restart() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    handler.cleanup_returns = False
    engine, repo, _handler = _build_engine(handler=handler)
    prepared = _mutation_record(
        mutation_id="mutation-complete-fail",
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
            pending_mutation_id="mutation-complete-fail",
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
    assert len(handler.staged_rows) == 1
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 0
    assert loaded_head.pending_mutation_id == "mutation-complete-fail"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.error_code == "configuration_mutation_cleanup_fenced"
    handler.cleanup_returns = True
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert not handler.staged_rows
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 0
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED


def test_cleanup_exception_preserves_fence_and_recovery_aborts() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    handler.cleanup_raises = RuntimeError("provider cleanup exploded")
    engine, repo, _handler = _build_engine(handler=handler)
    prepared = _mutation_record(
        mutation_id="mutation-cleanup-exc",
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
            pending_mutation_id="mutation-cleanup-exc",
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
    assert loaded.error_code == "configuration_mutation_cleanup_fenced"
    handler.cleanup_raises = None
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED


def test_post_cleanup_verification_failure_preserves_fence() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    handler.post_cleanup_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(handler=handler)
    prepared = _mutation_record(
        mutation_id="mutation-post-verify",
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
            pending_mutation_id="mutation-post-verify",
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
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.error_code == "configuration_mutation_cleanup_fenced"
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.pending_mutation_id is not None
    handler.post_cleanup_state = None
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED


def test_head_release_failure_preserves_fence_and_recovers() -> None:
    store = _FailingHeadReleaseStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    prepared = _mutation_record(
        mutation_id="mutation-head-release",
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
            pending_mutation_id="mutation-head-release",
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
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.error_code == "configuration_mutation_cleanup_fenced"
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.pending_mutation_id is not None
    assert not handler.staged_rows
    normal_engine, normal_repo, normal_handler = _build_engine(handler=_FakeHandler())
    normal_repo.put_knowledge_configuration_mutation_if_absent(loaded)
    normal_repo.put_knowledge_configuration_head_if_absent(loaded_head)
    recovery = normal_engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert normal_handler.cleanup_calls == 0


def test_crash_after_head_release_before_aborted_recovers() -> None:
    handler = _FakeHandler()
    engine, repo, _handler = _build_engine(handler=handler)
    fenced = _mutation_record(
        mutation_id="mutation-orphan-absent",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        target_revision=1,
        error_code="configuration_mutation_cleanup_fenced",
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
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
    assert loaded.error_code == "configuration_mutation_stage_failed"


def test_orphaned_cleanup_fence_with_staged_data_fails_closed() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(handler=handler)
    fenced = _mutation_record(
        mutation_id="mutation-orphan-staged",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        target_revision=1,
        error_code="configuration_mutation_cleanup_fenced",
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
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
        mutation=fenced,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    assert handler.cleanup_calls == 0
    assert handler.staged_rows
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.error_code == "configuration_mutation_cleanup_fenced"
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_cleanup_fenced_row_excluded_from_post_publication_repair() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    engine, repo, _handler = _build_engine(handler=handler)
    fenced = _mutation_record(
        mutation_id="mutation-post-contradiction",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        target_revision=1,
        error_code="configuration_mutation_cleanup_fenced",
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
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
        mutation=fenced,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    assert handler.cleanup_calls == 0
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 1
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert loaded.error_code == "configuration_mutation_cleanup_fenced"


def test_multiple_orphaned_cleanup_fences_fail_closed() -> None:
    engine, repo, _handler = _build_engine()
    fenced_a = _mutation_record(
        mutation_id="mutation-orphan-a",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        target_revision=1,
        error_code="configuration_mutation_cleanup_fenced",
        idempotency_key_hash=_SHA256,
    )
    fenced_b = _mutation_record(
        mutation_id="mutation-orphan-b",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        target_revision=2,
        error_code="configuration_mutation_cleanup_fenced",
        idempotency_key_hash=_SHA256_B,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced_a)
    repo.put_knowledge_configuration_mutation_if_absent(fenced_b)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    loaded_a = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    loaded_b = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256_B,
    )
    assert loaded_a is not None
    assert loaded_b is not None
    assert loaded_a.error_code == "configuration_mutation_cleanup_fenced"
    assert loaded_b.error_code == "configuration_mutation_cleanup_fenced"


# --- R5 cleanup-fence invariant closure ---


def test_cleanup_fenced_ownership_conflict_preserves_marker() -> None:
    store = _TransitionRecordingStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    fenced = _seed_cleanup_fenced_pending(
        repo=repo,
        mutation_id="mutation-cleanup-ownership",
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.error_code == _CLEANUP_FENCE_ERROR
    assert loaded.target_revision == 1
    assert loaded.mutation_id == fenced.mutation_id
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 0
    assert loaded_head.pending_revision == 1
    assert loaded_head.pending_mutation_id == fenced.mutation_id
    assert loaded_head.updated_at > _NOW
    assert handler.cleanup_calls == 0
    assert store.cleanup_fenced_to_prepared_attempts == 0
    assert store.cleanup_fenced_to_generic_recovery_attempts == 0
    assert store.cleanup_fenced_to_committed_attempts == 0
    assert store.publication_head_cas_attempts == 0
    assert store.head_release_cas_attempts == 0
    _assert_cleanup_fence_transitions_monotonic(store)


def test_cleanup_fenced_ownership_conflict_then_incomplete_resumes_rollback() -> None:
    store = _TransitionRecordingStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    fenced = _seed_cleanup_fenced_pending(
        repo=repo,
        mutation_id="mutation-cleanup-resume",
    )
    handler.stage(
        repository=repo,
        mutation=fenced,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    restart_engine, _, restart_handler = _build_engine(store=store, handler=handler)
    recovery = restart_engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert restart_handler.cleanup_calls == 1
    assert not restart_handler.staged_rows
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
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.PREPARED
    assert loaded.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert store.cleanup_fenced_to_prepared_attempts == 0
    assert store.cleanup_fenced_to_committed_attempts == 0
    assert store.publication_head_cas_attempts == 0
    _assert_cleanup_fence_transitions_monotonic(store)


def test_cleanup_fenced_ownership_conflict_then_complete_valid_rolls_back() -> None:
    store = _TransitionRecordingStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    fenced = _seed_cleanup_fenced_pending(
        repo=repo,
        mutation_id="mutation-cleanup-complete",
    )
    handler.stage(
        repository=repo,
        mutation=fenced,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    restart_engine, _, restart_handler = _build_engine(store=store, handler=handler)
    recovery = restart_engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert restart_handler.cleanup_calls == 1
    assert not restart_handler.staged_rows
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
    assert store.cleanup_fenced_to_prepared_attempts == 0
    assert store.publication_head_cas_attempts == 0
    _assert_cleanup_fence_transitions_monotonic(store)


@pytest.mark.parametrize(
    "requested_error_code",
    [
        "configuration_recovery_required",
        "configuration_mutation_cleanup_failed",
    ],
)
def test_mark_recovery_required_cannot_demote_cleanup_fence(
    requested_error_code: str,
) -> None:
    store = _TransitionRecordingStore()
    engine, repo, _handler = _build_engine(store=store)
    fenced = _mutation_record(
        mutation_id="mutation-mark-guard",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code=_CLEANUP_FENCE_ERROR,
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    engine._mark_recovery_required(
        mutation=fenced,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code=requested_error_code,
        now=_NOW,
    )
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == fenced.mutation_id
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.error_code == _CLEANUP_FENCE_ERROR
    assert loaded.target_revision == 1
    assert store.cleanup_fenced_to_generic_recovery_attempts == 0
    _assert_cleanup_fence_transitions_monotonic(store)


def test_cleanup_fence_transitions_are_monotonic() -> None:
    def _run_ownership_conflict() -> None:
        store = _TransitionRecordingStore()
        handler = _FakeHandler()
        handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
        engine, repo, _handler = _build_engine(store=store, handler=handler)
        _seed_cleanup_fenced_pending(repo=repo, mutation_id="mutation-mono-owner")
        with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
            engine.recover_workspace_knowledge_mutation(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
            )
        _assert_cleanup_fence_transitions_monotonic(store)

    def _run_cleanup_failure() -> None:
        store = _TransitionRecordingStore()
        handler = _FakeHandler()
        handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
        handler.cleanup_returns = False
        engine, repo, _handler = _build_engine(store=store, handler=handler)
        prepared = _mutation_record(
            mutation_id="mutation-mono-fail",
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
                pending_mutation_id="mutation-mono-fail",
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
        with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
            engine._handle_pre_publication_failure(
                mutation=prepared,
                handler=handler,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation=_OPERATION,
                idempotency_key_hash=_SHA256,
                error_code="configuration_mutation_stage_failed",
            )
        _assert_cleanup_fence_transitions_monotonic(store)

    def _run_cleanup_exception() -> None:
        store = _TransitionRecordingStore()
        handler = _FakeHandler()
        handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
        handler.cleanup_raises = RuntimeError("cleanup boom")
        engine, repo, _handler = _build_engine(store=store, handler=handler)
        prepared = _mutation_record(
            mutation_id="mutation-mono-exc",
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
                pending_mutation_id="mutation-mono-exc",
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
        with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
            engine._handle_pre_publication_failure(
                mutation=prepared,
                handler=handler,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation=_OPERATION,
                idempotency_key_hash=_SHA256,
                error_code="configuration_mutation_stage_failed",
            )
        _assert_cleanup_fence_transitions_monotonic(store)

    def _run_post_cleanup_verification_failure() -> None:
        store = _TransitionRecordingStore()
        handler = _FakeHandler()
        handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
        handler.post_cleanup_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
        engine, repo, _handler = _build_engine(store=store, handler=handler)
        prepared = _mutation_record(
            mutation_id="mutation-mono-verify",
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
                pending_mutation_id="mutation-mono-verify",
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
        with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
            engine._handle_pre_publication_failure(
                mutation=prepared,
                handler=handler,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation=_OPERATION,
                idempotency_key_hash=_SHA256,
                error_code="configuration_mutation_stage_failed",
            )
        _assert_cleanup_fence_transitions_monotonic(store)

    def _run_head_release_failure() -> None:
        store = _TransitionRecordingFailingHeadReleaseStore()
        handler = _FakeHandler()
        handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
        engine, repo, _handler = _build_engine(store=store, handler=handler)
        prepared = _mutation_record(
            mutation_id="mutation-mono-release",
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
                pending_mutation_id="mutation-mono-release",
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
        with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
            engine._handle_pre_publication_failure(
                mutation=prepared,
                handler=handler,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation=_OPERATION,
                idempotency_key_hash=_SHA256,
                error_code="configuration_mutation_stage_failed",
            )
        _assert_cleanup_fence_transitions_monotonic(store)

    def _run_successful_resumed_cleanup() -> None:
        store = _TransitionRecordingStore()
        handler = _FakeHandler()
        handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
        engine, repo, _handler = _build_engine(store=store, handler=handler)
        fenced = _seed_cleanup_fenced_pending(
            repo=repo,
            mutation_id="mutation-mono-success",
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
        _assert_cleanup_fence_transitions_monotonic(store)

    def _run_orphaned_absent_finalization() -> None:
        store = _TransitionRecordingStore()
        handler = _FakeHandler()
        engine, repo, _handler = _build_engine(store=store, handler=handler)
        fenced = _mutation_record(
            mutation_id="mutation-mono-orphan",
            status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
            error_code=_CLEANUP_FENCE_ERROR,
            target_revision=1,
        )
        repo.put_knowledge_configuration_mutation_if_absent(fenced)
        repo.put_knowledge_configuration_head_if_absent(
            WorkspaceKnowledgeConfigurationHead(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                committed_revision=0,
                updated_at=_NOW,
            )
        )
        recovery = engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
        _assert_cleanup_fence_transitions_monotonic(store)

    for runner in (
        _run_ownership_conflict,
        _run_cleanup_failure,
        _run_cleanup_exception,
        _run_post_cleanup_verification_failure,
        _run_head_release_failure,
        _run_successful_resumed_cleanup,
        _run_orphaned_absent_finalization,
    ):
        runner()


# --- R6 head-fence reacquisition and exact mutation identity ---


def _stale_publish_head_from_pending(
    head: WorkspaceKnowledgeConfigurationHead,
    *,
    mutation_id: str,
) -> WorkspaceKnowledgeConfigurationHead:
    target_revision = head.pending_revision
    assert target_revision is not None
    return head.model_copy(
        update={
            "committed_revision": target_revision,
            "pending_revision": None,
            "pending_mutation_id": None,
            "last_committed_mutation_id": mutation_id,
            "updated_at": head.updated_at,
        }
    )


def test_abandoned_cleanup_fence_reacquires_head_before_ownership_conflict() -> None:
    store = _TransitionRecordingStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    fenced = _seed_cleanup_fenced_pending(
        repo=repo,
        mutation_id="mutation-r6-abandoned",
    )
    stale_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert stale_head is not None
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.error_code == _CLEANUP_FENCE_ERROR
    assert loaded.mutation_id == fenced.mutation_id
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.committed_revision == 0
    assert loaded_head.pending_revision == 1
    assert loaded_head.pending_mutation_id == fenced.mutation_id
    assert loaded_head.updated_at > stale_head.updated_at
    assert _handler.cleanup_calls == 0
    assert store.head_release_cas_attempts == 0
    assert store.publication_head_cas_attempts == 0
    assert store.head_fence_cas_attempts >= 1
    stale_publish = _stale_publish_head_from_pending(
        stale_head,
        mutation_id=fenced.mutation_id,
    )
    assert not repo.replace_knowledge_configuration_head_if_match(
        expected=stale_head,
        replacement=stale_publish,
    )
    _assert_cleanup_fence_transitions_monotonic(store)


def test_abandoned_cleanup_fence_fixed_clock_refreshes_head_before_conflict() -> None:
    clock = _ControlledClock(_NOW)
    store = _TransitionRecordingStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    engine, repo, _handler = _build_engine(
        store=store,
        handler=handler,
        clock=clock,
    )
    fenced = _mutation_record(
        mutation_id="mutation-r6-fixed-clock",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code=_CLEANUP_FENCE_ERROR,
        target_revision=1,
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    stale_head = WorkspaceKnowledgeConfigurationHead(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        committed_revision=0,
        pending_revision=1,
        pending_mutation_id="mutation-r6-fixed-clock",
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_head_if_absent(stale_head)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.updated_at > fenced.updated_at
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.updated_at > stale_head.updated_at
    stale_publish = _stale_publish_head_from_pending(
        stale_head,
        mutation_id=fenced.mutation_id,
    )
    assert not repo.replace_knowledge_configuration_head_if_match(
        expected=stale_head,
        replacement=stale_publish,
    )


class _RecoveryPublisherWinsHeadFenceRaceStore(_PublisherWinsHeadFenceRaceStore):
    pass


def test_recovery_publisher_wins_head_fence_race() -> None:
    store = _RecoveryPublisherWinsHeadFenceRaceStore()
    handler = _FakeHandler()
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    fenced = _seed_cleanup_fenced_pending(
        repo=repo,
        mutation_id="mutation-r6-pub-win",
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
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    assert _handler.cleanup_calls == 0
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
    assert loaded.mutation_id == fenced.mutation_id


@pytest.mark.parametrize(
    "m2_status",
    [
        WorkspaceKnowledgeMutationStatusV1.RESERVED,
        WorkspaceKnowledgeMutationStatusV1.PREPARED,
    ],
)
def test_stale_mark_recovery_required_cannot_modify_newer_attempt(
    m2_status: WorkspaceKnowledgeMutationStatusV1,
) -> None:
    store = _TransitionRecordingStore()
    engine, repo, _handler = _build_engine(store=store)
    stale_m1 = _mutation_record(
        mutation_id="mutation-r6-stale-mark-1",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    m2_kwargs: dict[str, object] = {
        "mutation_id": "mutation-r6-stale-mark-2",
        "status": m2_status,
        "target_revision": None,
    }
    if m2_status is WorkspaceKnowledgeMutationStatusV1.PREPARED:
        m2_kwargs.update(
            {
                "target_revision": 2,
                "result_entity_type": _RESULT_TYPE,
                "result_entity_id": _RESULT_ID,
            }
        )
    m2 = _mutation_record(**m2_kwargs)
    repo.put_knowledge_configuration_mutation_if_absent(m2)
    before = m2.model_copy()
    engine._mark_recovery_required(
        mutation=stale_m1,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code="configuration_recovery_required",
        now=_NOW,
    )
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded == before
    assert loaded is not None
    assert loaded.status == m2_status
    assert loaded.mutation_id == "mutation-r6-stale-mark-2"
    assert not any(
        dst_id == "mutation-r6-stale-mark-2"
        for _src_id, dst_id, _src_status, _dst_status in store.mutation_cas_attempts
    )


def test_full_stale_mark_recovery_required_race_preserves_newer_attempt() -> None:
    store = _TransitionRecordingStore()
    engine, repo, handler = _build_engine(store=store)
    prepared = _mutation_record(
        mutation_id="mutation-r6-full-1",
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
            pending_mutation_id="mutation-r6-full-1",
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
    stale_m1 = prepared.model_copy()
    engine._handle_pre_publication_failure(
        mutation=prepared,
        handler=handler,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code="configuration_mutation_stage_failed",
    )
    aborted = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert aborted is not None
    assert aborted.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    m2 = _mutation_record(
        mutation_id="mutation-r6-full-2",
        status=WorkspaceKnowledgeMutationStatusV1.RESERVED,
        target_revision=None,
    )
    assert repo.replace_knowledge_configuration_mutation_if_match(
        expected=aborted,
        replacement=m2,
    )
    before = m2.model_copy()
    engine._mark_recovery_required(
        mutation=stale_m1,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code="configuration_recovery_required",
        now=_NOW,
    )
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded == before
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RESERVED
    assert loaded.mutation_id == "mutation-r6-full-2"


@pytest.mark.parametrize(
    "m2_status",
    [
        WorkspaceKnowledgeMutationStatusV1.RESERVED,
        WorkspaceKnowledgeMutationStatusV1.PREPARED,
        WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        WorkspaceKnowledgeMutationStatusV1.ABORTED,
    ],
)
def test_stale_confirm_mutation_aborted_cannot_abort_newer_attempt(
    m2_status: WorkspaceKnowledgeMutationStatusV1,
) -> None:
    store = _TransitionRecordingStore()
    engine, repo, _handler = _build_engine(store=store)
    stale_m1 = _mutation_record(
        mutation_id="mutation-r6-stale-abort-1",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code=_CLEANUP_FENCE_ERROR,
        target_revision=1,
    )
    m2_kwargs: dict[str, object] = {
        "mutation_id": "mutation-r6-stale-abort-2",
        "status": m2_status,
        "target_revision": None,
    }
    if m2_status is WorkspaceKnowledgeMutationStatusV1.PREPARED:
        m2_kwargs.update(
            {
                "target_revision": 2,
                "result_entity_type": _RESULT_TYPE,
                "result_entity_id": _RESULT_ID,
            }
        )
    if m2_status is WorkspaceKnowledgeMutationStatusV1.ABORTED:
        m2_kwargs["error_code"] = "configuration_mutation_stage_failed"
    if m2_status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED:
        m2_kwargs.update(
            {
                "target_revision": 2,
                "error_code": "configuration_recovery_required",
            }
        )
    m2 = _mutation_record(**m2_kwargs)
    repo.put_knowledge_configuration_mutation_if_absent(m2)
    before = m2.model_copy()
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._confirm_mutation_aborted(
            mutation=stale_m1,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=_OPERATION,
            idempotency_key_hash=_SHA256,
            error_code="configuration_mutation_stage_failed",
            now=_NOW,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded == before
    assert loaded is not None
    assert loaded.status == m2_status
    assert loaded.mutation_id == "mutation-r6-stale-abort-2"
    assert not any(
        dst_status == "aborted"
        for _src_id, _dst_id, _src_status, dst_status in store.mutation_cas_attempts
        if _dst_id == "mutation-r6-stale-abort-2"
    )


def test_full_stale_confirm_mutation_aborted_race_preserves_newer_attempt() -> None:
    store = _TransitionRecordingStore()
    engine, repo, _handler = _build_engine(store=store)
    fenced = _mutation_record(
        mutation_id="mutation-r6-delay-1",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code=_CLEANUP_FENCE_ERROR,
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
    )
    stale_m1 = fenced.model_copy()
    aborted_m1 = fenced.model_copy(
        update={
            "status": WorkspaceKnowledgeMutationStatusV1.ABORTED,
            "error_code": "configuration_mutation_stage_failed",
            "updated_at": _NOW,
        }
    )
    assert repo.replace_knowledge_configuration_mutation_if_match(
        expected=fenced,
        replacement=aborted_m1,
    )
    m2 = _mutation_record(
        mutation_id="mutation-r6-delay-2",
        status=WorkspaceKnowledgeMutationStatusV1.RESERVED,
        target_revision=None,
    )
    assert repo.replace_knowledge_configuration_mutation_if_match(
        expected=aborted_m1,
        replacement=m2,
    )
    before = m2.model_copy()
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._confirm_mutation_aborted(
            mutation=stale_m1,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=_OPERATION,
            idempotency_key_hash=_SHA256,
            error_code="configuration_mutation_stage_failed",
            now=_NOW,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded == before
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RESERVED
    assert loaded.mutation_id == "mutation-r6-delay-2"


def test_same_attempt_abort_finalization_still_works() -> None:
    engine, repo, _handler = _build_engine()
    fenced = _mutation_record(
        mutation_id="mutation-r6-same-abort",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code=_CLEANUP_FENCE_ERROR,
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    aborted = engine._confirm_mutation_aborted(
        mutation=fenced,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        error_code="configuration_mutation_stage_failed",
        now=_NOW,
    )
    assert aborted.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    assert aborted.mutation_id == fenced.mutation_id
    assert aborted.target_revision == 1
    assert aborted.error_code == "configuration_mutation_stage_failed"


class _DifferentAttemptPreparedCASStore(InMemoryDocumentStore):
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
            and expected.data.get("mutation_id") == "mutation-r6-stage-1"
        ):
            m2 = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "mutation_id": "mutation-r6-stage-2",
                        "status": "prepared",
                        "stage_claim_id": None,
                        "target_revision": expected.data.get("target_revision"),
                        "result_entity_type": _RESULT_TYPE,
                        "result_entity_id": _RESULT_ID,
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=m2)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


def test_failed_prepared_cas_does_not_adopt_newer_attempt() -> None:
    store = _DifferentAttemptPreparedCASStore()
    engine, repo, handler = _build_engine(store=store)
    reserved = _mutation_record(
        mutation_id="mutation-r6-stage-1",
        status=WorkspaceKnowledgeMutationStatusV1.RESERVED,
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(reserved)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-r6-stage-1",
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._stage_and_prepare(
            mutation=reserved,
            handler=handler,
            target_revision=1,
            intent=object(),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            now=_NOW,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-r6-stage-2"
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.PREPARED


class _DifferentAttemptCommittedFinalizeStore(InMemoryDocumentStore):
    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("status") == "prepared"
            and replacement.data.get("status") == "committed"
            and expected.data.get("mutation_id") == "mutation-r6-finalize-1"
        ):
            m2 = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "mutation_id": "mutation-r6-finalize-2",
                        "status": "committed",
                        "outcome": "applied",
                        "committed_revision": expected.data.get("target_revision"),
                        "committed_at": replacement.data.get("committed_at"),
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=m2)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


def test_failed_finalize_does_not_accept_committed_newer_attempt() -> None:
    store = _DifferentAttemptCommittedFinalizeStore()
    engine, repo, handler = _build_engine(store=store)
    prepared = _mutation_record(
        mutation_id="mutation-r6-finalize-1",
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
        result_entity_type=_RESULT_TYPE,
        result_entity_id=_RESULT_ID,
    )
    repo.put_knowledge_configuration_mutation_if_absent(prepared)
    head = WorkspaceKnowledgeConfigurationHead(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        committed_revision=1,
        pending_revision=None,
        pending_mutation_id=None,
        last_committed_mutation_id="mutation-r6-finalize-2",
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
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._finalize_published_mutation(
            mutation=prepared,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            head=head,
            now=_NOW,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-r6-finalize-2"
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_cleanup_fenced_ownership_conflict_refreshes_head_on_first_recovery() -> None:
    store = _TransitionRecordingStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    _seed_cleanup_fenced_pending(
        repo=repo,
        mutation_id="mutation-r6-conflict-head",
    )
    stale_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert stale_head is not None
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError):
        engine.recover_workspace_knowledge_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    loaded_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert loaded_head is not None
    assert loaded_head.updated_at > stale_head.updated_at
    assert store.head_fence_cas_attempts >= 1
    assert _handler.cleanup_calls == 0


# --- R7 pre-target attempt fencing and writer-slot compensation ---


def _mutation_row_key() -> str:
    return f"{_WORKSPACE}:{_OPERATION.value}:{_SHA256}"


def _mutation_partition_key() -> str:
    return f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_mutation"


class _MutationSwapsToM2AfterHeadCASStore(InMemoryDocumentStore):
    """Replaces M1 with M2 immediately after a successful M1 writer-slot CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_head" in expected.partition_key
            and not expected.data.get("pending_mutation_id")
            and replacement.data.get("pending_mutation_id") == "mutation-r7-postcas-1"
        ):
            if not super().replace_if_match(expected=expected, replacement=replacement):
                return False
            mutation_doc = self.get(_mutation_partition_key(), _mutation_row_key())
            if mutation_doc is not None:
                m2 = mutation_doc.model_copy(
                    update={
                        "data": {
                            **mutation_doc.data,
                            "mutation_id": "mutation-r7-postcas-2",
                        }
                    }
                )
                super().replace_if_match(expected=mutation_doc, replacement=m2)
            return True
        return super().replace_if_match(expected=expected, replacement=replacement)


class _DifferentAttemptTargetAssignmentCASStore(InMemoryDocumentStore):
    """Replaces M1 with M2 on failed target-assignment CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("mutation_id") == "mutation-r7-target-1"
            and expected.data.get("status") == "reserved"
            and replacement.data.get("target_revision") == 1
        ):
            m2 = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "mutation_id": "mutation-r7-target-2",
                        "target_revision": 1,
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=m2)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _ConcurrentTargetAssignmentCASStore(InMemoryDocumentStore):
    """Another caller assigns target to the same M1 before local CAS succeeds."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("mutation_id") == "mutation-r7-concurrent-1"
            and expected.data.get("status") == "reserved"
            and replacement.data.get("target_revision") == 1
        ):
            assigned = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "target_revision": 1,
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=assigned)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _TargetAssignmentReloadsPreparedStore(InMemoryDocumentStore):
    """Concurrent caller reaches PREPARED before local target-assignment CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("mutation_id") == "mutation-r7-prepared-1"
            and expected.data.get("status") == "reserved"
            and replacement.data.get("target_revision") == 1
        ):
            prepared = expected.model_copy(
                update={
                    "data": {
                        **expected.data,
                        "status": "prepared",
                        "target_revision": 1,
                        "result_entity_type": _RESULT_TYPE,
                        "result_entity_id": _RESULT_ID,
                    }
                }
            )
            super().replace_if_match(expected=expected, replacement=prepared)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)


class _MutationSwapsToAbortedM2AfterConfirmStore(InMemoryDocumentStore):
    """Replaces aborted M1 row with unrelated M2 ABORTED after confirm CAS."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if (
            "knowledge_configuration_mutation" in expected.partition_key
            and expected.data.get("mutation_id") == "mutation-r7-cleanup-1"
            and expected.data.get("status") == "recovery_required"
            and replacement.data.get("status") == "aborted"
        ):
            if not super().replace_if_match(expected=expected, replacement=replacement):
                return False
            m2_aborted = replacement.model_copy(
                update={
                    "data": {
                        **replacement.data,
                        "mutation_id": "mutation-r7-cleanup-2",
                    }
                }
            )
            aborted_doc = self.get(_mutation_partition_key(), _mutation_row_key())
            if aborted_doc is not None:
                super().replace_if_match(expected=aborted_doc, replacement=m2_aborted)
            return True
        return super().replace_if_match(expected=expected, replacement=replacement)


def test_stale_revision_conflict_abort_cannot_abort_newer_reserved_attempt() -> None:
    engine, repo, _handler = _build_engine()
    stale_m1 = _mutation_record(mutation_id="mutation-r7-abort-1")
    m2 = _mutation_record(mutation_id="mutation-r7-abort-2")
    repo.put_knowledge_configuration_mutation_if_absent(m2)
    before = m2.model_copy()
    engine._abort_mutation_for_revision_conflict(
        mutation=stale_m1,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        now=_NOW,
    )
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded == before
    assert loaded is not None
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RESERVED
    assert loaded.mutation_id == "mutation-r7-abort-2"


def test_stale_writer_slot_acquisition_cannot_reserve_when_newer_attempt_exists() -> None:
    engine, repo, _handler = _build_engine()
    stale_m1 = _mutation_record(mutation_id="mutation-r7-slot-1")
    m2 = _mutation_record(mutation_id="mutation-r7-slot-2")
    repo.put_knowledge_configuration_mutation_if_absent(m2)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._acquire_writer_slot_for_expected_revision(
            mutation=stale_m1,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            expected_revision=0,
            now=_NOW,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert head is not None
    assert head.pending_mutation_id is None
    assert head.pending_revision is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-r7-slot-2"
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RESERVED


def test_post_head_cas_mutation_swap_compensates_stale_slot_and_fails_closed() -> None:
    store = _MutationSwapsToM2AfterHeadCASStore()
    engine, repo, _handler = _build_engine(store=store)
    m1 = _mutation_record(mutation_id="mutation-r7-postcas-1")
    repo.put_knowledge_configuration_mutation_if_absent(m1)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._acquire_writer_slot_for_expected_revision(
            mutation=m1,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            expected_revision=0,
            now=_NOW,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert head is not None
    assert head.committed_revision == 0
    assert head.pending_mutation_id is None
    assert head.pending_revision is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-r7-postcas-2"
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RESERVED


@pytest.mark.parametrize(
    ("committed_revision", "pending_mutation_id"),
    [
        (1, None),
        (0, "mutation-r7-comp-other"),
    ],
)
def test_stale_slot_compensation_cannot_clear_published_or_foreign_writer_slot(
    committed_revision: int,
    pending_mutation_id: str | None,
) -> None:
    engine, repo, _handler = _build_engine()
    m1 = _mutation_record(mutation_id="mutation-r7-comp-1")
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=committed_revision,
            pending_revision=1 if pending_mutation_id is not None else None,
            pending_mutation_id=pending_mutation_id,
            last_committed_mutation_id="other" if committed_revision else None,
            updated_at=_NOW,
        )
    )
    before = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    engine._try_compensate_stale_writer_slot(
        mutation=m1,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        expected_revision=0,
        target_revision=1,
        now=_NOW,
    )
    after = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert after == before


def test_failed_target_assignment_cannot_adopt_newer_attempt_with_same_target() -> None:
    store = _DifferentAttemptTargetAssignmentCASStore()
    handler = _FakeHandler()
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    m1 = _mutation_record(mutation_id="mutation-r7-target-1")
    repo.put_knowledge_configuration_mutation_if_absent(m1)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-r7-target-1",
            updated_at=_NOW,
        )
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine._assign_target_revision(
            mutation=m1,
            target_revision=1,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=_OPERATION,
            idempotency_key_hash=_SHA256,
            expected_revision=0,
            now=_NOW,
        )
    assert exc.value.error_code == "configuration_recovery_required"
    assert handler.stage_calls == 0
    head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert head is not None
    assert head.pending_mutation_id is None
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-r7-target-2"
    assert loaded.target_revision == 1
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RESERVED


def test_same_attempt_concurrent_target_assignment_remains_supported() -> None:
    store = _ConcurrentTargetAssignmentCASStore()
    engine, repo, _handler = _build_engine(store=store)
    m1 = _mutation_record(mutation_id="mutation-r7-concurrent-1")
    repo.put_knowledge_configuration_mutation_if_absent(m1)
    assigned = engine._assign_target_revision(
        mutation=m1,
        target_revision=1,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
        expected_revision=0,
        now=_NOW,
    )
    assert assigned.mutation_id == "mutation-r7-concurrent-1"
    assert assigned.target_revision == 1
    assert assigned.status is WorkspaceKnowledgeMutationStatusV1.RESERVED


def test_prepared_continuation_does_not_stage_twice() -> None:
    store = _TargetAssignmentReloadsPreparedStore()
    handler = _FakeHandler()
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    m1 = _mutation_record(mutation_id="mutation-r7-prepared-1")
    repo.put_knowledge_configuration_mutation_if_absent(m1)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-r7-prepared-1",
            updated_at=_NOW,
        )
    )
    handler.stage(
        repository=repo,
        mutation=m1,
        target_revision=1,
        intent=object(),
        now=_NOW,
    )
    result = engine._continue_mutation_execution(
        mutation=m1,
        handler=handler,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        expected_revision=0,
        intent=object(),
    )
    assert handler.stage_calls == 1
    assert result.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED


def test_cleanup_abort_returns_exact_mutation_not_unrelated_aborted_attempt() -> None:
    store = _MutationSwapsToAbortedM2AfterConfirmStore()
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    engine, repo, _handler = _build_engine(store=store, handler=handler)
    fenced = _mutation_record(
        mutation_id="mutation-r7-cleanup-1",
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        error_code=_CLEANUP_FENCE_ERROR,
        target_revision=1,
    )
    repo.put_knowledge_configuration_mutation_if_absent(fenced)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="mutation-r7-cleanup-1",
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
    head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert head is not None
    recovery = engine._abort_incomplete_pending_mutation(
        mutation=fenced,
        handler=handler,
        head=head,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert recovery.mutation.mutation_id == "mutation-r7-cleanup-1"
    assert recovery.mutation.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-r7-cleanup-2"
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED


def _load_mutation(repo: ManagedWorkspaceRepository) -> WorkspaceKnowledgeMutationRecord:
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=_OPERATION,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    return loaded


def _seed_pending_head(
    repo: ManagedWorkspaceRepository,
    *,
    mutation_id: str,
    target_revision: int = 1,
    stage_claim_id: str | None = None,
) -> WorkspaceKnowledgeMutationRecord:
    mutation = _mutation_record(
        mutation_id=mutation_id,
        target_revision=target_revision,
        stage_claim_id=stage_claim_id,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=target_revision,
            pending_mutation_id=mutation_id,
            updated_at=_NOW,
        )
    )
    return mutation


# --- R2 durable stage claim ---


def test_one_stage_writer_per_mutation() -> None:
    store = InMemoryDocumentStore()
    handler_a = _FakeHandler()
    handler_b = _FakeHandler()
    handler_a.stage_release.clear()
    engine_a, repo, _ = _build_engine(
        store=store, handler=handler_a, mutation_ids=["mutation-a", "mutation-b"], claim_ids=["claim-a", "claim-b"],
    )
    engine_b, _, _ = _build_engine(
        store=store, handler=handler_b, mutation_ids=["mutation-b", "mutation-c"], claim_ids=["claim-b", "claim-c"],
    )
    errors: list[BaseException] = []
    thread_a = threading.Thread(target=lambda: _execute(engine_a))

    def run_b() -> None:
        try:
            _execute(engine_b)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread_b = threading.Thread(target=run_b)
    thread_a.start()
    assert handler_a.stage_entered.wait(timeout=5)
    thread_b.start()
    thread_b.join(timeout=5)
    handler_a.stage_release.set()
    thread_a.join(timeout=5)
    assert len(errors) == 1
    assert isinstance(errors[0], WorkspaceKnowledgeConfigurationMutationError)
    assert errors[0].error_code == "configuration_recovery_required"
    assert handler_a.stage_calls == 1 and handler_b.stage_calls == 0
    loaded = _load_mutation(repo)
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED and loaded.stage_claim_id is None
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == 1


def test_recovery_during_in_flight_claimed_stage() -> None:
    handler = _FakeHandler()
    handler.stage_release.clear()
    engine, repo, _handler = _build_engine(
        handler=handler, mutation_ids=["mutation-stage"], claim_ids=["claim-stage"],
    )
    writer_error: list[BaseException] = []

    def writer() -> None:
        try:
            _execute(engine)
        except BaseException as exc:  # noqa: BLE001
            writer_error.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert handler.stage_entered.wait(timeout=5)
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = _load_mutation(repo)
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.stage_claim_id == "claim-stage"
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.pending_mutation_id == "mutation-stage"
    handler.stage_release.set()
    thread.join(timeout=5)
    assert len(writer_error) == 1
    assert isinstance(writer_error[0], WorkspaceKnowledgeConfigurationMutationError)
    assert writer_error[0].error_code == "configuration_recovery_required"
    loaded = _load_mutation(repo)
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED and loaded.stage_claim_id is None
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.pending_mutation_id is None and head.committed_revision == 0
    assert not repo.list_knowledge_connection_attachment_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    config = WorkspaceKnowledgeConfigurationService(
        repo, _FakeWorkspaceLookup({(_TENANT, _WORKSPACE): _workspace()}),
    ).get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None and config.connection_attachments == ()


def test_claimed_stage_absent_recovery_preserves_fence() -> None:
    engine, repo, _handler = _build_engine(
        mutation_ids=["mutation-absent"], claim_ids=["claim-absent"],
    )
    _seed_pending_head(repo, mutation_id="mutation-absent", stage_claim_id="claim-absent")
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert exc.value.error_code == "configuration_recovery_required"
    loaded = _load_mutation(repo)
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.stage_claim_id == "claim-absent"
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.pending_mutation_id == "mutation-absent"


def test_recovery_after_complete_claimed_staging() -> None:
    engine, repo, handler = _build_engine(
        mutation_ids=["mutation-complete"], claim_ids=["claim-complete"],
    )
    mutation = _seed_pending_head(repo, mutation_id="mutation-complete", stage_claim_id="claim-complete")
    handler.stage(repository=repo, mutation=mutation, target_revision=1, intent=object(), now=_NOW)
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    assert recovery.mutation is not None
    assert recovery.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert recovery.mutation.stage_claim_id is None
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == 1 and handler.cleanup_calls == 0


def test_stage_claim_cas_race() -> None:
    store = InMemoryDocumentStore()
    handler_a = _FakeHandler()
    handler_b = _FakeHandler()
    engine_a, repo, _ = _build_engine(
        store=store, handler=handler_a, mutation_ids=["mutation-race"], claim_ids=["claim-winner", "claim-loser"],
    )
    engine_b, _, _ = _build_engine(
        store=store, handler=handler_b, mutation_ids=["mutation-race", "mutation-other"],
        claim_ids=["claim-loser", "claim-other"],
    )
    _seed_pending_head(repo, mutation_id="mutation-race")
    results: list[Any] = []
    errors: list[BaseException] = []

    def run(engine: WorkspaceKnowledgeConfigurationMutationEngine) -> None:
        try:
            results.append(_execute(engine))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = threading.Thread(target=run, args=(engine_a,))
    thread_b = threading.Thread(target=run, args=(engine_b,))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()
    assert handler_a.stage_calls + handler_b.stage_calls == 1 and len(results) == 1
    loaded = _load_mutation(repo)
    assert loaded.stage_claim_id in (None, "claim-winner")


def test_claim_owner_cleanup_failure_preserves_fence() -> None:
    handler = _FakeHandler()
    handler.inspection_state = WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    handler.cleanup_returns = False
    engine, repo, _handler = _build_engine(
        handler=handler, mutation_ids=["mutation-cleanup-fail"], claim_ids=["claim-cleanup-fail"],
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_mutation_cleanup_failed"
    loaded = _load_mutation(repo)
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert loaded.stage_claim_id == "claim-cleanup-fail"
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.pending_mutation_id is not None


# --- R3 stage-claim CAS loss and committed replay hardening ---

_CONCURRENT_PREPARED_ID = "concurrent-prepared-result"
_FENCE_REPLAY_ID = "fence-replay-result"


class _FenceReplayPhase(IntEnum):
    INITIAL = 0
    FENCE_PERSISTED = 1
    FIRST_RELOAD_RETURNED = 2
    CLEANUP_FENCE_ACQUISITION_STARTED = 3
    CONCURRENT_COMMIT_INJECTED = 4


class _StageClaimHardeningStore(InMemoryDocumentStore):
    def __init__(self, mode: str, *, finalize: bool = False) -> None:
        super().__init__()
        self._mode = mode
        self._finalize = finalize
        self._prepared_failures = 1
        self._fence_phase = _FenceReplayPhase.INITIAL
        self.fence_persisted = False
        self.first_recovery_required_reload_seen = False
        self.cleanup_fence_path_entered = False
        self.concurrent_commit_injected = False
        self.direct_committed_after_prepared_cas = False
        self.store_created_prepared = False
        self.store_prepared_to_committed_on_claim = False

    def _publish_head(
        self, *, tenant_id: str, workspace_id: str, mutation_id: str, target_revision: int,
    ) -> None:
        head_pk = f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_head"
        head_doc = self.get(head_pk, workspace_id)
        if head_doc is None:
            return
        published = head_doc.model_copy(update={"data": {
            **head_doc.data, "committed_revision": target_revision,
            "pending_revision": None, "pending_mutation_id": None,
            "last_committed_mutation_id": mutation_id,
        }})
        super().replace_if_match(expected=head_doc, replacement=published)

    def _inject_fence_replay_commit(self, mutation_doc: DocumentRecord) -> DocumentRecord:
        ed = mutation_doc.data
        target_revision = ed.get("target_revision")
        committed = mutation_doc.model_copy(update={"data": {
            **ed, "status": "committed", "outcome": "applied", "stage_claim_id": None,
            "committed_revision": target_revision, "result_entity_type": _RESULT_TYPE,
            "result_entity_id": _FENCE_REPLAY_ID, "committed_at": _NOW.isoformat(), "error_code": None,
        }})
        super().replace_if_match(expected=mutation_doc, replacement=committed)
        if target_revision is not None:
            self._publish_head(
                tenant_id=ed["tenant_id"], workspace_id=ed["workspace_id"],
                mutation_id=ed["mutation_id"], target_revision=target_revision,
            )
        self.concurrent_commit_injected = True
        self._fence_phase = _FenceReplayPhase.CONCURRENT_COMMIT_INJECTED
        return committed

    def replace_if_match(self, *, expected: DocumentRecord, replacement: DocumentRecord) -> bool:
        pk = expected.partition_key
        if "knowledge_configuration_mutation" not in pk:
            return super().replace_if_match(expected=expected, replacement=replacement)
        ed, rd = expected.data, replacement.data
        claim_cas = (
            ed.get("status") == "reserved" and ed.get("stage_claim_id") is None and rd.get("stage_claim_id")
        )
        prepared_cas = (
            ed.get("status") == "reserved" and rd.get("status") == "prepared" and ed.get("stage_claim_id")
        )
        if self._mode == "claim_prepared" and claim_cas:
            prepared = expected.model_copy(update={"data": {
                **ed, "status": "prepared", "stage_claim_id": None,
                "result_entity_type": _RESULT_TYPE, "result_entity_id": _CONCURRENT_PREPARED_ID,
            }})
            super().replace_if_match(expected=expected, replacement=prepared)
            self.store_created_prepared = True
            if self._finalize:
                target_revision = ed.get("target_revision")
                if target_revision is not None:
                    self._publish_head(
                        tenant_id=ed["tenant_id"], workspace_id=ed["workspace_id"],
                        mutation_id=ed["mutation_id"], target_revision=target_revision,
                    )
            return False
        if self._mode == "foreign_claim" and prepared_cas:
            foreign = expected.model_copy(update={"data": {**ed, "stage_claim_id": "foreign-claim"}})
            super().replace_if_match(expected=expected, replacement=foreign)
            return False
        if self._mode == "fail_first" and prepared_cas and self._prepared_failures > 0:
            self._prepared_failures -= 1
            return False
        if self._mode == "fence_replay" and prepared_cas:
            fenced = expected.model_copy(update={"data": {
                **ed, "status": "recovery_required", "error_code": _CLEANUP_FENCE_ERROR,
            }})
            super().replace_if_match(expected=expected, replacement=fenced)
            self.fence_persisted = True
            self._fence_phase = _FenceReplayPhase.FENCE_PERSISTED
            return False
        if (
            self._mode == "fence_replay"
            and self._fence_phase is _FenceReplayPhase.CLEANUP_FENCE_ACQUISITION_STARTED
            and ed.get("status") == "recovery_required"
            and ed.get("error_code") == _CLEANUP_FENCE_ERROR
            and rd.get("status") == "recovery_required"
        ):
            self._inject_fence_replay_commit(expected)
            return False
        return super().replace_if_match(expected=expected, replacement=replacement)

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        doc = super().get(partition_key, row_key)
        if (
            doc is None or self._mode != "fence_replay"
            or "knowledge_configuration_mutation" not in partition_key
        ):
            return doc
        status = doc.data.get("status")
        if status == "committed" and self._fence_phase < _FenceReplayPhase.CONCURRENT_COMMIT_INJECTED:
            self.direct_committed_after_prepared_cas = True
            return doc
        if self._fence_phase is _FenceReplayPhase.FENCE_PERSISTED:
            self._fence_phase = _FenceReplayPhase.FIRST_RELOAD_RETURNED
            self.first_recovery_required_reload_seen = True
            return doc
        if self._fence_phase is _FenceReplayPhase.FIRST_RELOAD_RETURNED:
            self._fence_phase = _FenceReplayPhase.CLEANUP_FENCE_ACQUISITION_STARTED
            self.cleanup_fence_path_entered = True
            return doc
        return doc


class _ConcurrentCommitStageErrorHandler(_FakeHandler):
    concurrent_result_id = "committed-during-stage"

    def stage(self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
              target_revision: int, intent: object, now: datetime) -> WorkspaceKnowledgeStagedResult:
        self.stage_calls += 1
        head = repository.get_knowledge_configuration_head(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id)
        assert head is not None
        repository.replace_knowledge_configuration_head_if_match(
            expected=head, replacement=head.model_copy(update={
                "committed_revision": target_revision, "pending_revision": None,
                "pending_mutation_id": None, "last_committed_mutation_id": mutation.mutation_id,
                "updated_at": now,
            }))
        current = repository.get_knowledge_configuration_mutation(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
            operation=mutation.operation, idempotency_key_hash=mutation.idempotency_key_hash)
        assert current is not None
        repository.replace_knowledge_configuration_mutation_if_match(
            expected=current, replacement=current.model_copy(update={
                "status": WorkspaceKnowledgeMutationStatusV1.COMMITTED,
                "outcome": WorkspaceKnowledgeMutationOutcomeV1.APPLIED, "stage_claim_id": None,
                "target_revision": target_revision, "committed_revision": target_revision,
                "result_entity_type": _RESULT_TYPE, "result_entity_id": self.concurrent_result_id,
                "committed_at": now,
            }))
        raise RuntimeError("stage failed after concurrent commit")


@pytest.mark.parametrize(
    ("finalize", "expected_disposition"),
    [
        (False, WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED),
        (True, WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY),
    ],
)
def test_claim_cas_loss_to_prepared_continues_without_staging(
    finalize: bool, expected_disposition: WorkspaceKnowledgeMutationExecutionDispositionV1,
) -> None:
    handler = _FakeHandler()
    if finalize:
        handler.inspection_id = _CONCURRENT_PREPARED_ID
        handler.staged_rows.append(
            WorkspaceConnectionAttachment(
                attachment_id=_CONCURRENT_PREPARED_ID,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                connection_ref="conn.concurrent",
                safe_display_label="Concurrent",
                status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
                mutation_id="mutation-1",
                effective_revision=1,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    store = _StageClaimHardeningStore("claim_prepared", finalize=finalize)
    engine, repo, handler = _build_engine(
        store=store, handler=handler, claim_ids=["claim-loser"],
    )
    result = _execute(engine)
    loaded = _load_mutation(repo)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert result.disposition is expected_disposition and handler.stage_calls == 0
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert loaded.stage_claim_id is None and loaded.result_entity_id == _CONCURRENT_PREPARED_ID
    assert result.result_entity_id == _CONCURRENT_PREPARED_ID
    assert head.committed_revision == 1 and head.pending_mutation_id is None
    assert store.store_created_prepared
    if finalize:
        assert not store.store_prepared_to_committed_on_claim


def test_stage_error_loses_to_committed_result() -> None:
    handler = _ConcurrentCommitStageErrorHandler()
    engine, repo, _handler = _build_engine(handler=handler, claim_ids=["claim-stage-error"])
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert result.result_entity_id == handler.concurrent_result_id
    assert handler.stage_calls == 1 and handler.cleanup_calls == 0
    assert _load_mutation(repo).result_entity_id == handler.concurrent_result_id


def test_foreign_claim_cannot_be_cleared() -> None:
    handler = _FakeHandler()
    engine, repo, _handler = _build_engine(
        store=_StageClaimHardeningStore("foreign_claim"), handler=handler, claim_ids=["claim-local"])
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        _execute(engine)
    assert exc.value.error_code == "configuration_recovery_required"
    assert handler.stage_calls == 1 and handler.cleanup_calls == 0
    loaded = _load_mutation(repo)
    assert loaded.status in (WorkspaceKnowledgeMutationStatusV1.RESERVED,
                             WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED)
    assert loaded.stage_claim_id == "foreign-claim" and loaded.result_entity_type is None
    assert repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE).pending_mutation_id


def test_local_claim_fallback_prepared_cas_still_works() -> None:
    engine, repo, handler = _build_engine(
        store=_StageClaimHardeningStore("fail_first"), claim_ids=["claim-fallback"])
    result = _execute(engine)
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert handler.stage_calls == 1
    loaded = _load_mutation(repo)
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED and loaded.stage_claim_id is None
    assert loaded.result_entity_type == _RESULT_TYPE and loaded.result_entity_id == _RESULT_ID


def test_same_claim_fence_path_returns_committed_replay() -> None:
    store = _StageClaimHardeningStore("fence_replay")
    engine, repo, handler = _build_engine(store=store, claim_ids=["claim-fence-replay"])
    result = _execute(engine)
    loaded = _load_mutation(repo)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert result.result_entity_id == _FENCE_REPLAY_ID
    assert handler.stage_calls == 1 and handler.cleanup_calls == 0
    assert loaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    assert loaded.stage_claim_id is None and loaded.result_entity_id == _FENCE_REPLAY_ID
    assert head.committed_revision == 1 and head.pending_mutation_id is None
    assert store.fence_persisted and store.first_recovery_required_reload_seen
    assert store.cleanup_fence_path_entered and store.concurrent_commit_injected
    assert not store.direct_committed_after_prepared_cas
