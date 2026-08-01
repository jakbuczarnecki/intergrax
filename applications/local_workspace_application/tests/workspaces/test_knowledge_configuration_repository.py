# © Artur Czarnecki. All rights reserved.

"""Unit tests for Workspace Knowledge Configuration persistence primitives."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Optional

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import (
    ManagedWorkspaceRepository,
    WorkspaceKnowledgeConfigurationRepositoryError,
    _revision_row_key,
)
from intergrax.integrations.contracts.base import IntegrationCategory

pytestmark = pytest.mark.unit

_NOW = datetime.now(UTC)
_SHA256 = "a" * 64
_SHA256_B = "b" * 64
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-1"
_WORKSPACE_B = "workspace-2"
_MUTATION = "mutation-1"


class _PlainDocumentStore:
    """Minimal non-conditional DocumentStore stub."""

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


def _repo() -> ManagedWorkspaceRepository:
    return ManagedWorkspaceRepository(InMemoryDocumentStore())


def _plain_repo() -> ManagedWorkspaceRepository:
    return ManagedWorkspaceRepository(_PlainDocumentStore())


def _head(**overrides: object) -> WorkspaceKnowledgeConfigurationHead:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceKnowledgeConfigurationHead(**payload)


def _mutation(**overrides: object) -> WorkspaceKnowledgeMutationRecord:
    payload = {
        "mutation_id": _MUTATION,
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation": WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
        "idempotency_key_hash": _SHA256,
        "normalized_request_hash": _SHA256,
        "status": WorkspaceKnowledgeMutationStatusV1.RESERVED,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceKnowledgeMutationRecord(**payload)


def _connection_attachment(**overrides: object) -> WorkspaceConnectionAttachment:
    payload = {
        "attachment_id": "attachment-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.primary",
        "safe_display_label": "Primary",
        "status": WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceConnectionAttachment(**payload)


def _indexed_source(**overrides: object) -> WorkspaceIndexedSourceBinding:
    payload = {
        "indexed_source_binding_id": "idx-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "knowledge_source_binding_ref": "ksb-1",
        "source_id": "source-1",
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceIndexedSourceBinding(**payload)


def _live_access(**overrides: object) -> WorkspaceLiveAccessBinding:
    payload = {
        "live_access_binding_id": "live-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.live",
        "allowed_capability_ids": ("cap.read",),
        "derived_provider_id": "provider-1",
        "derived_integration_kind": IntegrationCategory.WIKI_KNOWLEDGE,
        "derived_safe_display_label": "Wiki",
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceLiveAccessBinding(**payload)


def _query_policy(**overrides: object) -> WorkspaceQueryPolicy:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceQueryPolicy(**payload)


def _stale_child(family: str, fixture_fn: object) -> object:
    if family == "connection_attachment":
        return fixture_fn(safe_display_label="Stale")  # type: ignore[operator]
    if family == "indexed_source":
        return fixture_fn(cached_safe_display_label="Stale")  # type: ignore[operator]
    if family == "live_access":
        return fixture_fn(derived_safe_display_label="Stale")  # type: ignore[operator]
    if family == "query_policy":
        return fixture_fn(max_result_items=100)  # type: ignore[operator]
    raise ValueError(f"unknown family: {family}")


CHILD_FIXTURES = [
    (
        "connection_attachment",
        _connection_attachment,
        "put_knowledge_connection_attachment_version_if_absent",
        "get_knowledge_connection_attachment_version",
        "list_knowledge_connection_attachment_versions",
        "delete_knowledge_connection_attachment_version_if_match",
        {"attachment_id": "attachment-1"},
        "attachment_id",
    ),
    (
        "indexed_source",
        _indexed_source,
        "put_knowledge_indexed_source_version_if_absent",
        "get_knowledge_indexed_source_version",
        "list_knowledge_indexed_source_versions",
        "delete_knowledge_indexed_source_version_if_match",
        {"indexed_source_binding_id": "idx-1"},
        "indexed_source_binding_id",
    ),
    (
        "live_access",
        _live_access,
        "put_knowledge_live_access_version_if_absent",
        "get_knowledge_live_access_version",
        "list_knowledge_live_access_versions",
        "delete_knowledge_live_access_version_if_match",
        {"live_access_binding_id": "live-1"},
        "live_access_binding_id",
    ),
    (
        "query_policy",
        _query_policy,
        "put_knowledge_query_policy_version_if_absent",
        "get_knowledge_query_policy_version",
        "list_knowledge_query_policy_versions",
        "delete_knowledge_query_policy_version_if_match",
        {},
        None,
    ),
]


# --- Conditional capability ---


def test_read_operations_work_with_plain_document_store() -> None:
    repo = _plain_repo()
    assert repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE) is None
    assert (
        repo.get_knowledge_configuration_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            idempotency_key_hash=_SHA256,
        )
        is None
    )
    assert (
        repo.get_knowledge_connection_attachment_version(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            attachment_id="attachment-1",
            effective_revision=1,
        )
        is None
    )


@pytest.mark.parametrize(
    "method_name,args,kwargs",
    [
        ("put_knowledge_configuration_head_if_absent", (_head(),), {}),
        (
            "replace_knowledge_configuration_head_if_match",
            (),
            {"expected": _head(), "replacement": _head(committed_revision=1)},
        ),
        ("put_knowledge_configuration_mutation_if_absent", (_mutation(),), {}),
        (
            "replace_knowledge_configuration_mutation_if_match",
            (),
            {"expected": _mutation(), "replacement": _mutation(mutation_id="mutation-2")},
        ),
        ("put_knowledge_connection_attachment_version_if_absent", (_connection_attachment(),), {}),
        ("put_knowledge_indexed_source_version_if_absent", (_indexed_source(),), {}),
        ("put_knowledge_live_access_version_if_absent", (_live_access(),), {}),
        ("put_knowledge_query_policy_version_if_absent", (_query_policy(),), {}),
        (
            "delete_knowledge_connection_attachment_version_if_match",
            (_connection_attachment(),),
            {},
        ),
        ("delete_knowledge_indexed_source_version_if_match", (_indexed_source(),), {}),
        ("delete_knowledge_live_access_version_if_match", (_live_access(),), {}),
        ("delete_knowledge_query_policy_version_if_match", (_query_policy(),), {}),
    ],
)
def test_conditional_writes_require_conditional_store(
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    repo = _plain_repo()
    method = getattr(repo, method_name)
    with pytest.raises(WorkspaceKnowledgeConfigurationRepositoryError) as exc_info:
        method(*args, **kwargs)
    assert exc_info.value.error_code == "configuration_conditional_store_required"


def test_existing_workspace_methods_work_with_plain_store() -> None:
    repo = _plain_repo()
    workspace = Workspace(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        name="Test",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_workspace(workspace)
    loaded = repo.get_workspace(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert loaded is not None
    assert loaded.workspace_id == _WORKSPACE


# --- Revision head ---


def test_missing_head_returns_none() -> None:
    repo = _repo()
    assert repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE) is None


def test_first_head_put_if_absent_succeeds() -> None:
    repo = _repo()
    head = _head()
    assert repo.put_knowledge_configuration_head_if_absent(head) is True


def test_duplicate_head_put_if_absent_returns_false() -> None:
    repo = _repo()
    head = _head()
    assert repo.put_knowledge_configuration_head_if_absent(head) is True
    assert repo.put_knowledge_configuration_head_if_absent(head) is False


def test_duplicate_head_put_does_not_overwrite() -> None:
    repo = _repo()
    first = _head(committed_revision=0)
    second = _head(committed_revision=5)
    assert repo.put_knowledge_configuration_head_if_absent(first) is True
    assert repo.put_knowledge_configuration_head_if_absent(second) is False
    loaded = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert loaded is not None
    assert loaded.committed_revision == 0


def test_head_cas_replacement_succeeds() -> None:
    repo = _repo()
    initial = _head()
    assert repo.put_knowledge_configuration_head_if_absent(initial) is True
    updated = _head(committed_revision=1)
    assert (
        repo.replace_knowledge_configuration_head_if_match(
            expected=initial,
            replacement=updated,
        )
        is True
    )
    loaded = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert loaded is not None
    assert loaded.committed_revision == 1


def test_stale_head_cas_returns_false() -> None:
    repo = _repo()
    initial = _head()
    assert repo.put_knowledge_configuration_head_if_absent(initial) is True
    stale = _head(committed_revision=99)
    current = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert current is not None
    assert (
        repo.replace_knowledge_configuration_head_if_match(
            expected=stale,
            replacement=_head(committed_revision=1),
        )
        is False
    )
    loaded = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert loaded is not None
    assert loaded.committed_revision == 0


def test_head_cas_tenant_mismatch_raises() -> None:
    repo = _repo()
    expected = _head()
    replacement = _head(tenant_id=_TENANT_B)
    with pytest.raises(ValueError, match="knowledge_configuration_conditional_key_mismatch"):
        repo.replace_knowledge_configuration_head_if_match(
            expected=expected,
            replacement=replacement,
        )


def test_head_cas_workspace_mismatch_raises() -> None:
    repo = _repo()
    expected = _head()
    replacement = _head(workspace_id=_WORKSPACE_B)
    with pytest.raises(ValueError, match="knowledge_configuration_conditional_key_mismatch"):
        repo.replace_knowledge_configuration_head_if_match(
            expected=expected,
            replacement=replacement,
        )


def test_head_row_key_is_workspace_id() -> None:
    repo = _repo()
    head = _head()
    repo.put_knowledge_configuration_head_if_absent(head)
    store = repo.document_store
    record = store.get(
        f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_head",
        _WORKSPACE,
    )
    assert record is not None
    assert record.row_key == _WORKSPACE


# --- Mutation record ---


def test_mutation_row_key_uses_workspace_operation_and_hash() -> None:
    repo = _repo()
    mutation = _mutation()
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    store = repo.document_store
    expected_key = (
        f"{_WORKSPACE}:{WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE.value}:{_SHA256}"
    )
    record = store.get(
        f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_mutation",
        expected_key,
    )
    assert record is not None
    assert record.row_key == expected_key


def test_mutation_hash_is_not_rehashed() -> None:
    repo = _repo()
    mutation = _mutation(idempotency_key_hash=_SHA256)
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    store = repo.document_store
    for (_pk, rk), _doc in store._rows.items():  # noqa: SLF001 - inspect test store layout
        if "knowledge_configuration_mutation" in _pk:
            assert _SHA256 in rk
            assert rk.endswith(f":{_SHA256}")
            assert "raw-idempotency-key" not in rk


def test_raw_idempotency_key_not_in_partition() -> None:
    repo = _repo()
    raw_key = "raw-idempotency-key-should-not-appear"
    mutation = _mutation(idempotency_key_hash=_SHA256)
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    store = repo.document_store
    for (_pk, rk), doc in store._rows.items():  # noqa: SLF001
        if "knowledge_configuration_mutation" in _pk:
            assert raw_key not in rk
            assert raw_key not in str(doc.data)


def test_missing_mutation_returns_none() -> None:
    repo = _repo()
    assert (
        repo.get_knowledge_configuration_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            idempotency_key_hash=_SHA256,
        )
        is None
    )


def test_first_mutation_reservation_succeeds() -> None:
    repo = _repo()
    assert repo.put_knowledge_configuration_mutation_if_absent(_mutation()) is True


def test_duplicate_mutation_reservation_returns_false() -> None:
    repo = _repo()
    first = _mutation()
    second = _mutation(mutation_id="mutation-2")
    assert repo.put_knowledge_configuration_mutation_if_absent(first) is True
    assert repo.put_knowledge_configuration_mutation_if_absent(second) is False


def test_duplicate_mutation_reservation_does_not_overwrite() -> None:
    repo = _repo()
    first = _mutation(mutation_id="mutation-winner")
    second = _mutation(mutation_id="mutation-loser")
    assert repo.put_knowledge_configuration_mutation_if_absent(first) is True
    assert repo.put_knowledge_configuration_mutation_if_absent(second) is False
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-winner"


def test_mutation_cas_replacement_succeeds() -> None:
    repo = _repo()
    reserved = _mutation()
    assert repo.put_knowledge_configuration_mutation_if_absent(reserved) is True
    prepared = _mutation(
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
    )
    assert (
        repo.replace_knowledge_configuration_mutation_if_match(
            expected=reserved,
            replacement=prepared,
        )
        is True
    )


def test_stale_mutation_cas_returns_false() -> None:
    repo = _repo()
    reserved = _mutation()
    assert repo.put_knowledge_configuration_mutation_if_absent(reserved) is True
    stale = _mutation(status=WorkspaceKnowledgeMutationStatusV1.PREPARED, target_revision=1)
    current = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
        idempotency_key_hash=_SHA256,
    )
    assert current is not None
    assert (
        repo.replace_knowledge_configuration_mutation_if_match(
            expected=stale,
            replacement=_mutation(mutation_id="mutation-2"),
        )
        is False
    )


def test_mutation_cas_operation_mismatch_raises() -> None:
    repo = _repo()
    expected = _mutation()
    replacement = _mutation(
        operation=WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION
    )
    with pytest.raises(ValueError, match="knowledge_configuration_conditional_key_mismatch"):
        repo.replace_knowledge_configuration_mutation_if_match(
            expected=expected,
            replacement=replacement,
        )


def test_mutation_cas_idempotency_hash_mismatch_raises() -> None:
    repo = _repo()
    expected = _mutation()
    replacement = _mutation(idempotency_key_hash=_SHA256_B)
    with pytest.raises(ValueError, match="knowledge_configuration_conditional_key_mismatch"):
        repo.replace_knowledge_configuration_mutation_if_match(
            expected=expected,
            replacement=replacement,
        )


def test_mutation_cas_allows_mutation_id_change() -> None:
    repo = _repo()
    aborted = _mutation(
        status=WorkspaceKnowledgeMutationStatusV1.ABORTED,
        mutation_id="mutation-aborted",
    )
    assert repo.put_knowledge_configuration_mutation_if_absent(aborted) is True
    replacement = _mutation(
        status=WorkspaceKnowledgeMutationStatusV1.RESERVED,
        mutation_id="mutation-retry",
    )
    assert (
        repo.replace_knowledge_configuration_mutation_if_match(
            expected=aborted,
            replacement=replacement,
        )
        is True
    )
    loaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
        idempotency_key_hash=_SHA256,
    )
    assert loaded is not None
    assert loaded.mutation_id == "mutation-retry"


def test_invalid_idempotency_hash_rejected() -> None:
    repo = _repo()
    with pytest.raises(
        WorkspaceKnowledgeConfigurationRepositoryError,
        match="knowledge_configuration_idempotency_hash_invalid",
    ):
        repo.get_knowledge_configuration_mutation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            idempotency_key_hash="not-a-valid-hash",
        )


# --- Child families (parameterized) ---


@pytest.mark.parametrize(
    "family,fixture_fn,put_method,get_method,list_method,delete_method,get_kwargs,entity_field",
    CHILD_FIXTURES,
)
def test_child_first_insert_succeeds(
    family: str,
    fixture_fn: object,
    put_method: str,
    get_method: str,
    list_method: str,
    delete_method: str,
    get_kwargs: dict[str, str],
    entity_field: str | None,
) -> None:
    repo = _repo()
    model = fixture_fn()  # type: ignore[operator]
    assert getattr(repo, put_method)(model) is True


@pytest.mark.parametrize(
    "family,fixture_fn,put_method,get_method,list_method,delete_method,get_kwargs,entity_field",
    CHILD_FIXTURES,
)
def test_child_duplicate_insert_returns_false(
    family: str,
    fixture_fn: object,
    put_method: str,
    get_method: str,
    list_method: str,
    delete_method: str,
    get_kwargs: dict[str, str],
    entity_field: str | None,
) -> None:
    repo = _repo()
    model = fixture_fn()  # type: ignore[operator]
    duplicate = fixture_fn(safe_display_label="Changed") if family == "connection_attachment" else fixture_fn()  # type: ignore[operator]
    assert getattr(repo, put_method)(model) is True
    assert getattr(repo, put_method)(duplicate) is False


@pytest.mark.parametrize(
    "family,fixture_fn,put_method,get_method,list_method,delete_method,get_kwargs,entity_field",
    CHILD_FIXTURES,
)
def test_child_duplicate_insert_does_not_overwrite(
    family: str,
    fixture_fn: object,
    put_method: str,
    get_method: str,
    list_method: str,
    delete_method: str,
    get_kwargs: dict[str, str],
    entity_field: str | None,
) -> None:
    repo = _repo()
    first = fixture_fn(safe_display_label="Original") if family == "connection_attachment" else fixture_fn()  # type: ignore[operator]
    second = fixture_fn(safe_display_label="Changed") if family == "connection_attachment" else fixture_fn()  # type: ignore[operator]
    assert getattr(repo, put_method)(first) is True
    assert getattr(repo, put_method)(second) is False
    loaded = getattr(repo, get_method)(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        effective_revision=1,
        **get_kwargs,
    )
    if family == "connection_attachment":
        assert loaded.safe_display_label == "Original"
    else:
        assert loaded is not None


@pytest.mark.parametrize(
    "family,fixture_fn,put_method,get_method,list_method,delete_method,get_kwargs,entity_field",
    CHILD_FIXTURES,
)
def test_child_revision_read_and_missing(
    family: str,
    fixture_fn: object,
    put_method: str,
    get_method: str,
    list_method: str,
    delete_method: str,
    get_kwargs: dict[str, str],
    entity_field: str | None,
) -> None:
    repo = _repo()
    model = fixture_fn()  # type: ignore[operator]
    assert getattr(repo, put_method)(model) is True
    loaded = getattr(repo, get_method)(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        effective_revision=1,
        **get_kwargs,
    )
    assert loaded is not None
    assert getattr(repo, get_method)(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        effective_revision=99,
        **get_kwargs,
    ) is None


@pytest.mark.parametrize(
    "family,fixture_fn,put_method,get_method,list_method,delete_method,get_kwargs,entity_field",
    CHILD_FIXTURES,
)
def test_child_multiple_revisions_coexist_and_list(
    family: str,
    fixture_fn: object,
    put_method: str,
    get_method: str,
    list_method: str,
    delete_method: str,
    get_kwargs: dict[str, str],
    entity_field: str | None,
) -> None:
    repo = _repo()
    rev1 = fixture_fn(effective_revision=1)  # type: ignore[operator]
    rev2 = fixture_fn(effective_revision=2)  # type: ignore[operator]
    assert getattr(repo, put_method)(rev1) is True
    assert getattr(repo, put_method)(rev2) is True
    listed = getattr(repo, list_method)(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(listed) == 2
    assert [item.effective_revision for item in listed] == [1, 2]


@pytest.mark.parametrize(
    "family,fixture_fn,put_method,get_method,list_method,delete_method,get_kwargs,entity_field",
    CHILD_FIXTURES,
)
def test_child_delete_if_match_and_stale(
    family: str,
    fixture_fn: object,
    put_method: str,
    get_method: str,
    list_method: str,
    delete_method: str,
    get_kwargs: dict[str, str],
    entity_field: str | None,
) -> None:
    repo = _repo()
    model = fixture_fn()  # type: ignore[operator]
    assert getattr(repo, put_method)(model) is True
    stale = _stale_child(family, fixture_fn)
    assert getattr(repo, delete_method)(stale) is False
    assert getattr(repo, delete_method)(model) is True
    assert (
        getattr(repo, get_method)(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            effective_revision=1,
            **get_kwargs,
        )
        is None
    )


@pytest.mark.parametrize(
    "family,fixture_fn,put_method,get_method,list_method,delete_method,get_kwargs,entity_field",
    CHILD_FIXTURES,
)
def test_child_row_key_uses_20_digit_revision_padding(
    family: str,
    fixture_fn: object,
    put_method: str,
    get_method: str,
    list_method: str,
    delete_method: str,
    get_kwargs: dict[str, str],
    entity_field: str | None,
) -> None:
    repo = _repo()
    model = fixture_fn(effective_revision=7)  # type: ignore[operator]
    assert getattr(repo, put_method)(model) is True
    store = repo.document_store
    padded = "00000000000000000007"
    if family == "query_policy":
        expected_suffix = f"{_WORKSPACE}:query-policy:rev:{padded}"
    else:
        entity_id = get_kwargs[entity_field]  # type: ignore[index]
        expected_suffix = f"{_WORKSPACE}:{entity_id}:rev:{padded}"
    found = False
    for (_pk, rk), _doc in store._rows.items():  # noqa: SLF001
        if rk == expected_suffix:
            found = True
            break
    assert found


# --- Revision bounds ---


_MAX_VALID_REVISION = 10**20 - 1


def test_max_valid_revision_accepted_with_exact_20_digit_suffix() -> None:
    repo = _repo()
    model = _indexed_source(effective_revision=_MAX_VALID_REVISION)
    assert repo.put_knowledge_indexed_source_version_if_absent(model) is True
    suffix = f"{_MAX_VALID_REVISION:020d}"
    assert len(suffix) == 20
    row_key = _revision_row_key(
        workspace_id=_WORKSPACE,
        entity_id="idx-1",
        revision=_MAX_VALID_REVISION,
    )
    assert row_key.endswith(f":rev:{suffix}")
    loaded = repo.get_knowledge_indexed_source_version(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        indexed_source_binding_id="idx-1",
        effective_revision=_MAX_VALID_REVISION,
    )
    assert loaded is not None
    assert loaded.effective_revision == _MAX_VALID_REVISION


@pytest.mark.parametrize("invalid_revision", [10**20, 10**20 + 1])
def test_oversized_revision_rejected_on_write(invalid_revision: int) -> None:
    repo = _repo()
    with pytest.raises(ValueError, match="knowledge_configuration_revision_invalid"):
        repo.put_knowledge_indexed_source_version_if_absent(
            _indexed_source(effective_revision=invalid_revision)
        )


@pytest.mark.parametrize("invalid_revision", [0, -1])
def test_below_minimum_revision_rejected(invalid_revision: int) -> None:
    repo = _repo()
    with pytest.raises(ValueError, match="knowledge_configuration_revision_invalid"):
        _revision_row_key(
            workspace_id=_WORKSPACE,
            entity_id="idx-1",
            revision=invalid_revision,
        )
    with pytest.raises(ValueError, match="knowledge_configuration_revision_invalid"):
        repo.get_knowledge_indexed_source_version(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            indexed_source_binding_id="idx-1",
            effective_revision=invalid_revision,
        )


def test_oversized_revision_rejected_on_exact_read() -> None:
    repo = _repo()
    with pytest.raises(ValueError, match="knowledge_configuration_revision_invalid"):
        repo.get_knowledge_indexed_source_version(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            indexed_source_binding_id="idx-1",
            effective_revision=10**20,
        )


def _insert_child_revision_rows(
    store: InMemoryDocumentStore,
    *,
    family: str,
    tenant_id: str,
    workspace_id: str,
    count: int,
) -> None:
    partition_by_family = {
        "connection_attachment": (
            f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_connection_attachment"
        ),
        "indexed_source": (
            f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_indexed_source"
        ),
        "live_access": (
            f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_live_access"
        ),
        "query_policy": (
            f"lkw.managed_workspace:{tenant_id}:knowledge_configuration_query_policy"
        ),
    }
    partition = partition_by_family[family]
    for index in range(count):
        if family == "connection_attachment":
            entity_id = f"attachment-bulk-{index:04d}"
            model = _connection_attachment(
                workspace_id=workspace_id,
                attachment_id=entity_id,
                effective_revision=1,
            )
            revision = 1
        elif family == "indexed_source":
            entity_id = f"idx-bulk-{index:04d}"
            model = _indexed_source(
                workspace_id=workspace_id,
                indexed_source_binding_id=entity_id,
                effective_revision=1,
            )
            revision = 1
        elif family == "live_access":
            entity_id = f"live-bulk-{index:04d}"
            model = _live_access(
                workspace_id=workspace_id,
                live_access_binding_id=entity_id,
                effective_revision=1,
            )
            revision = 1
        elif family == "query_policy":
            entity_id = "query-policy"
            revision = index + 1
            model = _query_policy(workspace_id=workspace_id, effective_revision=revision)
        else:
            raise ValueError(f"unknown family: {family}")
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=entity_id,
            revision=revision,
        )
        store.put(
            DocumentRecord(
                partition_key=partition,
                row_key=row_key,
                data=model.model_dump(mode="json"),
            )
        )


_LIST_METHOD_BY_FAMILY = {
    "connection_attachment": "list_knowledge_connection_attachment_versions",
    "indexed_source": "list_knowledge_indexed_source_versions",
    "live_access": "list_knowledge_live_access_versions",
    "query_policy": "list_knowledge_query_policy_versions",
}


@pytest.mark.parametrize("family", list(_LIST_METHOD_BY_FAMILY))
def test_revision_scan_returns_exactly_2000_records(family: str) -> None:
    repo = _repo()
    _insert_child_revision_rows(
        repo.document_store,
        family=family,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        count=2000,
    )
    listed = getattr(repo, _LIST_METHOD_BY_FAMILY[family])(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert len(listed) == 2000
    revisions = [item.effective_revision for item in listed]
    assert revisions == sorted(revisions)


def test_revision_scan_limit_exceeded_at_2001() -> None:
    repo = _repo()
    _insert_child_revision_rows(
        repo.document_store,
        family="indexed_source",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        count=2001,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationRepositoryError) as exc_info:
        repo.list_knowledge_indexed_source_versions(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc_info.value.error_code == "knowledge_configuration_revision_scan_limit_exceeded"


@pytest.mark.parametrize("family", list(_LIST_METHOD_BY_FAMILY))
def test_all_list_methods_route_through_guarded_revision_scan_helper(family: str) -> None:
    repo = _repo()
    _insert_child_revision_rows(
        repo.document_store,
        family=family,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        count=2001,
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationRepositoryError) as exc_info:
        getattr(repo, _LIST_METHOD_BY_FAMILY[family])(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert exc_info.value.error_code == "knowledge_configuration_revision_scan_limit_exceeded"


def test_unrelated_rows_do_not_count_toward_revision_scan_limit() -> None:
    repo = _repo()
    _insert_child_revision_rows(
        repo.document_store,
        family="indexed_source",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE_B,
        count=2001,
    )
    _insert_child_revision_rows(
        repo.document_store,
        family="indexed_source",
        tenant_id=_TENANT_B,
        workspace_id=_WORKSPACE,
        count=2001,
    )
    _insert_child_revision_rows(
        repo.document_store,
        family="connection_attachment",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        count=2001,
    )
    target = _indexed_source()
    assert repo.put_knowledge_indexed_source_version_if_absent(target) is True
    listed = repo.list_knowledge_indexed_source_versions(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert len(listed) == 1
    assert listed[0].indexed_source_binding_id == "idx-1"


# --- Isolation ---


def test_tenant_partitions_are_isolated() -> None:
    repo = _repo()
    head_a = _head(tenant_id=_TENANT)
    head_b = _head(tenant_id=_TENANT_B, workspace_id=_WORKSPACE)
    assert repo.put_knowledge_configuration_head_if_absent(head_a) is True
    assert repo.put_knowledge_configuration_head_if_absent(head_b) is True
    assert (
        repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        is not None
    )
    assert (
        repo.get_knowledge_configuration_head(tenant_id=_TENANT_B, workspace_id=_WORKSPACE)
        is not None
    )


def test_workspace_row_key_prefix_isolation() -> None:
    repo = _repo()
    ws1 = _indexed_source(workspace_id=_WORKSPACE, indexed_source_binding_id="idx-1")
    ws2 = _indexed_source(workspace_id=_WORKSPACE_B, indexed_source_binding_id="idx-1")
    assert repo.put_knowledge_indexed_source_version_if_absent(ws1) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(ws2) is True
    listed = repo.list_knowledge_indexed_source_versions(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert len(listed) == 1
    assert listed[0].workspace_id == _WORKSPACE


def test_same_entity_id_different_families_do_not_collide() -> None:
    repo = _repo()
    shared_id = "shared-entity"
    attachment = _connection_attachment(attachment_id=shared_id)
    indexed = _indexed_source(indexed_source_binding_id=shared_id)
    assert repo.put_knowledge_connection_attachment_version_if_absent(attachment) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(indexed) is True
    assert len(repo.list_knowledge_connection_attachment_versions(
        tenant_id=_TENANT, workspace_id=_WORKSPACE
    )) == 1
    assert len(repo.list_knowledge_indexed_source_versions(
        tenant_id=_TENANT, workspace_id=_WORKSPACE
    )) == 1


# --- Integrity ---


def test_stored_child_workspace_mismatch_fails_closed() -> None:
    repo = _repo()
    store = repo.document_store
    partition = f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_indexed_source"
    row_key = f"{_WORKSPACE}:idx-1:rev:00000000000000000001"
    bad = _indexed_source(workspace_id="other-workspace").model_dump(mode="json")
    store.put(DocumentRecord(partition_key=partition, row_key=row_key, data=bad))
    with pytest.raises(ValueError, match="knowledge_configuration_record_identity_mismatch"):
        repo.get_knowledge_indexed_source_version(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            indexed_source_binding_id="idx-1",
            effective_revision=1,
        )


def test_stored_child_entity_id_mismatch_fails_closed() -> None:
    repo = _repo()
    store = repo.document_store
    partition = f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_indexed_source"
    row_key = f"{_WORKSPACE}:idx-wrong:rev:00000000000000000001"
    data = _indexed_source(indexed_source_binding_id="idx-1").model_dump(mode="json")
    store.put(DocumentRecord(partition_key=partition, row_key=row_key, data=data))
    with pytest.raises(ValueError, match="knowledge_configuration_record_identity_mismatch"):
        repo.get_knowledge_indexed_source_version(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            indexed_source_binding_id="idx-wrong",
            effective_revision=1,
        )


def test_stored_child_revision_mismatch_fails_closed() -> None:
    repo = _repo()
    store = repo.document_store
    partition = f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_indexed_source"
    row_key = f"{_WORKSPACE}:idx-1:rev:00000000000000000002"
    data = _indexed_source(effective_revision=1).model_dump(mode="json")
    store.put(DocumentRecord(partition_key=partition, row_key=row_key, data=data))
    with pytest.raises(ValueError, match="knowledge_configuration_record_identity_mismatch"):
        repo.get_knowledge_indexed_source_version(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            indexed_source_binding_id="idx-1",
            effective_revision=2,
        )


def test_malformed_child_row_not_silently_skipped_in_list() -> None:
    repo = _repo()
    store = repo.document_store
    partition = f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_indexed_source"
    row_key = f"{_WORKSPACE}:idx-1:rev:00000000000000000001"
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=row_key,
            data={"not": "a-valid-model"},
        )
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic validation failure
        repo.list_knowledge_indexed_source_versions(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


# --- Immutability API ---


def test_no_unconditional_child_write_methods() -> None:
    forbidden_suffixes = ("_version", "_version_replace")
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    for name, _method in inspect.getmembers(repo, predicate=inspect.ismethod):
        if not name.startswith("put_knowledge_") and not name.startswith("replace_knowledge_"):
            continue
        if name.endswith("_if_absent") or name.endswith("_if_match"):
            continue
        if "knowledge_configuration_head" in name or "knowledge_configuration_mutation" in name:
            continue
        if any(name.endswith(suffix) for suffix in forbidden_suffixes):
            pytest.fail(f"unexpected unconditional child write method: {name}")
