# © Artur Czarnecki. All rights reserved.

"""Unit tests for committed Workspace Knowledge Configuration projection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_store import DocumentRecord
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
    WorkspaceKnowledgeConfigurationServiceError,
    is_workspace_source_product_visible,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_NOW = datetime.now(UTC)
_T0 = _NOW - timedelta(days=3)
_T1 = _NOW - timedelta(days=2)
_T2 = _NOW - timedelta(days=1)
_SHA256 = "a" * 64
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-1"
_MUTATION = "mutation-1"


def _repo() -> ManagedWorkspaceRepository:
    return ManagedWorkspaceRepository(InMemoryDocumentStore())


def _workspace(**overrides: object) -> Workspace:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "name": "Test",
        "status": WorkspaceStatus.ACTIVE,
        "created_at": _T0,
        "updated_at": _T1,
    }
    payload.update(overrides)
    return Workspace(**payload)


def _head(**overrides: object) -> WorkspaceKnowledgeConfigurationHead:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceKnowledgeConfigurationHead(**payload)


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
        "created_at": _T0,
        "updated_at": _T0,
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
        "created_at": _T0,
        "updated_at": _T0,
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
        "created_at": _T0,
        "updated_at": _T0,
    }
    payload.update(overrides)
    return WorkspaceLiveAccessBinding(**payload)


def _query_policy(**overrides: object) -> WorkspaceQueryPolicy:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "updated_at": _T0,
    }
    payload.update(overrides)
    return WorkspaceQueryPolicy(**payload)


def _legacy_source(**overrides: object) -> WorkspaceSource:
    payload = {
        "source_id": "legacy-source",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "source_type": WorkspaceSourceType.LOCAL_FOLDER,
        "path": "/tmp/docs",
        "status": WorkspaceSourceStatus.REGISTERED,
        "created_at": _T0,
    }
    payload.update(overrides)
    return WorkspaceSource(**payload)


def _connected_source(**overrides: object) -> WorkspaceSource:
    payload = {
        "source_id": "connected-source",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "source_type": WorkspaceSourceType.CONNECTED_SOURCE,
        "status": WorkspaceSourceStatus.REGISTERED,
        "created_at": _T0,
        "knowledge_configuration_creation_mutation_id": _MUTATION,
        "knowledge_configuration_visibility_revision": 4,
    }
    payload.update(overrides)
    return WorkspaceSource(**payload)


def _service_bundle() -> tuple[ManagedWorkspaceRepository, ManagedWorkspaceService, WorkspaceKnowledgeConfigurationService]:
    repo = _repo()
    managed = ManagedWorkspaceService(repo)
    knowledge = WorkspaceKnowledgeConfigurationService(repo, managed)
    return repo, managed, knowledge


def _seed_workspace(repo: ManagedWorkspaceRepository) -> Workspace:
    workspace = _workspace()
    repo.put_workspace(workspace)
    return workspace


class _HeadReadScriptRepository:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        head_reads: list[WorkspaceKnowledgeConfigurationHead | None],
    ) -> None:
        self._repository = repository
        self._head_reads = list(head_reads)
        self._index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._repository, name)

    def get_knowledge_configuration_head(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationHead | None:
        if self._index < len(self._head_reads):
            head = self._head_reads[self._index]
            self._index += 1
            return head
        return self._repository.get_knowledge_configuration_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )


# --- Workspace existence ---


def test_unknown_workspace_returns_none() -> None:
    _, _, knowledge = _service_bundle()
    assert knowledge.get_configuration(tenant_id=_TENANT, workspace_id="missing") is None


def test_cross_tenant_workspace_returns_none() -> None:
    repo, managed, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert (
        knowledge.get_configuration(tenant_id=_TENANT_B, workspace_id=_WORKSPACE) is None
    )
    assert managed.get_source(
        tenant_id=_TENANT_B,
        workspace_id=_WORKSPACE,
        source_id="any",
    ) is None


# --- Empty configuration ---


def test_missing_head_returns_empty_projection_at_revision_zero() -> None:
    repo, _, knowledge = _service_bundle()
    workspace = _seed_workspace(repo)
    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.configuration_revision == 0
    assert config.connection_attachments == ()
    assert config.indexed_sources == ()
    assert config.live_access_bindings == ()
    assert config.query_policy is None
    assert config.updated_at == workspace.updated_at
    assert repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE) is None


# --- Committed selection per child family ---


@pytest.mark.parametrize(
    ("family", "fixture_fn", "put_method", "id_field"),
    [
        (
            "connection_attachment",
            _connection_attachment,
            "put_knowledge_connection_attachment_version_if_absent",
            "attachment_id",
        ),
        (
            "indexed_source",
            _indexed_source,
            "put_knowledge_indexed_source_version_if_absent",
            "indexed_source_binding_id",
        ),
        (
            "live_access",
            _live_access,
            "put_knowledge_live_access_version_if_absent",
            "live_access_binding_id",
        ),
    ],
)
def test_committed_selection_ignores_pending_revision(
    family: str,
    fixture_fn: Any,
    put_method: str,
    id_field: str,
) -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=2)
    ) is True

    rev1 = fixture_fn(effective_revision=1, updated_at=_T0)
    rev2 = fixture_fn(effective_revision=2, updated_at=_T1)
    rev3 = fixture_fn(effective_revision=3, updated_at=_T2)
    assert getattr(repo, put_method)(rev1) is True
    assert getattr(repo, put_method)(rev2) is True
    assert getattr(repo, put_method)(rev3) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.configuration_revision == 2

    if family == "connection_attachment":
        selected = config.connection_attachments
    elif family == "indexed_source":
        selected = config.indexed_sources
    else:
        selected = config.live_access_bindings

    assert len(selected) == 1
    assert getattr(selected[0], id_field) == getattr(rev2, id_field)
    assert selected[0].effective_revision == 2


def test_earlier_committed_fallback_selects_latest_at_or_below_head() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=3)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=1, updated_at=_T0)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=5, updated_at=_T2)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert len(config.indexed_sources) == 1
    assert config.indexed_sources[0].effective_revision == 1


def test_no_applicable_committed_version_omits_logical_entity() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=1)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=2, updated_at=_T1)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.indexed_sources == ()


def test_query_policy_selects_latest_committed_and_ignores_future() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=2)
    ) is True
    assert repo.put_knowledge_query_policy_version_if_absent(
        _query_policy(effective_revision=1, updated_at=_T0, max_result_items=10)
    ) is True
    assert repo.put_knowledge_query_policy_version_if_absent(
        _query_policy(effective_revision=2, updated_at=_T1, max_result_items=20)
    ) is True
    assert repo.put_knowledge_query_policy_version_if_absent(
        _query_policy(effective_revision=3, updated_at=_T2, max_result_items=30)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.query_policy is not None
    assert config.query_policy.effective_revision == 2
    assert config.query_policy.max_result_items == 20


def test_query_policy_none_when_no_committed_policy() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=1)
    ) is True
    assert repo.put_knowledge_query_policy_version_if_absent(
        _query_policy(effective_revision=2, updated_at=_T1)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.query_policy is None


def test_status_preservation_for_disabled_and_detached_entities() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=2)
    ) is True
    assert repo.put_knowledge_connection_attachment_version_if_absent(
        _connection_attachment(
            attachment_id="att-detached",
            status=WorkspaceConnectionAttachmentStatusV1.DETACHED,
            effective_revision=2,
        )
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(
            indexed_source_binding_id="idx-disabled",
            status=WorkspaceIndexedSourceBindingStatusV1.DISABLED,
            effective_revision=2,
        )
    ) is True
    assert repo.put_knowledge_live_access_version_if_absent(
        _live_access(
            live_access_binding_id="live-unavailable",
            status=LiveAccessBindingStatusV1.UNAVAILABLE,
            effective_revision=2,
        )
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.connection_attachments[0].status == WorkspaceConnectionAttachmentStatusV1.DETACHED
    assert config.indexed_sources[0].status == WorkspaceIndexedSourceBindingStatusV1.DISABLED
    assert config.live_access_bindings[0].status == LiveAccessBindingStatusV1.UNAVAILABLE


def test_deterministic_ordering_for_all_child_families() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=1)
    ) is True

    assert repo.put_knowledge_connection_attachment_version_if_absent(
        _connection_attachment(
            attachment_id="att-b",
            connection_ref="conn.z",
            effective_revision=1,
        )
    ) is True
    assert repo.put_knowledge_connection_attachment_version_if_absent(
        _connection_attachment(
            attachment_id="att-a",
            connection_ref="conn.a",
            effective_revision=1,
        )
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(indexed_source_binding_id="idx-b", effective_revision=1)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(indexed_source_binding_id="idx-a", effective_revision=1)
    ) is True
    assert repo.put_knowledge_live_access_version_if_absent(
        _live_access(live_access_binding_id="live-b", effective_revision=1)
    ) is True
    assert repo.put_knowledge_live_access_version_if_absent(
        _live_access(live_access_binding_id="live-a", effective_revision=1)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert [item.attachment_id for item in config.connection_attachments] == ["att-a", "att-b"]
    assert [item.indexed_source_binding_id for item in config.indexed_sources] == ["idx-a", "idx-b"]
    assert [item.live_access_binding_id for item in config.live_access_bindings] == [
        "live-a",
        "live-b",
    ]


def test_configuration_revision_comes_from_head_not_max_child_revision() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=2)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=1, updated_at=_T0)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=5, updated_at=_T2)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.configuration_revision == 2


def test_projection_updated_at_uses_max_committed_child_timestamp() -> None:
    repo, _, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=2)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=1, updated_at=_T0)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=2, updated_at=_T1)
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(effective_revision=3, updated_at=_T2)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.updated_at == _T1


def test_empty_projection_updated_at_uses_workspace_updated_at() -> None:
    repo, _, knowledge = _service_bundle()
    workspace = _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=1)
    ) is True

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.updated_at == workspace.updated_at


# --- Head stability ---


def test_pending_only_head_change_does_not_trigger_instability() -> None:
    repo, managed, _ = _service_bundle()
    _seed_workspace(repo)
    idle_head = _head(committed_revision=4)
    pending_head = _head(
        committed_revision=4,
        pending_revision=5,
        pending_mutation_id="mutation-pending",
    )
    scripted_repo = _HeadReadScriptRepository(
        repo,
        [idle_head, pending_head],
    )
    knowledge = WorkspaceKnowledgeConfigurationService(scripted_repo, managed)

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.configuration_revision == 4


def test_one_successful_retry_rebuilds_at_new_revision() -> None:
    repo, managed, _ = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(
            indexed_source_binding_id="idx-binding",
            effective_revision=1,
            updated_at=_T0,
        )
    ) is True
    assert repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(
            indexed_source_binding_id="idx-binding",
            effective_revision=2,
            updated_at=_T1,
        )
    ) is True

    head_rev1 = _head(committed_revision=1)
    head_rev2 = _head(committed_revision=2)
    scripted_repo = _HeadReadScriptRepository(
        repo,
        [head_rev1, head_rev2, head_rev2, head_rev2],
    )
    knowledge = WorkspaceKnowledgeConfigurationService(scripted_repo, managed)

    config = knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.configuration_revision == 2
    assert len(config.indexed_sources) == 1
    assert config.indexed_sources[0].indexed_source_binding_id == "idx-binding"
    assert config.indexed_sources[0].effective_revision == 2


def test_repeated_revision_change_raises_unstable_error() -> None:
    repo, managed, _ = _service_bundle()
    _seed_workspace(repo)
    heads = [
        _head(committed_revision=1),
        _head(committed_revision=2),
        _head(committed_revision=2),
        _head(committed_revision=3),
    ]
    scripted_repo = _HeadReadScriptRepository(repo, heads)
    knowledge = WorkspaceKnowledgeConfigurationService(scripted_repo, managed)

    with pytest.raises(WorkspaceKnowledgeConfigurationServiceError) as exc_info:
        knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert exc_info.value.error_code == "configuration_projection_unstable"


def test_malformed_child_row_propagates_failure() -> None:
    repo, managed, knowledge = _service_bundle()
    _seed_workspace(repo)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=1)
    ) is True
    partition = f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_indexed_source"
    row_key = f"{_WORKSPACE}:idx-1:rev:00000000000000000001"
    repo.document_store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=row_key,
            data={"not": "a-valid-model"},
        )
    )

    with pytest.raises(ValidationError):
        knowledge.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)


# --- Source visibility helper ---


@pytest.mark.parametrize(
    "source_type",
    [
        WorkspaceSourceType.LOCAL_FOLDER,
        WorkspaceSourceType.MANAGED_UPLOAD,
        WorkspaceSourceType.WEB_RESOURCE,
    ],
)
def test_legacy_and_non_connected_sources_always_visible(source_type: WorkspaceSourceType) -> None:
    kwargs: dict[str, object] = {"source_type": source_type}
    if source_type is WorkspaceSourceType.LOCAL_FOLDER:
        kwargs["path"] = "/tmp/docs"
    else:
        kwargs["path"] = ""
    source = _legacy_source(**kwargs)
    assert is_workspace_source_product_visible(source, committed_configuration_revision=0) is True
    assert is_workspace_source_product_visible(source, committed_configuration_revision=99) is True


@pytest.mark.parametrize(
    ("committed", "visible"),
    [
        (3, False),
        (4, True),
        (5, True),
    ],
)
def test_connected_source_visibility_by_committed_revision(
    committed: int,
    visible: bool,
) -> None:
    source = _connected_source(knowledge_configuration_visibility_revision=4)
    assert (
        is_workspace_source_product_visible(
            source,
            committed_configuration_revision=committed,
        )
        is visible
    )


# --- ManagedWorkspaceService integration ---


def test_list_sources_preserves_order_and_filters_connected_sources() -> None:
    repo, managed, _ = _service_bundle()
    _seed_workspace(repo)
    legacy = _legacy_source(source_id="legacy-1", created_at=_T0)
    pending = _connected_source(
        source_id="connected-pending",
        knowledge_configuration_visibility_revision=4,
        created_at=_T1,
    )
    committed = _connected_source(
        source_id="connected-visible",
        knowledge_configuration_visibility_revision=3,
        created_at=_T2,
    )
    repo.put_source(legacy)
    repo.put_source(pending)
    repo.put_source(committed)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=3)
    ) is True

    listed = managed.list_sources(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert listed is not None
    assert [source.source_id for source in listed] == ["legacy-1", "connected-visible"]


def test_get_source_hides_pending_connected_and_returns_committed() -> None:
    repo, managed, _ = _service_bundle()
    _seed_workspace(repo)
    legacy = _legacy_source(source_id="legacy-1")
    pending = _connected_source(
        source_id="connected-pending",
        knowledge_configuration_visibility_revision=5,
    )
    committed = _connected_source(
        source_id="connected-visible",
        knowledge_configuration_visibility_revision=2,
    )
    repo.put_source(legacy)
    repo.put_source(pending)
    repo.put_source(committed)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=4)
    ) is True

    assert managed.get_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id="legacy-1",
    ) is not None
    assert managed.get_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id="connected-visible",
    ) is not None
    assert (
        managed.get_source(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id="connected-pending",
        )
        is None
    )


def test_get_source_unknown_and_cross_tenant_return_none() -> None:
    repo, managed, _ = _service_bundle()
    _seed_workspace(repo)
    repo.put_source(_legacy_source(source_id="legacy-1"))

    assert (
        managed.get_source(
            tenant_id=_TENANT,
            workspace_id="missing",
            source_id="legacy-1",
        )
        is None
    )
    assert (
        managed.get_source(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE,
            source_id="legacy-1",
        )
        is None
    )
    assert (
        managed.get_source(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id="missing",
        )
        is None
    )


def test_list_sources_unknown_workspace_returns_none() -> None:
    _, managed, _ = _service_bundle()
    assert managed.list_sources(tenant_id=_TENANT, workspace_id="missing") is None


def test_repository_internal_visibility_unfiltered_while_service_hides() -> None:
    repo, managed, _ = _service_bundle()
    _seed_workspace(repo)
    staged = _connected_source(
        source_id="connected-staged",
        knowledge_configuration_visibility_revision=9,
    )
    repo.put_source(staged)
    assert repo.put_knowledge_configuration_head_if_absent(
        _head(committed_revision=2)
    ) is True

    assert repo.get_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id="connected-staged",
    ) is not None
    assert (
        managed.get_source(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id="connected-staged",
        )
        is None
    )
