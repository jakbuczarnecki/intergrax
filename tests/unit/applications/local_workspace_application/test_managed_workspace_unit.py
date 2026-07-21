# © Artur Czarnecki. All rights reserved.

"""Unit tests for managed workspace domain (LKW-PRODUCT-1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.workspaces.idempotency import (
    content_hash_for_file,
    logical_document_id,
    normalize_source_path,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.path_policy import (
    SourcePathPolicyError,
    validate_local_folder_source_path,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService


@pytest.mark.unit
def test_logical_document_id_stable_for_same_inputs() -> None:
    first = logical_document_id(
        tenant_id="t1",
        workspace_id="w1",
        source_id="s1",
        normalized_source_path="/docs/a.txt",
        content_hash="sha256:abc",
    )
    second = logical_document_id(
        tenant_id="t1",
        workspace_id="w1",
        source_id="s1",
        normalized_source_path="/docs/a.txt",
        content_hash="sha256:abc",
    )
    assert first == second
    assert first.startswith("lkwdoc:")


@pytest.mark.unit
def test_content_hash_changes_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("one", encoding="utf-8")
    first = content_hash_for_file(path)
    path.write_text("two", encoding="utf-8")
    second = content_hash_for_file(path)
    assert first != second
    assert first.startswith("sha256:")


@pytest.mark.unit
def test_path_policy_rejects_out_of_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(SourcePathPolicyError, match="path_not_in_allowlist"):
        validate_local_folder_source_path(
            str(outside),
            allowlist_roots=frozenset({str(allowed.resolve())}),
        )


@pytest.mark.unit
def test_path_policy_rejects_shadow_workspace(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    nested = shadow / "nested"
    nested.mkdir()
    with pytest.raises(SourcePathPolicyError, match="shadow_workspace_not_allowed_as_source"):
        validate_local_folder_source_path(
            str(nested),
            allowlist_roots=frozenset({str(tmp_path.resolve())}),
            shadow_roots=(shadow,),
        )


@pytest.mark.unit
def test_path_policy_accepts_allowlisted_directory(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    resolved = validate_local_folder_source_path(
        str(root),
        allowlist_roots=frozenset({str(tmp_path.resolve())}),
    )
    assert resolved == root.resolve()


@pytest.mark.unit
def test_repository_workspace_and_source_roundtrip() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    now = datetime.now(UTC)
    workspace = Workspace(
        workspace_id="w1",
        tenant_id="tenant-a",
        name="Case",
        description="desc",
        status=WorkspaceStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    repo.put_workspace(workspace)
    assert repo.get_workspace(tenant_id="tenant-a", workspace_id="w1") == workspace
    assert repo.get_workspace(tenant_id="tenant-b", workspace_id="w1") is None

    source = WorkspaceSource(
        source_id="s1",
        workspace_id="w1",
        tenant_id="tenant-a",
        path="/tmp/docs",
        recursive=True,
        status=WorkspaceSourceStatus.REGISTERED,
        created_at=now,
    )
    repo.put_source(source)
    assert repo.get_source(tenant_id="tenant-a", workspace_id="w1", source_id="s1") == source
    assert repo.list_sources(tenant_id="tenant-a", workspace_id="w1") == [source]


@pytest.mark.unit
def test_repository_operation_transitions_and_document_idempotency(tmp_path: Path) -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    operation = WorkspaceOperation(
        operation_id="op1",
        tenant_id="tenant-a",
        workspace_id="w1",
        source_id="s1",
        status=WorkspaceOperationStatus.QUEUED,
    )
    repo.put_operation(operation)
    running = operation.model_copy(update={"status": WorkspaceOperationStatus.RUNNING})
    repo.put_operation(running)
    loaded = repo.get_operation(tenant_id="tenant-a", operation_id="op1")
    assert loaded is not None
    assert loaded.status == WorkspaceOperationStatus.RUNNING
    assert repo.get_operation(tenant_id="tenant-b", operation_id="op1") is None

    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    normalized = normalize_source_path(path)
    digest = content_hash_for_file(path)
    ref = WorkspaceDocumentReference(
        document_id="doc1",
        tenant_id="tenant-a",
        workspace_id="w1",
        source_id="s1",
        source_path=normalized,
        file_name="a.txt",
        content_hash=digest,
        indexed_at=datetime.now(UTC),
    )
    repo.put_document_ref(ref)
    by_path = repo.get_document_ref_by_path(
        tenant_id="tenant-a",
        workspace_id="w1",
        source_id="s1",
        source_path=normalized,
    )
    assert by_path == ref


@pytest.mark.unit
def test_service_create_and_register_source(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    service = ManagedWorkspaceService(
        ManagedWorkspaceRepository(InMemoryDocumentStore()),
        allowlist_roots=frozenset({str(tmp_path.resolve())}),
    )
    workspace = service.create_workspace(tenant_id="tenant-a", name="Buildlogic Legal Case")
    assert workspace.status == WorkspaceStatus.ACTIVE
    source = service.register_local_folder_source(
        tenant_id="tenant-a",
        workspace_id=workspace.workspace_id,
        path=str(docs),
        recursive=True,
    )
    assert source.status == WorkspaceSourceStatus.REGISTERED
    assert Path(source.path) == docs.resolve()
