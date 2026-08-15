# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.workspaces.document_inspect_service import (
    DocumentInspectError,
    DocumentInspectService,
    _MAX_PREVIEW_CHARS,
)
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    KnowledgeInputKind,
    KnowledgeInputStatus,
    ManagedFileObject,
    ManagedFileObjectStatus,
    WebUrlSourceLocator,
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 9, 0, tzinfo=UTC)
_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE_A = "ws-a"
_WORKSPACE_B = "ws-b"
_SOURCE_MANAGED = "src-managed"
_SOURCE_WEB = "src-web"
_SOURCE_CONNECTED = "src-connected"
_DOCUMENT_ID = "doc-managed-1"


@pytest.fixture
def repository() -> ManagedWorkspaceRepository:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    for tenant_id, workspace_id in (
        (_TENANT_A, _WORKSPACE_A),
        (_TENANT_B, _WORKSPACE_B),
    ):
        repo.put_workspace(
            Workspace(
                workspace_id=workspace_id,
                tenant_id=tenant_id,
                name="Docs",
                status=WorkspaceStatus.ACTIVE,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    return repo


@pytest.fixture
def service(repository: ManagedWorkspaceRepository) -> DocumentInspectService:
    return DocumentInspectService(repository=repository)


def _seed_managed_document(repository: ManagedWorkspaceRepository) -> None:
    repository.put_source(
        WorkspaceSource(
            source_id=_SOURCE_MANAGED,
            workspace_id=_WORKSPACE_A,
            tenant_id=_TENANT_A,
            source_type=WorkspaceSourceType.MANAGED_UPLOAD,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.READY,
            created_at=_NOW,
        )
    )
    repository.put_managed_file(
        ManagedFileObject(
            object_id="obj-1",
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            input_id="input-1",
            operation_id="op-1",
            source_id=_SOURCE_MANAGED,
            storage_key="tenant-a/ws-a/secret/path/report.pdf",
            safe_file_name="report.pdf",
            content_type="application/pdf",
            size_bytes=128,
            content_hash="sha256:" + "a" * 64,
            status=ManagedFileObjectStatus.STORED,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repository.put_document_ref(
        WorkspaceDocumentReference(
            document_id=_DOCUMENT_ID,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            source_id=_SOURCE_MANAGED,
            source_path="managed/report.pdf",
            file_name="report.pdf",
            content_hash="sha256:" + "b" * 64,
            indexed_at=_NOW,
        )
    )


def _seed_web_document(repository: ManagedWorkspaceRepository) -> None:
    repository.put_source(
        WorkspaceSource(
            source_id=_SOURCE_WEB,
            workspace_id=_WORKSPACE_A,
            tenant_id=_TENANT_A,
            source_type=WorkspaceSourceType.WEB_RESOURCE,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.READY,
            created_at=_NOW,
        )
    )
    repository.put_knowledge_input(
        KnowledgeInput(
            input_id="input-web-1",
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            input_kind=KnowledgeInputKind.WEB_URL,
            idempotency_key="idem-web",
            operation_id="op-web",
            source_id=_SOURCE_WEB,
            status=KnowledgeInputStatus.RESOLVED,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repository.put_web_url_locator(
        WebUrlSourceLocator(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            input_id="input-web-1",
            canonical_private_url="https://example.com/docs/policy?token=secret",
            requested_url_fingerprint="sha256:" + "c" * 64,
            safe_display_url="https://example.com/docs/policy",
            final_safe_display_url="https://example.com/docs/policy",
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repository.put_document_ref(
        WorkspaceDocumentReference(
            document_id="doc-web-1",
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            source_id=_SOURCE_WEB,
            source_path="web/content.txt",
            file_name="Policy page",
            content_hash="sha256:" + "d" * 64,
            indexed_at=_NOW,
        )
    )


def _seed_connected_document(repository: ManagedWorkspaceRepository) -> None:
    repository.put_source(
        WorkspaceSource(
            source_id=_SOURCE_CONNECTED,
            workspace_id=_WORKSPACE_A,
            tenant_id=_TENANT_A,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.READY,
            created_at=_NOW,
            knowledge_configuration_creation_mutation_id="mut-1",
            knowledge_configuration_visibility_revision=1,
        )
    )
    repository.put_document_ref(
        WorkspaceDocumentReference(
            document_id="doc-connected-1",
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            source_id=_SOURCE_CONNECTED,
            source_path="connected/page-1",
            file_name="Architecture Decision Record",
            content_hash="sha256:" + "e" * 64,
            indexed_at=_NOW,
        )
    )


def test_inspect_managed_file_citation(service: DocumentInspectService, repository: ManagedWorkspaceRepository) -> None:
    _seed_managed_document(repository)
    view = service.inspect(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        document_id=_DOCUMENT_ID,
        preview_hint="Payment terms are net 30.",
        page=4,
    )
    assert view.document_id == _DOCUMENT_ID
    assert view.display_name == "report.pdf"
    assert view.source_type == "managed_upload"
    assert view.preview == "Payment terms are net 30."
    assert view.location is not None
    assert view.location.page == 4
    assert view.external_url is None


def test_inspect_correct_tenant_workspace_succeeds(
    service: DocumentInspectService,
    repository: ManagedWorkspaceRepository,
) -> None:
    _seed_managed_document(repository)
    view = service.inspect(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        document_id=_DOCUMENT_ID,
    )
    assert view.source_id == _SOURCE_MANAGED


def test_inspect_wrong_workspace_fails_closed(
    service: DocumentInspectService,
    repository: ManagedWorkspaceRepository,
) -> None:
    _seed_managed_document(repository)
    with pytest.raises(DocumentInspectError) as exc:
        service.inspect(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            document_id=_DOCUMENT_ID,
        )
    assert exc.value.error_code == "document_not_found"


def test_inspect_wrong_tenant_fails_closed(
    service: DocumentInspectService,
    repository: ManagedWorkspaceRepository,
) -> None:
    _seed_managed_document(repository)
    with pytest.raises(DocumentInspectError):
        service.inspect(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            document_id=_DOCUMENT_ID,
        )


def test_inspect_unknown_document_not_found(service: DocumentInspectService) -> None:
    with pytest.raises(DocumentInspectError) as exc:
        service.inspect(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            document_id="missing-doc",
        )
    assert exc.value.error_code == "document_not_found"


def test_inspect_preview_bounded(service: DocumentInspectService, repository: ManagedWorkspaceRepository) -> None:
    _seed_managed_document(repository)
    view = service.inspect(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        document_id=_DOCUMENT_ID,
        preview_hint="x" * (_MAX_PREVIEW_CHARS + 50),
    )
    assert view.preview is not None
    assert len(view.preview) <= _MAX_PREVIEW_CHARS


def test_inspect_does_not_expose_raw_local_path(
    service: DocumentInspectService,
    repository: ManagedWorkspaceRepository,
) -> None:
    _seed_managed_document(repository)
    view = service.inspect(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        document_id=_DOCUMENT_ID,
    )
    payload = view.model_dump_json()
    assert "secret/path" not in payload
    assert "storage_key" not in payload
    assert "managed/report.pdf" not in payload


def test_web_url_open_target_from_canonical_provenance(
    service: DocumentInspectService,
    repository: ManagedWorkspaceRepository,
) -> None:
    _seed_web_document(repository)
    view = service.inspect(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        document_id="doc-web-1",
    )
    assert view.external_url == "https://example.com/docs/policy"
    assert "token=secret" not in (view.external_url or "")


def test_connected_source_without_deep_link_has_no_fake_target(
    service: DocumentInspectService,
    repository: ManagedWorkspaceRepository,
) -> None:
    _seed_connected_document(repository)
    view = service.inspect(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        document_id="doc-connected-1",
    )
    assert view.external_url is None
    assert view.display_name == "Architecture Decision Record"


def test_unsafe_document_identifier_rejected(service: DocumentInspectService) -> None:
    with pytest.raises(DocumentInspectError):
        service.inspect(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            document_id="../etc/passwd",
        )
