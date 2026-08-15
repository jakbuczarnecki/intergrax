from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_workspace_application.workspaces.connected_source_materializer import (
    default_connected_source_materializer_registry,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.repository import (
    ManagedWorkspaceRepository,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    VendorKnowledgeMaterializationError,
)
from intergrax.runtime.vendor_knowledge.jira_indexed_materializers import (
    JIRA_ISSUE_STRUCTURED_RECORD_SCHEMA,
    JiraIssueStructuredRecordMaterializer,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)

pytestmark = pytest.mark.unit

_UPDATED_AT = datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc)


def _source() -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id="jira",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        connection_ref="connection-1",
        scope=KnowledgeSourceScope(
            remote_scope_id="PROJ",
            remote_scope_type="jira_project",
            safe_display_name="Project PROJ",
        ),
    )


def _record(
    *,
    remote_id: str = "10001",
    key: str = "PROJ-1",
    summary: str = "Build indexed Jira path",
    description: str = "Issue body evidence",
    updated_at: datetime = _UPDATED_AT,
) -> dict[str, object]:
    return {
        "schema_version": JIRA_ISSUE_STRUCTURED_RECORD_SCHEMA,
        "remote_id": remote_id,
        "key": key,
        "summary": summary,
        "description": description,
        "status": {"id": "3", "name": "In Progress"},
        "issue_type": {"id": "1", "name": "Task"},
        "project": {"id": "10000", "key": "PROJ", "name": "Project"},
        "labels": ["backend"],
        "components": ["API"],
        "created_at": "2024-01-01T10:00:00+00:00",
        "updated_at": updated_at.isoformat(),
        "web_url": "https://example.atlassian.net/browse/PROJ-1",
        "priority": "High",
        "assignee": {"account_id": "acc-1", "display_name": "Alex", "active": True},
        "reporter": {"account_id": "acc-2", "display_name": "Reporter", "active": True},
    }


def _content(**overrides: object) -> KnowledgeContent:
    record = _record()
    record.update(overrides)
    if isinstance(record["updated_at"], datetime):
        record["updated_at"] = record["updated_at"].isoformat()
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record=record,
    )


def _revision(updated_at: datetime = _UPDATED_AT) -> KnowledgeItemRevision:
    return KnowledgeItemRevision(
        version=updated_at.isoformat(),
        updated_at=updated_at,
    )


def _kwargs() -> dict[str, object]:
    return {
        "source": _source(),
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "binding_id": "binding-1",
        "source_id": "source-1",
        "remote_id": "10001",
        "content": _content(),
        "revision": _revision(),
        "permissions": None,
    }


def test_jira_materializer_projects_content_and_preserves_remote_identity() -> None:
    materializer = JiraIssueStructuredRecordMaterializer()
    first = materializer.materialize(**_kwargs())
    newer_updated_at = datetime(2024, 1, 3, 11, 0, tzinfo=timezone.utc)
    newer = materializer.materialize(
        **{
            **_kwargs(),
            "content": _content(
                summary="Changed summary",
                description="Changed body evidence",
                updated_at=newer_updated_at,
            ),
            "revision": _revision(newer_updated_at),
        }
    )

    assert first.document_id == newer.document_id
    assert first.content_hash != newer.content_hash
    assert first.source_revision != newer.source_revision
    assert "# PROJ-1: Build indexed Jira path" in first.markdown
    assert "Issue body evidence" in first.markdown
    assert "Status: In Progress" in first.markdown
    assert "Priority: High" in first.markdown
    assert "Assignee: Alex" in first.markdown
    assert "customfield" not in first.markdown
    assert first.knowledge_document.provenance.provider_id == "jira"
    assert first.knowledge_document.provenance.source_id == "10001"


def test_jira_materializer_is_registered_by_canonical_identity_and_schema() -> None:
    source = _source()
    materializer = default_connected_source_materializer_registry().resolve(
        source,
        schema_name=JIRA_ISSUE_STRUCTURED_RECORD_SCHEMA,
    )

    assert isinstance(materializer, JiraIssueStructuredRecordMaterializer)
    assert materializer.runtime_ref == "indexed-source:jira:issues"


@pytest.mark.parametrize(
    "invalid",
    [
        {"source": _source().model_copy(update={"provider_id": "other"})},
        {
            "source": _source().model_copy(
                update={"integration_kind": IntegrationCategory.WIKI_KNOWLEDGE}
            )
        },
        {"source": _source().model_copy(update={"source_kind": "pages"})},
        {"remote_id": "not-numeric"},
        {"content": _content(schema_version="wrong.schema.v1")},
        {"content": _content(remote_id="10002")},
        {"content": _content(key="OTHER-1")},
        {"revision": KnowledgeItemRevision(version="1")},
        {"content": KnowledgeContent(mode=KnowledgeContentMode.BINARY, binary=b"binary")},
        {
            "content": KnowledgeContent(
                mode=KnowledgeContentMode.STRUCTURED_RECORD,
                structured_record={**_record(), "description": None},
            )
        },
        {
            "content": KnowledgeContent(
                mode=KnowledgeContentMode.STRUCTURED_RECORD,
                structured_record={
                    field: value for field, value in _record().items() if field != "key"
                },
            )
        },
        {"content": _content(description="x" * 8_000_001)},
    ],
)
def test_jira_materializer_fails_closed_for_invalid_contracts(
    invalid: dict[str, object],
) -> None:
    with pytest.raises(VendorKnowledgeMaterializationError):
        JiraIssueStructuredRecordMaterializer().materialize(
            **{**_kwargs(), **invalid}
        )


@pytest.mark.asyncio
async def test_jira_materializer_reaches_generic_index_service(tmp_path: Path) -> None:
    materialized = JiraIssueStructuredRecordMaterializer().materialize(**_kwargs())
    physical_path = tmp_path / materialized.safe_file_name
    physical_path.write_text(materialized.markdown, encoding="utf-8")

    class _Executor:
        async def execute(self, _task):
            return SimpleNamespace(
                metadata={"ingest_summary": {"used": True, "num_chunks": 1}}
            )

    indexing = WorkspaceDocumentIndexingService(
        ManagedWorkspaceRepository(InMemoryDocumentStore()),
        _Executor(),
    )
    result = await indexing.index_connected_source_one(
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        source_id="source-1",
        operation_id="operation-jira-1",
        physical_path=physical_path,
        logical_source_path=materialized.logical_source_path,
        safe_file_name=materialized.safe_file_name,
        content_hash=materialized.content_hash,
        document_id=materialized.document_id,
        materialization_ownership=KnowledgeMaterializationOwnershipV1.connected(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            source_id="source-1",
            indexed_source_binding_id="indexed-binding-1",
            knowledge_source_binding_ref="binding-1",
            delivery_id="delivery-jira-1",
            remote_id="10001",
            materialization_sequence=1,
        ),
    )

    assert result.indexed is True
    assert result.document_id == materialized.document_id
