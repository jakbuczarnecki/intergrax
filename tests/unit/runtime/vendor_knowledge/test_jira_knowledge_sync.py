# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Jira knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.issue_tracker.jira.adapter import _JiraIssueTracker
from intergrax.integrations.providers.issue_tracker.jira.client import JiraRestClient
from intergrax.integrations.providers.issue_tracker.jira.config import JiraIntegrationConfig
from intergrax.integrations.providers.issue_tracker.jira.integration import JiraIssueTrackerIntegration
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import JIRA_ISSUES_SOURCE_KIND
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import register_jira_issues_knowledge_adapter
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContentMode,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncRunStatus
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_API_TOKEN = "top-secret-token"
_EMAIL = "bot@example.com"
_JQL = 'project = "PROJ" ORDER BY id ASC'
_PAGE_TOKEN = "page-2"


def _issue_payload(
    *,
    issue_id: str,
    key: str,
    project_key: str = "PROJ",
    project_id: str = "10000",
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "key": key,
        "fields": {
            "summary": f"Summary for {key}",
            "description": f"Description for {key}",
            "status": {"id": "3", "name": "In Progress"},
            "issuetype": {"id": "1", "name": "Task"},
            "project": {"id": project_id, "key": project_key, "name": "Project"},
            "priority": {"name": "High"},
            "labels": ["backend"],
            "components": [{"name": "API"}],
            "assignee": {
                "accountId": "acc-1",
                "displayName": "Alex",
                "active": True,
            },
            "reporter": {
                "accountId": "acc-2",
                "displayName": "Reporter",
                "active": True,
            },
            "resolution": None,
            "created": "2024-01-01T10:00:00.000+0000",
            "updated": "2024-01-02T11:00:00.000+0000",
        },
    }


@dataclass
class _FakeHttpResponse:
    status_code: int = 200
    _payload: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _JiraHttpFake:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def post(self, path: str, *, json: dict[str, Any] | None = None) -> _FakeHttpResponse:
        self.post_calls.append({"path": path, "json": json})
        body = json or {}
        token = body.get("nextPageToken")
        if token is None:
            return _FakeHttpResponse(
                _payload={
                    "issues": [_issue_payload(issue_id="10001", key="PROJ-1")],
                    "isLast": False,
                    "nextPageToken": _PAGE_TOKEN,
                }
            )
        if token == _PAGE_TOKEN:
            return _FakeHttpResponse(
                _payload={
                    "issues": [_issue_payload(issue_id="10002", key="PROJ-2")],
                    "isLast": True,
                }
            )
        raise AssertionError(f"unexpected nextPageToken: {token}")

    def get(self, path: str, *, params: dict[str, str] | None = None) -> _FakeHttpResponse:
        self.get_calls.append({"path": path, "params": params})
        if path == "/issue/PROJ-1":
            return _FakeHttpResponse(_payload=_issue_payload(issue_id="10001", key="PROJ-1"))
        if path == "/issue/PROJ-2":
            return _FakeHttpResponse(_payload=_issue_payload(issue_id="10002", key="PROJ-2"))
        return _FakeHttpResponse(status_code=404)


@dataclass
class _JiraResolver:
    integration: JiraIssueTrackerIntegration

    def resolve(self, *, source) -> JiraIssueTrackerIntegration:
        return self.integration


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_no_secrets(blob: str) -> None:
    forbidden = (
        _API_TOKEN,
        _EMAIL,
        _PAGE_TOKEN,
        _JQL,
        "Authorization",
        "raw-body",
    )
    for item in forbidden:
        assert item not in blob
    assert "credential_ref" not in blob


def _assert_envelope_safe(envelope) -> None:
    blob = _public_blob(
        {
            "descriptor": envelope.descriptor.model_dump(mode="json") if envelope.descriptor else None,
            "content": envelope.content.model_dump(mode="json") if envelope.content else None,
        }
    )
    _assert_no_secrets(blob)
    assert "connection_ref" not in blob


@pytest.mark.asyncio
async def test_jira_facade_coordinator_reconciliation_proof() -> None:
    config = JiraIntegrationConfig(
        base_url="https://example.atlassian.net",
        email=_EMAIL,
        api_token=_API_TOKEN,
    )
    http = _JiraHttpFake()
    integration = JiraIssueTrackerIntegration.from_client(
        _JiraIssueTracker(JiraRestClient(config, http_client=http))
    )
    registry = KnowledgeAdapterRegistry()
    register_jira_issues_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_JiraResolver(integration=integration),
        adapter_registry=registry,
    )
    document_store = InMemoryDocumentStore()
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    sink = IdempotentRecordingSink()
    binding = KnowledgeSourceBinding(
        binding_id="jira-binding",
        tenant_id="tenant-1",
        provider_id="jira",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=JIRA_ISSUES_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Jira Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id="PROJ",
            remote_scope_type="jira_project",
            safe_display_name="Project PROJ",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease_repo,
        checkpoint_repository=checkpoint_repo,
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
    )

    first = await coordinator.reconcile_once(binding_id="jira-binding", restart=True)
    second = await coordinator.reconcile_once(binding_id="jira-binding", restart=False)

    assert first.status is KnowledgeSyncRunStatus.COMPLETED
    assert second.status is KnowledgeSyncRunStatus.COMPLETED
    assert first.has_more is True
    assert second.has_more is False
    assert len(sink.calls) == 2
    assert sink.calls[0].delivery_id != sink.calls[1].delivery_id

    for batch in sink.calls:
        assert batch.mode.value == "reconciliation"
        assert len(batch.envelopes) == 1
        envelope = batch.envelopes[0]
        assert envelope.change_kind.value == "upsert"
        assert envelope.content is not None
        assert envelope.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
        assert envelope.content.structured_record is not None
        assert envelope.content.structured_record["schema_version"] == "jira.issue.knowledge.v1"
        descriptor = envelope.descriptor
        assert descriptor is not None
        assert descriptor.metadata is not None
        assert descriptor.metadata["project_key"] == "PROJ"
        logical_key = descriptor.identity.logical_key
        assert logical_key is not None
        assert logical_key.startswith("PROJ-")
        assert envelope.content.structured_record["remote_id"] == descriptor.identity.remote_id
        _assert_envelope_safe(envelope)
        assert re.fullmatch(r"^[1-9][0-9]*$", descriptor.identity.remote_id)


    remote_ids = set()
    for remote_id in ("10001", "10002"):
        assert re.fullmatch(r"^[1-9][0-9]*$", remote_id)
        state = state_repo.get(
            tenant_id="tenant-1",
            binding_id="jira-binding",
            remote_id=remote_id,
        )
        assert state is not None
        remote_ids.add(state.remote_id)
    assert remote_ids == {"10001", "10002"}

    checkpoint = checkpoint_repo.get(tenant_id="tenant-1", binding_id="jira-binding")
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == "jira.issues.cursor.v1"
    assert _PAGE_TOKEN not in _public_blob(checkpoint.model_dump(mode="json"))

    busy_lease = lease_repo.acquire(
        tenant_id="tenant-1",
        binding_id="jira-binding",
        owner_id="owner-2",
        ttl_seconds=30,
    )
    assert busy_lease is not None
    assert http.post_calls[0]["path"] == "/search/jql"
    assert http.post_calls[0]["json"]["jql"] == _JQL
    assert "nextPageToken" not in http.post_calls[0]["json"]
    assert http.post_calls[1]["json"]["nextPageToken"] == _PAGE_TOKEN
    assert {call["path"] for call in http.get_calls} == {"/issue/PROJ-1", "/issue/PROJ-2"}

    public_proof = _public_blob(
        {
            "first": first.model_dump(mode="json"),
            "second": second.model_dump(mode="json"),
            "envelopes": [
                {
                    "descriptor": batch.envelopes[0].descriptor.model_dump(mode="json"),
                    "content": batch.envelopes[0].content.model_dump(mode="json"),
                }
                for batch in sink.calls
            ],
        }
    )
    _assert_no_secrets(public_proof)


class _JiraHttpFakeWrongProjectFirstPage:
    def post(self, path: str, *, json: dict[str, Any] | None = None) -> _FakeHttpResponse:
        return _FakeHttpResponse(
            _payload={
                "issues": [
                    _issue_payload(
                        issue_id="10099",
                        key="OTHER-1",
                        project_key="OTHER",
                        project_id="20000",
                    )
                ],
                "isLast": True,
            }
        )

    def get(self, path: str, *, params: dict[str, str] | None = None) -> _FakeHttpResponse:
        raise AssertionError("get should not be called")


@pytest.mark.asyncio
async def test_jira_coordinator_rejects_cross_project_first_page() -> None:
    config = JiraIntegrationConfig(
        base_url="https://example.atlassian.net",
        email=_EMAIL,
        api_token=_API_TOKEN,
    )
    integration = JiraIssueTrackerIntegration.from_client(
        _JiraIssueTracker(JiraRestClient(config, http_client=_JiraHttpFakeWrongProjectFirstPage()))
    )
    registry = KnowledgeAdapterRegistry()
    register_jira_issues_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_JiraResolver(integration=integration),
        adapter_registry=registry,
    )
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = KnowledgeSourceBinding(
        binding_id="jira-binding",
        tenant_id="tenant-1",
        provider_id="jira",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=JIRA_ISSUES_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Jira Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id="PROJ",
            remote_scope_type="jira_project",
            safe_display_name="Project PROJ",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=DocumentStoreKnowledgeSourceLeaseRepository(document_store),
        checkpoint_repository=DocumentStoreKnowledgeSyncCheckpointRepository(document_store),
        item_state_repository=DocumentStoreKnowledgeRemoteItemStateRepository(document_store),
        sink=sink,
        lease_ttl_seconds=30,
    )

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(binding_id="jira-binding", restart=True)

    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert sink.calls == []
    checkpoint = DocumentStoreKnowledgeSyncCheckpointRepository(document_store).get(
        tenant_id="tenant-1",
        binding_id="jira-binding",
    )
    assert checkpoint is None
    state = DocumentStoreKnowledgeRemoteItemStateRepository(document_store).get(
        tenant_id="tenant-1",
        binding_id="jira-binding",
        remote_id="10099",
    )
    assert state is None
