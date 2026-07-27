# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for JiraIssuesKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueSearchResult
from intergrax.integrations.providers.issue_tracker.jira.adapter import _JiraIssueTracker
from intergrax.integrations.providers.issue_tracker.jira.client import JiraRestClient
from intergrax.integrations.providers.issue_tracker.jira.config import JiraIntegrationConfig
from intergrax.integrations.providers.issue_tracker.jira.integration import JiraIssueTrackerIntegration
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JIRA_ISSUES_SOURCE_KIND,
    JIRA_KNOWLEDGE_CURSOR_VERSION,
    JIRA_PROJECT_SCOPE_TYPE,
    JiraKnowledgeIssue,
    JiraKnowledgeIssuePage,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    JiraIssuesKnowledgeAdapter,
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _config() -> JiraIntegrationConfig:
    return JiraIntegrationConfig(
        base_url="https://example.atlassian.net",
        email="bot@example.com",
        api_token="secret",
    )


def _source(
    *,
    remote_scope_id: str = "PROJ",
    remote_scope_type: str = JIRA_PROJECT_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id="jira",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=JIRA_ISSUES_SOURCE_KIND,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type=remote_scope_type,
            safe_display_name="Project PROJ",
            parameters=parameters or {},
        ),
    )


def _issue(
    *,
    remote_id: str = "10001",
    key: str = "PROJ-1",
    updated: datetime | None = None,
) -> JiraKnowledgeIssue:
    stamp = updated or datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc)
    return JiraKnowledgeIssue(
        remote_id=remote_id,
        key=key,
        summary="Summary",
        description="Description",
        status_id="3",
        status_name="In Progress",
        issue_type_id="1",
        issue_type_name="Task",
        project_id="10000",
        project_key="PROJ",
        project_name="Project",
        priority_name="High",
        labels=("backend",),
        components=("API",),
        assignee=None,
        reporter=None,
        resolution_name=None,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        updated_at=stamp,
        web_url=f"https://example.atlassian.net/browse/{key}",
    )


def _integration() -> JiraIssueTrackerIntegration:
    rest = JiraRestClient(_config(), http_client=MagicMock())
    return JiraIssueTrackerIntegration.from_client(_JiraIssueTracker(rest))


class _FakeKnowledgeTracker:
    def __init__(
        self,
        *,
        pages: list[JiraKnowledgeIssuePage] | None = None,
        issue: JiraKnowledgeIssue | None = None,
    ) -> None:
        self._pages = list(pages or [])
        self._issue = issue or _issue()
        self.search_calls: list[dict[str, Any]] = []

    def get_issue(self, issue_key: str) -> IssueRecord:
        return IssueRecord(key=issue_key, summary="ok")

    def add_comment(self, issue_key: str, body: str):
        raise NotImplementedError

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        return IssueSearchResult(issues=[], total=0)

    def search_knowledge_issues(
        self,
        *,
        project_key: str,
        next_page_token: str | None,
        limit: int,
    ) -> JiraKnowledgeIssuePage:
        self.search_calls.append(
            {
                "project_key": project_key,
                "next_page_token": next_page_token,
                "limit": limit,
            }
        )
        if not self._pages:
            return JiraKnowledgeIssuePage(issues=(_issue(),), is_last=True)
        return self._pages.pop(0)

    def get_knowledge_issue(self, *, issue_key: str) -> JiraKnowledgeIssue:
        return self._issue


def _integration_with_tracker(tracker: _FakeKnowledgeTracker) -> JiraIssueTrackerIntegration:
    return JiraIssueTrackerIntegration.from_client(tracker)


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return KnowledgeCursor(value=encoded, version=JIRA_KNOWLEDGE_CURSOR_VERSION)


async def test_adapter_identity() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    assert adapter.provider_id == "jira"
    assert adapter.integration_kind is IntegrationCategory.ISSUE_TRACKER
    assert adapter.source_kind == JIRA_ISSUES_SOURCE_KIND


async def test_capabilities_exact_set() -> None:
    caps = JiraIssuesKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is False
    assert caps.content_fetch is True
    assert caps.binary_content is False
    assert caps.rich_text_content is False
    assert caps.structured_content is True
    assert caps.permissions is False
    assert caps.tombstones is False
    assert caps.remote_versions is True
    assert caps.reconciliation is True


async def test_valid_project_scope_inspect() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    source = _source()
    info = await adapter.inspect_scope(integration=_integration(), source=source)
    assert info.safe_display_name == "Project PROJ"
    assert info.capabilities == adapter.capabilities


async def test_invalid_scope_type_rejected() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(),
            source=_source(remote_scope_type="board"),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_non_empty_parameters_rejected() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(parameters={"jql": "bad"}),
            cursor=None,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_invalid_project_key_rejected() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(remote_scope_id="bad key"),
            cursor=None,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert "bad key" not in str(exc_info.value)


async def test_wrong_integration_type_rejected() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=object(), source=_source())
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_first_page_read() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    integration = _integration_with_tracker(_FakeKnowledgeTracker())
    page = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=None,
        limit=10,
    )
    assert len(page.changes) == 1
    assert page.has_more is False


async def test_multipage_continuation() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    tracker = _FakeKnowledgeTracker(
        pages=[
            JiraKnowledgeIssuePage(
                issues=(_issue(remote_id="10001", key="PROJ-1"),),
                next_page_token="page-2",
                is_last=False,
            ),
            JiraKnowledgeIssuePage(
                issues=(_issue(remote_id="10002", key="PROJ-2"),),
                is_last=True,
            ),
        ]
    )
    integration = _integration_with_tracker(tracker)
    first = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=None,
        limit=10,
    )
    assert first.has_more is True
    second = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=first.next_cursor,
        limit=10,
    )
    assert second.has_more is False
    assert tracker.search_calls[1]["next_page_token"] == "page-2"


async def test_intermediate_page_checkpoint_equals_next_cursor() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    integration = _integration_with_tracker(
        _FakeKnowledgeTracker(
            pages=[
                JiraKnowledgeIssuePage(
                    issues=(_issue(),),
                    next_page_token="page-2",
                    is_last=False,
                )
            ]
        )
    )
    page = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=None,
        limit=10,
    )
    assert page.next_cursor == page.proposed_checkpoint


async def test_final_complete_checkpoint() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    integration = _integration_with_tracker(_FakeKnowledgeTracker())
    page = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=None,
        limit=10,
    )
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    raw = base64.urlsafe_b64decode(page.proposed_checkpoint.value + "==")
    payload = json.loads(raw.decode("utf-8"))
    assert payload["complete"] is True
    assert payload["next_page_token"] is None


async def test_complete_cursor_read_rejected() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    cursor = _encode_cursor(
        {
            "schema_version": JIRA_KNOWLEDGE_CURSOR_VERSION,
            "project_key": "PROJ",
            "next_page_token": None,
            "complete": True,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(),
            cursor=cursor,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert cursor.value not in str(exc_info.value)


async def test_cursor_other_project_rejected() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    cursor = _encode_cursor(
        {
            "schema_version": JIRA_KNOWLEDGE_CURSOR_VERSION,
            "project_key": "OTHER",
            "next_page_token": "page-2",
            "complete": False,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(),
            cursor=cursor,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_malformed_cursor_does_not_leak_value() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    secret_cursor = KnowledgeCursor(value="not-valid-base64!!!", version=JIRA_KNOWLEDGE_CURSOR_VERSION)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(),
            cursor=secret_cursor,
            limit=10,
        )
    assert secret_cursor.value not in str(exc_info.value)


async def test_descriptor_identity_mapping() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    integration = _integration_with_tracker(_FakeKnowledgeTracker())
    page = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=None,
        limit=10,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.remote_id == "10001"
    assert descriptor.identity.logical_key == "PROJ-1"
    assert descriptor.identity.parent_remote_id == "10000"


async def test_revision_from_updated_and_provenance_safe() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    updated = datetime(2024, 3, 4, 12, 30, tzinfo=timezone.utc)
    integration = _integration_with_tracker(
        _FakeKnowledgeTracker(
            pages=[
                JiraKnowledgeIssuePage(
                    issues=(_issue(updated=updated),),
                    is_last=True,
                )
            ]
        )
    )
    page = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=None,
        limit=10,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.revision.version == updated.isoformat()
    assert descriptor.revision.updated_at == updated
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.provenance.safe_locator == "PROJ-1"
    assert descriptor.provenance.web_url.endswith("/browse/PROJ-1")


async def test_fetch_content_uses_asyncio_to_thread() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    integration = _integration_with_tracker(_FakeKnowledgeTracker(issue=_issue()))
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="10001", logical_key="PROJ-1"),
        revision=KnowledgeItemRevision(version="1"),
        title="Summary",
        item_type="jira_issue",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="jira",
            source_kind=JIRA_ISSUES_SOURCE_KIND,
            remote_id="10001",
        ),
    )
    with patch(
        "intergrax.runtime.vendor_knowledge.adapters.jira_issues.asyncio.to_thread",
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs),
    ) as to_thread:
        content = await adapter.fetch_content(
            integration=integration,
            source=_source(),
            item=item,
        )
    to_thread.assert_called_once()
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD


async def test_structured_record_shape_and_hash() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    issue = _issue()
    integration = _integration_with_tracker(_FakeKnowledgeTracker(issue=issue))
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="10001", logical_key="PROJ-1"),
        revision=KnowledgeItemRevision(version="1"),
        title="Summary",
        item_type="jira_issue",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="jira",
            source_kind=JIRA_ISSUES_SOURCE_KIND,
            remote_id="10001",
        ),
    )
    first = await adapter.fetch_content(integration=integration, source=_source(), item=item)
    second = await adapter.fetch_content(integration=integration, source=_source(), item=item)
    record = first.structured_record
    assert record is not None
    assert record["schema_version"] == "jira.issue.knowledge.v1"
    assert record["key"] == "PROJ-1"
    assert "comment" not in record
    assert "attachment" not in record
    assert "customfield" not in json.dumps(record)
    assert first.content_hash == second.content_hash
    expected = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert first.content_hash == expected


async def test_fetch_permissions_unsupported() -> None:
    adapter = JiraIssuesKnowledgeAdapter()
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="10001", logical_key="PROJ-1"),
        revision=KnowledgeItemRevision(version="1"),
        title="Summary",
        item_type="jira_issue",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="jira",
            source_kind=JIRA_ISSUES_SOURCE_KIND,
            remote_id="10001",
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(),
            source=_source(),
            item=item,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY


async def test_register_helper_uses_instance_local_registry() -> None:
    registry = KnowledgeAdapterRegistry()
    adapter = register_jira_issues_knowledge_adapter(registry)
    assert isinstance(adapter, JiraIssuesKnowledgeAdapter)
    assert registry.registered_keys() == (("jira", IntegrationCategory.ISSUE_TRACKER, JIRA_ISSUES_SOURCE_KIND),)


async def test_duplicate_registration_rejected() -> None:
    registry = KnowledgeAdapterRegistry()
    register_jira_issues_knowledge_adapter(registry)
    with pytest.raises(ValueError, match="already registered"):
        register_jira_issues_knowledge_adapter(registry)
