# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira issues knowledge source adapter (JIRA-KNOWLEDGE-ADAPTER-1)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.issue_tracker.jira.integration import JiraIssueTrackerIntegration
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JIRA_ISSUES_SOURCE_KIND,
    JIRA_KNOWLEDGE_CURSOR_VERSION,
    JIRA_PROJECT_SCOPE_TYPE,
    JiraKnowledgeIssue,
    JiraKnowledgeUser,
    issue_key_project_part,
    validate_jira_issue_key,
    validate_jira_knowledge_issue_project_scope,
    validate_jira_project_key,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    JsonObject,
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

_STRUCTURED_RECORD_SCHEMA = "jira.issue.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.jira-issue+json"
_JIRA_REMOTE_ID_RE = re.compile(r"^[1-9][0-9]*$")


class _JiraIssuesReconciliationCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["jira.issues.cursor.v1"]
    project_key: str
    next_page_token: str | None = Field(default=None, repr=False)
    complete: bool

    @field_validator("project_key")
    @classmethod
    def _validate_project_key(cls, value: str) -> str:
        return validate_jira_project_key(value)

    @field_validator("next_page_token")
    @classmethod
    def _validate_next_page_token(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("next_page_token must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _token_rules(self) -> _JiraIssuesReconciliationCursor:
        if not self.complete and not self.next_page_token:
            raise ValueError("next_page_token is required when complete is False")
        if self.complete and self.next_page_token is not None:
            raise ValueError("next_page_token must be None when complete is True")
        return self


class JiraIssuesKnowledgeAdapter:
    """Thin mapping from Jira issue tracker integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return "jira"

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.ISSUE_TRACKER

    @property
    def source_kind(self) -> str:
        return JIRA_ISSUES_SOURCE_KIND

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=False,
            content_fetch=True,
            binary_content=False,
            rich_text_content=False,
            structured_content=True,
            permissions=False,
            tombstones=False,
            remote_versions=True,
            reconciliation=True,
        )

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self._require_jira_integration(integration=integration, source=source)
        self._validate_source(source)
        return KnowledgeScopeInfo(
            source=source,
            capabilities=self.capabilities,
            safe_display_name=source.scope.safe_display_name,
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        jira_integration = self._require_jira_integration(integration=integration, source=source)
        project_key = self._validate_source(source)
        decoded = self._decode_cursor(cursor, project_key=project_key)
        if decoded is not None and decoded.complete:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Jira reconciliation cursor is complete; restart reconciliation",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )

        next_page_token = None if decoded is None else decoded.next_page_token
        page = await asyncio.to_thread(
            jira_integration.search_knowledge_issues,
            project_key=project_key,
            next_page_token=next_page_token,
            limit=limit,
        )
        try:
            seen_remote_ids: set[str] = set()
            for issue in page.issues:
                if issue.remote_id in seen_remote_ids:
                    raise ValueError("duplicate issue id on page")
                seen_remote_ids.add(issue.remote_id)
                self._validate_issue_for_source(issue, project_key=project_key)
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Jira knowledge page contains an issue outside the requested project",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        changes = tuple(self._issue_to_change(issue) for issue in page.issues)
        if not page.is_last:
            checkpoint = self._encode_cursor(
                _JiraIssuesReconciliationCursor(
                    schema_version=JIRA_KNOWLEDGE_CURSOR_VERSION,
                    project_key=project_key,
                    next_page_token=page.next_page_token,
                    complete=False,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=checkpoint,
                proposed_checkpoint=checkpoint,
                has_more=True,
            )

        final_checkpoint = self._encode_cursor(
            _JiraIssuesReconciliationCursor(
                schema_version=JIRA_KNOWLEDGE_CURSOR_VERSION,
                project_key=project_key,
                next_page_token=None,
                complete=True,
            )
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=final_checkpoint,
            has_more=False,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        jira_integration = self._require_jira_integration(integration=integration, source=source)
        project_key = self._validate_source(source)
        self._validate_item(item, source=source)
        issue_key = item.identity.logical_key
        if issue_key is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue logical key is required",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            validated_issue_key = validate_jira_issue_key(issue_key)
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue logical key is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

        issue = await asyncio.to_thread(
            jira_integration.get_knowledge_issue,
            issue_key=validated_issue_key,
        )
        try:
            self._validate_issue_for_source(issue, project_key=project_key)
            self._validate_fetched_issue_identity(
                issue,
                item=item,
                project_key=project_key,
            )
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Jira issue response identity does not match requested item",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        structured_record = _build_structured_record(issue)
        canonical = json.dumps(
            structured_record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        content_hash = hashlib.sha256(canonical).hexdigest()
        return KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record=structured_record,
            mime_type=_STRUCTURED_RECORD_MIME,
            content_hash=content_hash,
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self._require_jira_integration(integration=integration, source=source)
        self._validate_source(source)
        self._validate_item(item, source=source)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message="Jira issue permission projection is not implemented",
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _require_jira_integration(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> JiraIssueTrackerIntegration:
        if not isinstance(integration, JiraIssueTrackerIntegration):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Jira knowledge adapter requires Jira issue tracker integration",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return integration

    def _validate_source(self, source: KnowledgeSourceRef) -> str:
        if (
            source.provider_id != self.provider_id
            or source.integration_kind != self.integration_kind
            or source.source_kind != self.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source identity is not supported by the Jira issues adapter",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        scope = source.scope
        if scope.remote_scope_type != JIRA_PROJECT_SCOPE_TYPE:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source scope type is not supported",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if scope.parameters:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source scope parameters are not supported",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        try:
            return validate_jira_project_key(scope.remote_scope_id)
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source scope identifier is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None

    def _validate_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
    ) -> None:
        provenance = item.provenance
        if (
            provenance.provider_id != source.provider_id
            or provenance.source_kind != source.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Item provenance does not match the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue content mode must be structured record",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.item_type != "jira_issue":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira knowledge item type is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if not item.content_available:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira knowledge item content is not available",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        remote_id = item.identity.remote_id
        if not _JIRA_REMOTE_ID_RE.fullmatch(str(remote_id).strip()):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue remote id is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        logical_key = item.identity.logical_key
        if logical_key is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue logical key is required",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        try:
            validated_logical_key = validate_jira_issue_key(logical_key)
            project_key = validate_jira_project_key(source.scope.remote_scope_id)
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue logical key is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        if issue_key_project_part(validated_logical_key) != project_key:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue logical key does not match source scope",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        parent_remote_id = item.identity.parent_remote_id
        if parent_remote_id is not None and not str(parent_remote_id).strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Jira issue parent remote id is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _validate_issue_for_source(
        self,
        issue: JiraKnowledgeIssue,
        *,
        project_key: str,
    ) -> None:
        validate_jira_knowledge_issue_project_scope(issue, project_key=project_key)
        if issue.created_at.tzinfo is None or issue.updated_at.tzinfo is None:
            raise ValueError("Jira issue timestamps must be timezone-aware")

    def _validate_fetched_issue_identity(
        self,
        issue: JiraKnowledgeIssue,
        *,
        item: KnowledgeItemDescriptor,
        project_key: str,
    ) -> None:
        validated_issue_key = validate_jira_issue_key(item.identity.logical_key or "")
        if issue.key != validated_issue_key:
            raise ValueError("Jira issue response identity does not match requested item")
        if issue.remote_id != str(item.identity.remote_id).strip():
            raise ValueError("Jira issue response identity does not match requested item")
        if issue.project_key != project_key:
            raise ValueError("Jira issue response identity does not match requested item")
        if issue_key_project_part(issue.key) != project_key:
            raise ValueError("Jira issue response identity does not match requested item")
        parent_remote_id = item.identity.parent_remote_id
        if parent_remote_id is not None and issue.project_id != str(parent_remote_id).strip():
            raise ValueError("Jira issue response identity does not match requested item")

    def _issue_to_change(self, issue: JiraKnowledgeIssue) -> KnowledgeChange:
        descriptor = KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=issue.remote_id,
                logical_key=issue.key,
                parent_remote_id=issue.project_id,
            ),
            revision=KnowledgeItemRevision(
                version=issue.updated_at.isoformat(),
                updated_at=issue.updated_at,
            ),
            title=issue.summary,
            item_type="jira_issue",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=issue.remote_id,
                web_url=issue.web_url,
                safe_locator=issue.key,
            ),
            metadata={
                "issue_key": issue.key,
                "project_key": issue.project_key,
                "status": issue.status_name,
                "issue_type": issue.issue_type_name,
            },
        )
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=issue.remote_id,
            descriptor=descriptor,
        )

    def _encode_cursor(self, cursor: _JiraIssuesReconciliationCursor) -> KnowledgeCursor:
        payload = cursor.model_dump()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return KnowledgeCursor(value=encoded, version=JIRA_KNOWLEDGE_CURSOR_VERSION)

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        project_key: str,
    ) -> _JiraIssuesReconciliationCursor | None:
        if cursor is None:
            return None
        if cursor.version != JIRA_KNOWLEDGE_CURSOR_VERSION:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Jira reconciliation cursor version is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            padding = "=" * (-len(cursor.value) % 4)
            raw = base64.urlsafe_b64decode(cursor.value + padding)
            data = json.loads(raw.decode("utf-8"))
            decoded = _JiraIssuesReconciliationCursor.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Jira reconciliation cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if decoded.project_key != project_key:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Jira reconciliation cursor scope does not match source",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded


def _named_object(*, object_id: str | None, name: str) -> dict[str, str]:
    payload: dict[str, str] = {"name": name}
    if object_id is not None:
        payload["id"] = object_id
    return payload


def _user_object(user: JiraKnowledgeUser | None) -> dict[str, Any] | None:
    if user is None:
        return None
    payload: dict[str, Any] = {}
    if user.account_id is not None:
        payload["account_id"] = user.account_id
    if user.display_name is not None:
        payload["display_name"] = user.display_name
    if user.active is not None:
        payload["active"] = user.active
    return payload or None


def _build_structured_record(issue: JiraKnowledgeIssue) -> JsonObject:
    record: JsonObject = {
        "schema_version": _STRUCTURED_RECORD_SCHEMA,
        "remote_id": issue.remote_id,
        "key": issue.key,
        "summary": issue.summary,
        "description": issue.description,
        "status": _named_object(object_id=issue.status_id, name=issue.status_name),
        "issue_type": _named_object(
            object_id=issue.issue_type_id,
            name=issue.issue_type_name,
        ),
        "project": {
            "id": issue.project_id,
            "key": issue.project_key,
            "name": issue.project_name,
        },
        "labels": list(issue.labels),
        "components": list(issue.components),
        "created_at": issue.created_at.isoformat(),
        "updated_at": issue.updated_at.isoformat(),
        "web_url": issue.web_url,
    }
    if issue.priority_name is not None:
        record["priority"] = issue.priority_name
    assignee = _user_object(issue.assignee)
    if assignee is not None:
        record["assignee"] = assignee
    reporter = _user_object(issue.reporter)
    if reporter is not None:
        record["reporter"] = reporter
    if issue.resolution_name is not None:
        record["resolution"] = issue.resolution_name
    return record


def register_jira_issues_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> JiraIssuesKnowledgeAdapter:
    adapter = JiraIssuesKnowledgeAdapter()
    registry.register(adapter)
    return adapter
