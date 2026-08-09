"""Jira provider-owned Indexed materialization strategies."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
)
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JIRA_ISSUES_SOURCE_KIND,
    JIRA_PROJECT_SCOPE_TYPE,
    issue_key_project_part,
    validate_jira_issue_key,
    validate_jira_project_key,
)
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    MaterializedConnectedSourceDocument,
    VendorKnowledgeMaterializationError,
    build_materialized_connected_source_document,
    validate_materializer_source,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity

JIRA_ISSUE_STRUCTURED_RECORD_SCHEMA = "jira.issue.knowledge.v1"

_MAX_JIRA_MATERIALIZED_CHARS = 8_000_000
_REMOTE_ID_RE = re.compile(r"^[1-9][0-9]*$")
_JIRA_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
    integration_category=IntegrationCategory.ISSUE_TRACKER,
    source_kind=JIRA_ISSUES_SOURCE_KIND,
)


class _JiraNamedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str | None = None
    name: str


class _JiraProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    key: str
    name: str


class _JiraUserRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    account_id: str | None = None
    display_name: str | None = None
    active: bool | None = None


class _JiraIssueStructuredRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["jira.issue.knowledge.v1"]
    remote_id: str
    key: str
    summary: str
    description: str
    status: _JiraNamedRecord
    issue_type: _JiraNamedRecord
    project: _JiraProjectRecord
    labels: list[str]
    components: list[str]
    created_at: str
    updated_at: str
    web_url: str
    priority: str | None = None
    assignee: _JiraUserRecord | None = None
    reporter: _JiraUserRecord | None = None
    resolution: str | None = None


class JiraIssueStructuredRecordMaterializer:
    """Materialize one accepted Jira issue record into deterministic Markdown."""

    identity = _JIRA_IDENTITY
    runtime_ref = "indexed-source:jira:issues"
    schema_name = JIRA_ISSUE_STRUCTURED_RECORD_SCHEMA

    def materialize(
        self,
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
        revision: KnowledgeItemRevision | None,
        permissions: KnowledgePermissions | None,
    ) -> MaterializedConnectedSourceDocument:
        validate_materializer_source(self.identity, source)
        self._validate_identity(
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
        )
        project_key = self._validate_scope(source)
        self._validate_revision(revision)
        if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeMaterializationError(
                "connected_source_content_mode_invalid"
            )
        if not isinstance(content.structured_record, dict):
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            )
        try:
            record = _JiraIssueStructuredRecord.model_validate(content.structured_record)
        except ValueError:
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            ) from None
        self._validate_record(
            record=record,
            project_key=project_key,
            remote_id=remote_id,
            revision=revision,
        )
        markdown = _render_jira_issue_markdown(record)
        if len(markdown) > _MAX_JIRA_MATERIALIZED_CHARS:
            raise VendorKnowledgeMaterializationError("connected_source_content_too_large")
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=f"jira-issue-{_remote_hash_prefix(remote_id)}.md",
            revision=revision,
            permissions=permissions,
        )

    @staticmethod
    def _validate_identity(
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
    ) -> None:
        if source.tenant_id != tenant_id or any(
            not isinstance(value, str) or not value.strip()
            for value in (tenant_id, workspace_id, binding_id, source_id)
        ):
            raise VendorKnowledgeMaterializationError(
                "connected_source_identity_invalid"
            )
        if not isinstance(remote_id, str) or not _REMOTE_ID_RE.fullmatch(remote_id):
            raise VendorKnowledgeMaterializationError(
                "connected_source_remote_id_mismatch"
            )

    @staticmethod
    def _validate_scope(source: KnowledgeSourceRef) -> str:
        try:
            project_key = validate_jira_project_key(source.scope.remote_scope_id)
        except ValueError:
            raise VendorKnowledgeMaterializationError(
                "connected_source_scope_invalid"
            ) from None
        if (
            source.scope.remote_scope_type != JIRA_PROJECT_SCOPE_TYPE
            or source.scope.parameters
            or source.scope.remote_scope_id != project_key
        ):
            raise VendorKnowledgeMaterializationError("connected_source_scope_invalid")
        return project_key

    @staticmethod
    def _validate_revision(revision: KnowledgeItemRevision | None) -> None:
        if revision is None or not isinstance(revision.version, str):
            raise VendorKnowledgeMaterializationError(
                "connected_source_revision_invalid"
            )
        updated_at = revision.updated_at
        if (
            not isinstance(updated_at, datetime)
            or updated_at.tzinfo is None
            or updated_at.utcoffset() is None
            or revision.version != updated_at.isoformat()
        ):
            raise VendorKnowledgeMaterializationError(
                "connected_source_revision_invalid"
            )

    @staticmethod
    def _validate_record(
        *,
        record: _JiraIssueStructuredRecord,
        project_key: str,
        remote_id: str,
        revision: KnowledgeItemRevision,
    ) -> None:
        if record.remote_id != remote_id:
            raise VendorKnowledgeMaterializationError(
                "connected_source_remote_id_mismatch"
            )
        try:
            issue_key = validate_jira_issue_key(record.key)
        except ValueError:
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            ) from None
        if (
            not _REMOTE_ID_RE.fullmatch(record.remote_id)
            or not record.summary.strip()
            or not record.status.name.strip()
            or not record.issue_type.name.strip()
            or record.project.key != project_key
            or issue_key_project_part(issue_key) != project_key
            or not record.project.name.strip()
        ):
            raise VendorKnowledgeMaterializationError(
                "connected_source_scope_invalid"
            )
        if record.updated_at != revision.version:
            raise VendorKnowledgeMaterializationError(
                "connected_source_revision_invalid"
            )
        for value in (record.created_at, record.updated_at):
            try:
                timestamp = datetime.fromisoformat(value)
            except ValueError:
                raise VendorKnowledgeMaterializationError(
                    "connected_source_revision_invalid"
                ) from None
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise VendorKnowledgeMaterializationError(
                    "connected_source_revision_invalid"
                )


def _render_jira_issue_markdown(record: _JiraIssueStructuredRecord) -> str:
    lines = [
        f"# {record.key}: {record.summary}",
        "",
        f"Status: {record.status.name}",
        f"Type: {record.issue_type.name}",
    ]
    if record.priority:
        lines.append(f"Priority: {record.priority}")
    lines.append(f"Project: {record.project.key}")
    description = _normalize_plain_text(record.description)
    if description:
        lines.extend(["", description])
    if record.labels:
        lines.extend(["", f"Labels: {', '.join(record.labels)}"])
    if record.components:
        lines.append(f"Components: {', '.join(record.components)}")
    if record.assignee is not None:
        assignee = _user_label(record.assignee)
        if assignee:
            lines.append(f"Assignee: {assignee}")
    if record.reporter is not None:
        reporter = _user_label(record.reporter)
        if reporter:
            lines.append(f"Reporter: {reporter}")
    if record.resolution:
        lines.append(f"Resolution: {record.resolution}")
    return "\n".join(lines)


def _normalize_plain_text(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    normalized: list[str] = []
    for line in lines:
        if not line and normalized and not normalized[-1]:
            continue
        normalized.append(line)
    while normalized and not normalized[0]:
        normalized.pop(0)
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized)


def _user_label(user: _JiraUserRecord) -> str:
    return (user.display_name or user.account_id or "").strip()


def _remote_hash_prefix(remote_id: str) -> str:
    return hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "JIRA_ISSUE_STRUCTURED_RECORD_SCHEMA",
    "JiraIssueStructuredRecordMaterializer",
]
