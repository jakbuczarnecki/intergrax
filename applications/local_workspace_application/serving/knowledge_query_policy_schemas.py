# © Artur Czarnecki. All rights reserved.

"""HTTP schemas for workspace Query Policy and knowledge configuration projection."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UpdateQueryPolicyRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: str = "indexed_only"
    allowed_connection_refs: tuple[str, ...] = ()
    allowed_capability_ids: tuple[str, ...] = ()
    max_live_calls: int = 0
    max_total_duration_ms: int = 30_000
    max_result_items: int = 50
    max_result_bytes: int = 1_048_576
    live_result_retention: str = "ephemeral"


class WorkspaceQueryPolicyResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    mode: str
    allowed_connection_refs: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    max_live_calls: int
    max_total_duration_ms: int
    max_result_items: int
    max_result_bytes: int
    live_result_retention: str
    effective_revision: int
    configuration_revision: int
    updated_at: datetime


class ConnectionAttachmentProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str
    connection_ref: str
    safe_display_label: str
    status: str
    effective_revision: int
    created_at: datetime
    updated_at: datetime


class IndexedSourceBindingProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    indexed_source_binding_id: str
    knowledge_source_binding_ref: str
    source_id: str
    sync_mode: str
    status: str
    audience_eligibility: str
    effective_revision: int
    cached_safe_display_label: str | None
    created_at: datetime
    updated_at: datetime


class LiveAccessBindingProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    live_access_binding_id: str
    connection_ref: str
    remote_resource_id: str | None
    allowed_capability_ids: tuple[str, ...]
    derived_provider_id: str
    derived_integration_kind: str
    derived_resource_type: str | None
    derived_safe_display_label: str
    status: str
    audience_eligibility: str
    effective_revision: int
    created_at: datetime
    updated_at: datetime


class QueryPolicyProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: str
    allowed_connection_refs: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    max_live_calls: int
    max_total_duration_ms: int
    max_result_items: int
    max_result_bytes: int
    live_result_retention: str
    effective_revision: int
    updated_at: datetime


class WorkspaceKnowledgeConfigurationResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    configuration_revision: int
    connection_attachments: tuple[ConnectionAttachmentProjectionV1, ...]
    indexed_sources: tuple[IndexedSourceBindingProjectionV1, ...]
    live_access_bindings: tuple[LiveAccessBindingProjectionV1, ...]
    query_policy: QueryPolicyProjectionV1 | None
    updated_at: datetime
