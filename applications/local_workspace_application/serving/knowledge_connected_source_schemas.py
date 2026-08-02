# © Artur Czarnecki. All rights reserved.

"""HTTP schemas for connected workspace knowledge sources."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RemoteResourceCandidateResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opaque_candidate_ref: str
    resource_type: str
    safe_display_label: str
    conversation_kind: str | None = None
    is_archived: bool = False
    is_private: bool = False
    safe_description: str | None = None


class RemoteResourceDiscoveryResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[RemoteResourceCandidateResponseV1]
    next_cursor: str | None = None


class CreateConnectedIndexedSourceRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(..., min_length=1, max_length=128)
    opaque_candidate_ref: str = Field(..., min_length=1, max_length=1024)
    root_oldest: str = Field(..., min_length=1, max_length=64)
    root_latest: str = Field(..., min_length=1, max_length=64)


class CreateConnectedIndexedSourceResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    indexed_source_binding_id: str
    source_id: str
    knowledge_source_binding_ref: str
    safe_display_label: str
    status: str
    sync_mode: str
    audience_eligibility: str
    configuration_revision: int


class ConnectedIndexedSourceSyncAcceptedV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    workspace_id: str
    source_id: str
    status: str
    created_at: datetime | None = None
