# © Artur Czarnecki. All rights reserved.

"""HTTP schemas for workspace Live Access Binding routes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
)


class CreateWorkspaceLiveAccessBindingRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    allowed_capability_ids: tuple[str, ...] = Field(..., min_length=1)
    audience_eligibility: KnowledgeAudienceEligibilityV1 = (
        KnowledgeAudienceEligibilityV1.PERSONAL_ONLY
    )


class WorkspaceLiveAccessBindingResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
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
    configuration_revision: int
