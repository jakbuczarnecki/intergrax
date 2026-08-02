# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Conversation Context durable models (LKW-CONVERSATION-CONTEXT-1A)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_OPAQUE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,1023}$")
_MAX_PROVIDER_EVENT_REF_LEN = 256


def _validate_bounded_ref(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name}_must_be_non_blank")
    if _REF_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name}_invalid")
    return normalized


def _validate_opaque_ref(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name}_must_be_non_blank")
    if _OPAQUE_REF_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name}_invalid")
    return normalized


def _validate_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime_must_be_timezone_aware")
    offset = value.utcoffset()
    if offset is None or offset != timedelta(0):
        raise ValueError("datetime_must_be_utc")
    return value


class ConversationObservedAudience(StrEnum):
    PERSONAL = "personal"
    SHARED = "shared"
    UNKNOWN = "unknown"


class ConversationAudienceMode(StrEnum):
    PERSONAL = "personal"
    SHARED = "shared"


class ConversationWorkspaceResolutionPolicy(StrEnum):
    FIXED_WORKSPACE = "fixed_workspace"
    PERSONAL_SELECTION = "personal_selection"


class ConversationActivationPolicy(StrEnum):
    ALWAYS = "always"
    MENTION_ONLY = "mention_only"
    EXPLICIT_COMMAND = "explicit_command"


class ConversationActivationSignal(StrEnum):
    ORDINARY_MESSAGE = "ordinary_message"
    MENTION = "mention"
    EXPLICIT_COMMAND = "explicit_command"
    THREAD_CONTINUATION = "thread_continuation"


class ConversationThreadContextPolicy(StrEnum):
    CURRENT_THREAD_BOUNDED = "current_thread_bounded"


class ConversationContextBindingStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class WorkspaceConversationAudience(StrEnum):
    PERSONAL = "personal"
    SHARED = "shared"


class ConversationIngressContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_connection_ref: str
    opaque_conversation_ref: str
    opaque_thread_ref: str
    actor_principal_ref: str
    observed_audience: ConversationObservedAudience
    activation_signal: ConversationActivationSignal
    provider_event_ref: str

    @field_validator("conversation_connection_ref")
    @classmethod
    def _validate_connection_ref(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="conversation_connection_ref")

    @field_validator("opaque_conversation_ref")
    @classmethod
    def _validate_opaque_conversation_ref(cls, value: str) -> str:
        return _validate_opaque_ref(value, field_name="opaque_conversation_ref")

    @field_validator("opaque_thread_ref")
    @classmethod
    def _validate_opaque_thread_ref(cls, value: str) -> str:
        return _validate_opaque_ref(value, field_name="opaque_thread_ref")

    @field_validator("actor_principal_ref")
    @classmethod
    def _validate_actor_principal_ref(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="actor_principal_ref")

    @field_validator("provider_event_ref")
    @classmethod
    def _validate_provider_event_ref(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider_event_ref_must_be_non_blank")
        if len(normalized) > _MAX_PROVIDER_EVENT_REF_LEN:
            raise ValueError("provider_event_ref_too_long")
        if any(ord(ch) < 32 for ch in normalized):
            raise ValueError("provider_event_ref_invalid")
        return normalized


class ConversationContextBindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_context_binding_id: str
    tenant_id: str
    conversation_connection_ref: str
    frontend_provider_id: str
    opaque_conversation_ref: str
    audience_mode: ConversationAudienceMode
    workspace_resolution_policy: ConversationWorkspaceResolutionPolicy
    workspace_id: str | None = None
    owner_principal_ref: str | None = None
    activation_policy: ConversationActivationPolicy
    thread_context_policy: ConversationThreadContextPolicy
    administrative_status: ConversationContextBindingStatus
    configuration_version: int = Field(..., ge=1)
    created_at: datetime
    updated_at: datetime

    @field_validator("conversation_context_binding_id")
    @classmethod
    def _validate_binding_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="conversation_context_binding_id")

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="tenant_id")

    @field_validator("frontend_provider_id")
    @classmethod
    def _validate_frontend_provider_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="frontend_provider_id")

    @field_validator("conversation_connection_ref")
    @classmethod
    def _validate_connection_ref(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="conversation_connection_ref")

    @field_validator("opaque_conversation_ref")
    @classmethod
    def _validate_opaque_conversation_ref(cls, value: str) -> str:
        return _validate_opaque_ref(value, field_name="opaque_conversation_ref")

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_bounded_ref(value, field_name="workspace_id")

    @field_validator("owner_principal_ref")
    @classmethod
    def _validate_owner_principal_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_bounded_ref(value, field_name="owner_principal_ref")

    @field_validator("created_at", "updated_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime) -> datetime:
        return _validate_utc_datetime(value)

    @model_validator(mode="after")
    def _validate_binding_invariants(self) -> Self:
        if (
            self.workspace_resolution_policy
            is ConversationWorkspaceResolutionPolicy.PERSONAL_SELECTION
        ):
            if self.audience_mode is not ConversationAudienceMode.PERSONAL:
                raise ValueError("personal_selection_requires_personal_audience")
            if self.workspace_id is not None:
                raise ValueError("personal_selection_forbids_binding_workspace_id")

        if self.audience_mode is ConversationAudienceMode.PERSONAL:
            if self.owner_principal_ref is None:
                raise ValueError("personal_binding_requires_owner_principal_ref")
        elif self.audience_mode is ConversationAudienceMode.SHARED:
            if self.owner_principal_ref is not None:
                raise ValueError("shared_binding_forbids_owner_principal_ref")
            if (
                self.workspace_resolution_policy
                is not ConversationWorkspaceResolutionPolicy.FIXED_WORKSPACE
            ):
                raise ValueError("shared_binding_requires_fixed_workspace_policy")
            if self.workspace_id is None:
                raise ValueError("shared_binding_requires_workspace_id")

        if (
            self.workspace_resolution_policy
            is ConversationWorkspaceResolutionPolicy.FIXED_WORKSPACE
        ):
            if self.workspace_id is None:
                raise ValueError("fixed_workspace_requires_workspace_id")

        return self


class WorkspaceConversationAudiencePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    audience: WorkspaceConversationAudience
    configuration_version: int = Field(..., ge=1)
    updated_at: datetime

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="tenant_id")

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="workspace_id")

    @field_validator("updated_at")
    @classmethod
    def _validate_updated_at(cls, value: datetime) -> datetime:
        return _validate_utc_datetime(value)


class PersonalConversationStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    conversation_context_binding_id: str
    owner_principal_ref: str
    selected_workspace_id: str
    configuration_version: int = Field(..., ge=1)
    updated_at: datetime

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="tenant_id")

    @field_validator("conversation_context_binding_id")
    @classmethod
    def _validate_binding_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="conversation_context_binding_id")

    @field_validator("owner_principal_ref")
    @classmethod
    def _validate_owner_principal_ref(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="owner_principal_ref")

    @field_validator("selected_workspace_id")
    @classmethod
    def _validate_selected_workspace_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="selected_workspace_id")

    @field_validator("updated_at")
    @classmethod
    def _validate_updated_at(cls, value: datetime) -> datetime:
        return _validate_utc_datetime(value)


class ResolvedConversationWorkspaceContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    conversation_context_binding_id: str
    audience_mode: ConversationAudienceMode
    workspace_id: str
    principal_ref: str
    canonical_thread_ref: str
    activation_policy: ConversationActivationPolicy
    thread_context_policy: ConversationThreadContextPolicy

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="tenant_id")

    @field_validator("conversation_context_binding_id")
    @classmethod
    def _validate_binding_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="conversation_context_binding_id")

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="workspace_id")

    @field_validator("principal_ref")
    @classmethod
    def _validate_principal_ref(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="principal_ref")

    @field_validator("canonical_thread_ref")
    @classmethod
    def _validate_canonical_thread_ref(cls, value: str) -> str:
        return _validate_opaque_ref(value, field_name="canonical_thread_ref")
