# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Conversation Context durable models (LKW-CONVERSATION-CONTEXT-1A)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
)

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


class ConversationProductCapability(StrEnum):
    READ_ONLY_ASK = "read_only_ask"
    WORKSPACE_DISCOVERY = "workspace_discovery"
    WORKSPACE_SELECTION = "workspace_selection"
    WORKSPACE_ADMINISTRATION = "workspace_administration"
    SOURCE_DISCOVERY = "source_discovery"
    SOURCE_INTAKE = "source_intake"
    ATTACHMENT_INTAKE = "attachment_intake"
    KNOWLEDGE_CONFIGURATION_DISCOVERY = "knowledge_configuration_discovery"


_SHARED_ONLY_CAPABILITIES = frozenset({ConversationProductCapability.READ_ONLY_ASK})

_MUTATION_PRODUCT_CAPABILITIES = frozenset(
    {
        ConversationProductCapability.WORKSPACE_DISCOVERY,
        ConversationProductCapability.WORKSPACE_SELECTION,
        ConversationProductCapability.WORKSPACE_ADMINISTRATION,
        ConversationProductCapability.SOURCE_DISCOVERY,
        ConversationProductCapability.SOURCE_INTAKE,
        ConversationProductCapability.ATTACHMENT_INTAKE,
    }
)

_MAX_THREAD_MEMORY_MESSAGES = 200
_MAX_THREAD_MEMORY_BYTES = 1_000_000
_MAX_THREAD_MEMORY_AGE_SECONDS = 2_592_000
_MAX_THREAD_MEMORY_MESSAGE_CHARS = 100_000


class ConversationThreadMemoryMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationThreadMemoryLimitsV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_messages: int
    max_bytes: int
    max_age_seconds: int

    @field_validator("max_messages", "max_bytes", "max_age_seconds", mode="before")
    @classmethod
    def _reject_boolean_limits(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("limit_must_be_positive_integer")
        return value

    @field_validator("max_messages", "max_bytes", "max_age_seconds")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("limit_must_be_positive_integer")
        if value <= 0:
            raise ValueError("limit_must_be_positive")
        return value

    @model_validator(mode="after")
    def _validate_upper_bounds(self) -> Self:
        if self.max_messages > _MAX_THREAD_MEMORY_MESSAGES:
            raise ValueError("max_messages_exceeds_upper_bound")
        if self.max_bytes > _MAX_THREAD_MEMORY_BYTES:
            raise ValueError("max_bytes_exceeds_upper_bound")
        if self.max_age_seconds > _MAX_THREAD_MEMORY_AGE_SECONDS:
            raise ValueError("max_age_seconds_exceeds_upper_bound")
        return self


class ConversationThreadMemoryMessageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ConversationThreadMemoryMessageRole
    content: str
    created_at: datetime

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content_must_be_non_blank")
        if len(normalized) > _MAX_THREAD_MEMORY_MESSAGE_CHARS:
            raise ValueError("content_exceeds_max_length")
        return normalized

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        return _validate_utc_datetime(value)


class ConversationExecutionContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    conversation_context_binding_id: str
    audience_mode: ConversationAudienceMode
    workspace_id: str
    principal_ref: str
    canonical_thread_ref: str
    activation_policy: ConversationActivationPolicy
    thread_context_policy: ConversationThreadContextPolicy
    allowed_product_capabilities: frozenset[ConversationProductCapability]

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

    @field_validator("allowed_product_capabilities")
    @classmethod
    def _validate_capabilities(
        cls,
        value: frozenset[ConversationProductCapability],
    ) -> frozenset[ConversationProductCapability]:
        if not value:
            raise ValueError("allowed_product_capabilities_must_be_non_empty")
        return value

    @model_validator(mode="after")
    def _validate_capability_policy(self) -> Self:
        if self.audience_mode is ConversationAudienceMode.SHARED:
            if self.allowed_product_capabilities != _SHARED_ONLY_CAPABILITIES:
                raise ValueError("shared_context_requires_read_only_ask_only")
            if self.allowed_product_capabilities & _MUTATION_PRODUCT_CAPABILITIES:
                raise ValueError("shared_context_rejects_mutation_capabilities")
        return self


class ConversationModelInputKindV1(StrEnum):
    INDEXED_EVIDENCE = "indexed_evidence"
    LIVE_EVIDENCE = "live_evidence"
    THREAD_MEMORY = "thread_memory"
    ATTACHMENT_CONTENT = "attachment_content"
    PLANNER_CONTEXT = "planner_context"
    SYSTEM_CONTEXT = "system_context"


class ConversationScopedModelInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_id: str
    input_kind: ConversationModelInputKindV1
    tenant_id: str
    workspace_id: str
    audience_eligibility: KnowledgeAudienceEligibilityV1
    source_active: bool
    source_ref: str
    origin_audience_mode: ConversationAudienceMode | None = None
    conversation_context_binding_id: str | None = None
    canonical_thread_ref: str | None = None

    @field_validator("input_id")
    @classmethod
    def _validate_input_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="input_id")

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="tenant_id")

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="workspace_id")

    @field_validator("source_ref")
    @classmethod
    def _validate_source_ref(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="source_ref")

    @field_validator("conversation_context_binding_id")
    @classmethod
    def _validate_binding_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_bounded_ref(value, field_name="conversation_context_binding_id")

    @field_validator("canonical_thread_ref")
    @classmethod
    def _validate_canonical_thread_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_opaque_ref(value, field_name="canonical_thread_ref")

    @field_validator("source_active", mode="before")
    @classmethod
    def _validate_source_active_strict(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("source_active_must_be_strict_boolean")
        return value

    @model_validator(mode="after")
    def _validate_thread_memory_requirements(self) -> Self:
        if self.input_kind is not ConversationModelInputKindV1.THREAD_MEMORY:
            return self
        if self.origin_audience_mode is None:
            raise ValueError("thread_memory_requires_origin_audience_mode")
        if self.conversation_context_binding_id is None:
            raise ValueError("thread_memory_requires_conversation_context_binding_id")
        if self.canonical_thread_ref is None:
            raise ValueError("thread_memory_requires_canonical_thread_ref")
        return self


class ConversationOutboundTargetV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    conversation_context_binding_id: str
    audience_mode: ConversationAudienceMode
    workspace_id: str
    canonical_thread_ref: str

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

    @field_validator("canonical_thread_ref")
    @classmethod
    def _validate_canonical_thread_ref(cls, value: str) -> str:
        return _validate_opaque_ref(value, field_name="canonical_thread_ref")


class ConversationApprovedModelInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_id: str
    input_kind: ConversationModelInputKindV1
    audience_eligibility: KnowledgeAudienceEligibilityV1
    origin_audience_mode: ConversationAudienceMode | None = None

    @field_validator("input_id")
    @classmethod
    def _validate_input_id(cls, value: str) -> str:
        return _validate_bounded_ref(value, field_name="input_id")


_CONVERSATION_EXECUTION_GUARD_SCHEMA = "lkw.conversation_execution_guard.v1"
_CONVERSATION_OUTBOUND_GUARD_SCHEMA = "lkw.conversation_outbound_guard.v1"


class ConversationExecutionGuardReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_CONVERSATION_EXECUTION_GUARD_SCHEMA)
    receipt_id: str
    tenant_id: str
    conversation_context_binding_id: str
    audience_mode: ConversationAudienceMode
    workspace_id: str
    canonical_thread_ref: str
    requested_capability: ConversationProductCapability
    approved_inputs: tuple[ConversationApprovedModelInputV1, ...]

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _CONVERSATION_EXECUTION_GUARD_SCHEMA:
            raise ValueError("execution_guard_schema_version_invalid")
        return value

    @field_validator("receipt_id")
    @classmethod
    def _validate_receipt_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("receipt_id_must_be_non_blank")
        return normalized

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

    @field_validator("canonical_thread_ref")
    @classmethod
    def _validate_canonical_thread_ref(cls, value: str) -> str:
        return _validate_opaque_ref(value, field_name="canonical_thread_ref")


class ConversationOutboundGuardReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_CONVERSATION_OUTBOUND_GUARD_SCHEMA)
    receipt_id: str
    execution_receipt_id: str
    tenant_id: str
    conversation_context_binding_id: str
    audience_mode: ConversationAudienceMode
    workspace_id: str
    canonical_thread_ref: str
    used_input_ids: tuple[str, ...]
    citation_input_ids: tuple[str, ...]

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _CONVERSATION_OUTBOUND_GUARD_SCHEMA:
            raise ValueError("outbound_guard_schema_version_invalid")
        return value

    @field_validator("receipt_id", "execution_receipt_id")
    @classmethod
    def _validate_receipt_ids(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("receipt_id_must_be_non_blank")
        return normalized

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

    @field_validator("canonical_thread_ref")
    @classmethod
    def _validate_canonical_thread_ref(cls, value: str) -> str:
        return _validate_opaque_ref(value, field_name="canonical_thread_ref")

    @field_validator("used_input_ids", "citation_input_ids")
    @classmethod
    def _validate_input_id_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            _validate_bounded_ref(item, field_name="input_id")
        return value


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
