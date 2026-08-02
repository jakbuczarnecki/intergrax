# © Artur Czarnecki. All rights reserved.

"""Unit tests for Conversation Context durable models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationContextBindingStatus,
    ConversationContextBindingV1,
    ConversationExecutionContextV1,
    ConversationIngressContextV1,
    ConversationObservedAudience,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
    ConversationThreadMemoryLimitsV1,
    ConversationThreadMemoryMessageRole,
    ConversationThreadMemoryMessageV1,
    ConversationWorkspaceResolutionPolicy,
    PersonalConversationStateV1,
    WorkspaceConversationAudience,
    WorkspaceConversationAudiencePolicyV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_BINDING_ID = "binding-1"
_CONNECTION = "conn.primary"
_CONVERSATION = "conv.alpha"
_PRINCIPAL = "principal.alice"
_WORKSPACE = "workspace-1"


def _binding(**overrides: object) -> ConversationContextBindingV1:
    payload = {
        "conversation_context_binding_id": _BINDING_ID,
        "tenant_id": _TENANT,
        "conversation_connection_ref": _CONNECTION,
        "frontend_provider_id": "provider.web",
        "opaque_conversation_ref": _CONVERSATION,
        "audience_mode": ConversationAudienceMode.PERSONAL,
        "workspace_resolution_policy": ConversationWorkspaceResolutionPolicy.FIXED_WORKSPACE,
        "workspace_id": _WORKSPACE,
        "owner_principal_ref": _PRINCIPAL,
        "activation_policy": ConversationActivationPolicy.ALWAYS,
        "thread_context_policy": ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        "administrative_status": ConversationContextBindingStatus.ACTIVE,
        "configuration_version": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return ConversationContextBindingV1(**payload)  # type: ignore[arg-type]


def test_personal_binding_requires_owner_principal() -> None:
    with pytest.raises(ValidationError, match="personal_binding_requires_owner_principal_ref"):
        _binding(owner_principal_ref=None)


def test_shared_binding_forbids_owner_principal() -> None:
    with pytest.raises(ValidationError, match="shared_binding_forbids_owner_principal_ref"):
        _binding(
            audience_mode=ConversationAudienceMode.SHARED,
            owner_principal_ref=_PRINCIPAL,
        )


def test_shared_binding_forbids_personal_selection() -> None:
    with pytest.raises(ValidationError, match="personal_selection_requires_personal_audience"):
        _binding(
            audience_mode=ConversationAudienceMode.SHARED,
            owner_principal_ref=None,
            workspace_resolution_policy=ConversationWorkspaceResolutionPolicy.PERSONAL_SELECTION,
            workspace_id=None,
        )


def test_fixed_workspace_requires_workspace_id() -> None:
    with pytest.raises(ValidationError, match="fixed_workspace_requires_workspace_id"):
        _binding(workspace_id=None)


def test_personal_selection_forbids_binding_workspace_id() -> None:
    with pytest.raises(ValidationError, match="personal_selection_forbids_binding_workspace_id"):
        _binding(
            workspace_resolution_policy=ConversationWorkspaceResolutionPolicy.PERSONAL_SELECTION,
            workspace_id=_WORKSPACE,
        )


def test_unknown_cannot_be_stored_as_durable_audience() -> None:
    with pytest.raises(ValidationError):
        _binding(audience_mode="unknown")  # type: ignore[arg-type]


def test_naive_datetime_is_rejected() -> None:
    naive = datetime(2024, 6, 1, 12, 0)
    with pytest.raises(ValidationError, match="datetime_must_be_timezone_aware"):
        _binding(created_at=naive)


def test_non_utc_offset_is_rejected() -> None:
    warsaw = datetime(2024, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    with pytest.raises(ValidationError, match="datetime_must_be_utc"):
        _binding(updated_at=warsaw)


def test_raw_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _binding(unknown_field="value")  # type: ignore[arg-type]


def test_models_contain_no_credential_or_provider_payload_fields() -> None:
    forbidden = {
        "token",
        "password",
        "client_secret",
        "refresh_token",
        "access_token",
        "provider_payload",
        "raw_event",
        "credentials",
    }
    for model in (
        ConversationIngressContextV1,
        ConversationContextBindingV1,
        WorkspaceConversationAudiencePolicyV1,
        PersonalConversationStateV1,
        ConversationExecutionContextV1,
        ConversationThreadMemoryLimitsV1,
        ConversationThreadMemoryMessageV1,
    ):
        assert forbidden.isdisjoint(set(model.model_fields))


def test_ingress_context_accepts_unknown_observed_audience() -> None:
    ingress = ConversationIngressContextV1(
        conversation_connection_ref=_CONNECTION,
        opaque_conversation_ref=_CONVERSATION,
        opaque_thread_ref="thread-1",
        actor_principal_ref=_PRINCIPAL,
        observed_audience=ConversationObservedAudience.UNKNOWN,
        activation_signal="ordinary_message",  # type: ignore[arg-type]
        provider_event_ref="evt-1",
    )
    assert ingress.observed_audience is ConversationObservedAudience.UNKNOWN


def test_shared_binding_requires_workspace_id() -> None:
    with pytest.raises(ValidationError, match="shared_binding_requires_workspace_id"):
        _binding(
            audience_mode=ConversationAudienceMode.SHARED,
            owner_principal_ref=None,
            workspace_id=None,
        )


def test_personal_selection_requires_personal_audience() -> None:
    with pytest.raises(ValidationError, match="personal_selection_requires_personal_audience"):
        _binding(
            audience_mode=ConversationAudienceMode.SHARED,
            owner_principal_ref=None,
            workspace_resolution_policy=ConversationWorkspaceResolutionPolicy.PERSONAL_SELECTION,
            workspace_id=None,
        )


def test_workspace_audience_policy_round_trip() -> None:
    policy = WorkspaceConversationAudiencePolicyV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        audience=WorkspaceConversationAudience.SHARED,
        configuration_version=1,
        updated_at=_NOW,
    )
    assert policy.audience is WorkspaceConversationAudience.SHARED


def test_personal_state_round_trip() -> None:
    state = PersonalConversationStateV1(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING_ID,
        owner_principal_ref=_PRINCIPAL,
        selected_workspace_id=_WORKSPACE,
        configuration_version=1,
        updated_at=_NOW,
    )
    assert state.selected_workspace_id == _WORKSPACE


def _execution_context(**overrides: object) -> ConversationExecutionContextV1:
    payload = {
        "tenant_id": _TENANT,
        "conversation_context_binding_id": _BINDING_ID,
        "audience_mode": ConversationAudienceMode.PERSONAL,
        "workspace_id": _WORKSPACE,
        "principal_ref": _PRINCIPAL,
        "canonical_thread_ref": "thread-1",
        "activation_policy": ConversationActivationPolicy.ALWAYS,
        "thread_context_policy": ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        "allowed_product_capabilities": frozenset({ConversationProductCapability.READ_ONLY_ASK}),
    }
    payload.update(overrides)
    return ConversationExecutionContextV1(**payload)  # type: ignore[arg-type]


def test_shared_execution_context_accepts_read_only_ask_only() -> None:
    context = _execution_context(
        audience_mode=ConversationAudienceMode.SHARED,
        allowed_product_capabilities=frozenset({ConversationProductCapability.READ_ONLY_ASK}),
    )
    assert context.allowed_product_capabilities == frozenset({ConversationProductCapability.READ_ONLY_ASK})


def test_shared_execution_context_rejects_mutation_capability() -> None:
    with pytest.raises(ValidationError, match="shared_context_requires_read_only_ask_only"):
        _execution_context(
            audience_mode=ConversationAudienceMode.SHARED,
            allowed_product_capabilities=frozenset(
                {
                    ConversationProductCapability.READ_ONLY_ASK,
                    ConversationProductCapability.WORKSPACE_DISCOVERY,
                }
            ),
        )


def test_personal_execution_context_accepts_explicit_capabilities() -> None:
    capabilities = frozenset(
        {
            ConversationProductCapability.READ_ONLY_ASK,
            ConversationProductCapability.SOURCE_DISCOVERY,
        }
    )
    context = _execution_context(allowed_product_capabilities=capabilities)
    assert context.allowed_product_capabilities == capabilities


def test_execution_context_rejects_empty_capability_set() -> None:
    with pytest.raises(ValidationError, match="allowed_product_capabilities_must_be_non_empty"):
        _execution_context(allowed_product_capabilities=frozenset())


def test_memory_limits_reject_zero() -> None:
    with pytest.raises(ValidationError, match="limit_must_be_positive"):
        ConversationThreadMemoryLimitsV1(max_messages=0, max_bytes=100, max_age_seconds=100)


def test_memory_limits_reject_negative_values() -> None:
    with pytest.raises(ValidationError, match="limit_must_be_positive"):
        ConversationThreadMemoryLimitsV1(max_messages=10, max_bytes=-1, max_age_seconds=100)


def test_memory_limits_reject_booleans() -> None:
    with pytest.raises(ValidationError, match="limit_must_be_positive_integer"):
        ConversationThreadMemoryLimitsV1(max_messages=True, max_bytes=100, max_age_seconds=100)  # type: ignore[arg-type]


def test_memory_limits_enforce_upper_bounds() -> None:
    with pytest.raises(ValidationError, match="max_messages_exceeds_upper_bound"):
        ConversationThreadMemoryLimitsV1(max_messages=201, max_bytes=100, max_age_seconds=100)
    with pytest.raises(ValidationError, match="max_bytes_exceeds_upper_bound"):
        ConversationThreadMemoryLimitsV1(max_messages=10, max_bytes=1_000_001, max_age_seconds=100)
    with pytest.raises(ValidationError, match="max_age_seconds_exceeds_upper_bound"):
        ConversationThreadMemoryLimitsV1(max_messages=10, max_bytes=100, max_age_seconds=2_592_001)


def test_memory_message_rejects_blank_content() -> None:
    with pytest.raises(ValidationError, match="content_must_be_non_blank"):
        ConversationThreadMemoryMessageV1(
            role=ConversationThreadMemoryMessageRole.USER,
            content="   ",
            created_at=_NOW,
        )


def test_memory_message_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="datetime_must_be_timezone_aware"):
        ConversationThreadMemoryMessageV1(
            role=ConversationThreadMemoryMessageRole.USER,
            content="hello",
            created_at=datetime(2024, 6, 1, 12, 0),
        )
