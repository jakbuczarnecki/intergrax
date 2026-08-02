# © Artur Czarnecki. All rights reserved.

"""Unit tests for Conversation Execution Context builder."""

from __future__ import annotations

import pytest

from local_workspace_application.workspaces.conversation_context_execution import (
    ConversationExecutionContextError,
    build_conversation_execution_context,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
    ResolvedConversationWorkspaceContextV1,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_BINDING = "binding-1"
_WORKSPACE = "workspace-1"
_PRINCIPAL = "principal.alice"
_THREAD = "thread-1"


def _resolved(*, audience: ConversationAudienceMode = ConversationAudienceMode.PERSONAL) -> ResolvedConversationWorkspaceContextV1:
    return ResolvedConversationWorkspaceContextV1(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        audience_mode=audience,
        workspace_id=_WORKSPACE,
        principal_ref=_PRINCIPAL,
        canonical_thread_ref=_THREAD,
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
    )


def test_builder_copies_all_resolved_identity_fields() -> None:
    resolved = _resolved()
    capabilities = frozenset({ConversationProductCapability.READ_ONLY_ASK})
    context = build_conversation_execution_context(
        resolved=resolved,
        personal_allowed_capabilities=capabilities,
    )
    assert context.tenant_id == resolved.tenant_id
    assert context.conversation_context_binding_id == resolved.conversation_context_binding_id
    assert context.audience_mode == resolved.audience_mode
    assert context.workspace_id == resolved.workspace_id
    assert context.principal_ref == resolved.principal_ref
    assert context.canonical_thread_ref == resolved.canonical_thread_ref
    assert context.activation_policy == resolved.activation_policy
    assert context.thread_context_policy == resolved.thread_context_policy


def test_shared_produces_exactly_read_only_ask() -> None:
    context = build_conversation_execution_context(resolved=_resolved(audience=ConversationAudienceMode.SHARED))
    assert context.allowed_product_capabilities == frozenset({ConversationProductCapability.READ_ONLY_ASK})


def test_shared_expansion_attempt_fails() -> None:
    with pytest.raises(ConversationExecutionContextError) as exc_info:
        build_conversation_execution_context(
            resolved=_resolved(audience=ConversationAudienceMode.SHARED),
            personal_allowed_capabilities=frozenset(
                {
                    ConversationProductCapability.READ_ONLY_ASK,
                    ConversationProductCapability.WORKSPACE_DISCOVERY,
                }
            ),
        )
    assert exc_info.value.error_code == "SHARED_CAPABILITY_EXPANSION_FORBIDDEN"


def test_personal_missing_policy_fails() -> None:
    with pytest.raises(ConversationExecutionContextError) as exc_info:
        build_conversation_execution_context(resolved=_resolved())
    assert exc_info.value.error_code == "PERSONAL_CAPABILITY_POLICY_MISSING"


def test_personal_explicit_policy_preserved_exactly() -> None:
    capabilities = frozenset(
        {
            ConversationProductCapability.READ_ONLY_ASK,
            ConversationProductCapability.WORKSPACE_DISCOVERY,
        }
    )
    context = build_conversation_execution_context(
        resolved=_resolved(),
        personal_allowed_capabilities=capabilities,
    )
    assert context.allowed_product_capabilities == capabilities


def test_builder_does_not_invoke_external_services() -> None:
    context = build_conversation_execution_context(
        resolved=_resolved(),
        personal_allowed_capabilities=frozenset({ConversationProductCapability.READ_ONLY_ASK}),
    )
    assert context.tenant_id == _TENANT
