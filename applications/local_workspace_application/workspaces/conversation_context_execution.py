# © Artur Czarnecki. All rights reserved.

"""Deterministic Conversation Execution Context builder (LKW-CONVERSATION-CONTEXT-1B1)."""

from __future__ import annotations

from local_workspace_application.workspaces.conversation_context_models import (
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ResolvedConversationWorkspaceContextV1,
)

_SHARED_CAPABILITIES = frozenset({ConversationProductCapability.READ_ONLY_ASK})

_MUTATION_CAPABILITIES = frozenset(
    {
        ConversationProductCapability.WORKSPACE_DISCOVERY,
        ConversationProductCapability.WORKSPACE_SELECTION,
        ConversationProductCapability.WORKSPACE_ADMINISTRATION,
        ConversationProductCapability.SOURCE_DISCOVERY,
        ConversationProductCapability.SOURCE_INTAKE,
        ConversationProductCapability.ATTACHMENT_INTAKE,
    }
)


class ConversationExecutionContextError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def build_conversation_execution_context(
    *,
    resolved: ResolvedConversationWorkspaceContextV1,
    personal_allowed_capabilities: frozenset[ConversationProductCapability] | None = None,
) -> ConversationExecutionContextV1:
    """Build a deterministic execution context from a resolved workspace context."""
    if resolved.audience_mode is ConversationAudienceMode.SHARED:
        if personal_allowed_capabilities is not None:
            if personal_allowed_capabilities != _SHARED_CAPABILITIES:
                raise ConversationExecutionContextError("SHARED_CAPABILITY_EXPANSION_FORBIDDEN")
            if personal_allowed_capabilities & _MUTATION_CAPABILITIES:
                raise ConversationExecutionContextError("SHARED_CAPABILITY_EXPANSION_FORBIDDEN")
        allowed = _SHARED_CAPABILITIES
    else:
        if not personal_allowed_capabilities:
            raise ConversationExecutionContextError("PERSONAL_CAPABILITY_POLICY_MISSING")
        allowed = frozenset(personal_allowed_capabilities)

    return ConversationExecutionContextV1(
        tenant_id=resolved.tenant_id,
        conversation_context_binding_id=resolved.conversation_context_binding_id,
        audience_mode=resolved.audience_mode,
        workspace_id=resolved.workspace_id,
        principal_ref=resolved.principal_ref,
        canonical_thread_ref=resolved.canonical_thread_ref,
        activation_policy=resolved.activation_policy,
        thread_context_policy=resolved.thread_context_policy,
        allowed_product_capabilities=allowed,
    )
