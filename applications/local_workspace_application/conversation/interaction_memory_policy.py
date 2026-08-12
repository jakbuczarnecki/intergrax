# © Artur Czarnecki. All rights reserved.

"""Typed thread-memory policy for conversation interaction persistence."""

from __future__ import annotations

from enum import StrEnum

from local_workspace_application.conversation.interaction_execution_models import (
    ConversationInteractionExecutionResult,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
)
from local_workspace_application.workspaces.conversation_connection_auth_context_service import (
    ConversationConnectionAuthContextError,
    ConversationConnectionAuthContextService,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationExecutionContextV1,
)
from local_workspace_application.workspaces.tenant_connection_conversation_models import (
    THREAD_MEMORY_CREDENTIAL_REDACTION,
)

MANUAL_CREDENTIAL_COMPLETE_ACTION = "tenant_connection.authorization.complete_manual"


class ConversationDurableMemoryPolicyV1(StrEnum):
    NORMAL = "normal"
    REDACT_MANUAL_CREDENTIAL_INPUT = "redact_manual_credential_input"


def resolve_interaction_durable_memory_policy(
    *,
    context: ConversationExecutionContextV1,
    connection_auth_context_service: ConversationConnectionAuthContextService | None,
) -> ConversationDurableMemoryPolicyV1:
    if connection_auth_context_service is None:
        return ConversationDurableMemoryPolicyV1.NORMAL
    try:
        connection_auth_context_service.require_pending_manual_authorization(
            context=context,
        )
    except ConversationConnectionAuthContextError as exc:
        if exc.error_code == "conversation_context_storage_unavailable":
            return ConversationDurableMemoryPolicyV1.REDACT_MANUAL_CREDENTIAL_INPUT
        return ConversationDurableMemoryPolicyV1.NORMAL
    return ConversationDurableMemoryPolicyV1.REDACT_MANUAL_CREDENTIAL_INPUT


def interaction_plan_requires_credential_memory_redaction(
    plan: ConversationInteractionPlan,
) -> bool:
    return any(
        action.action_type == MANUAL_CREDENTIAL_COMPLETE_ACTION
        for action in plan.actions
    )


def resolve_durable_thread_memory_user_text(
    *,
    memory_policy: ConversationDurableMemoryPolicyV1,
    message_text: str,
    execution_result: ConversationInteractionExecutionResult | None = None,
) -> str:
    if (
        memory_policy
        is ConversationDurableMemoryPolicyV1.REDACT_MANUAL_CREDENTIAL_INPUT
    ):
        return THREAD_MEMORY_CREDENTIAL_REDACTION
    if (
        execution_result is not None
        and execution_result.thread_memory_user_text is not None
    ):
        return execution_result.thread_memory_user_text
    return message_text.strip()


__all__ = [
    "ConversationDurableMemoryPolicyV1",
    "MANUAL_CREDENTIAL_COMPLETE_ACTION",
    "interaction_plan_requires_credential_memory_redaction",
    "resolve_durable_thread_memory_user_text",
    "resolve_interaction_durable_memory_policy",
]
