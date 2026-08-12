# © Artur Czarnecki. All rights reserved.

"""Typed thread-memory policy for conversation interaction persistence."""

from __future__ import annotations

from local_workspace_application.conversation.interaction_execution_models import (
    ConversationInteractionExecutionResult,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
)
from local_workspace_application.workspaces.tenant_connection_conversation_models import (
    THREAD_MEMORY_CREDENTIAL_REDACTION,
)

MANUAL_CREDENTIAL_COMPLETE_ACTION = "tenant_connection.authorization.complete_manual"


def interaction_plan_requires_credential_memory_redaction(
    plan: ConversationInteractionPlan,
) -> bool:
    return any(
        action.action_type == MANUAL_CREDENTIAL_COMPLETE_ACTION
        for action in plan.actions
    )


def resolve_durable_thread_memory_user_text(
    *,
    plan: ConversationInteractionPlan | None,
    message_text: str,
    execution_result: ConversationInteractionExecutionResult | None = None,
) -> str:
    if plan is not None and interaction_plan_requires_credential_memory_redaction(plan):
        return THREAD_MEMORY_CREDENTIAL_REDACTION
    if (
        execution_result is not None
        and execution_result.thread_memory_user_text is not None
    ):
        return execution_result.thread_memory_user_text
    return message_text.strip()


__all__ = [
    "MANUAL_CREDENTIAL_COMPLETE_ACTION",
    "interaction_plan_requires_credential_memory_redaction",
    "resolve_durable_thread_memory_user_text",
]
