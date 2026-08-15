# © Artur Czarnecki. All rights reserved.

"""Graph-node helpers for DefaultNexusContextEngine (CE-3.7)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.llm.messages import (
    ChatMessage,
    StructuredModelInputRequiredError,
    requires_structured_model_input,
)
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task


def build_graph_assembly_request(
    task: Task,
    node: ExecutionNode,
    *,
    policy: TaskContextAssemblyOptions,
    budget_policy: ContextBudgetPolicy,
    trace_id: str = "",
) -> ContextAssemblyRequest:
    """Build a CE assembly request for one execution-graph node."""
    active_run_id, _ = require_active_execution_identity()
    return ContextAssemblyRequest(
        trace_id=trace_id or task.task_id,
        run_id=active_run_id,
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        assembly_scope="graph_node",
        objective=task.message,
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(
            max_chars=budget_policy.max_chars,
            max_tokens_estimate=max(budget_policy.max_chars // 4, 256),
            summary_tier=policy.summary_tier,
        ),
        assembly_options=policy,
        graph_node_id=node.node_id,
        step_kind=node.capability,
    )


def graph_messages_from_text(message: str) -> list[ChatMessage]:
    """Wrap composed graph context as a single user turn for the compiler spine."""
    return [ChatMessage(role="user", content=message)]


def compatibility_text_from_assembled_messages(messages: tuple[ChatMessage, ...]) -> str:
    """UI/debug/legacy compatibility projection — context blocks plus final user turn."""
    if not messages:
        return ""
    blocks: list[str] = []
    final_user_content: str | None = None
    for index, message in enumerate(messages):
        content = message.content or ""
        if message.role == "system" and content.startswith("[context:"):
            blocks.append(content)
        elif index == len(messages) - 1 and message.role == "user":
            final_user_content = content
    parts = list(blocks)
    if final_user_content is not None:
        parts.append(final_user_content)
    return "\n\n".join(parts)


def text_from_assembled_messages(messages: tuple[ChatMessage, ...]) -> str:
    """Strict text projection — fails when structured history cannot be losslessly flattened."""
    if requires_structured_model_input(messages):
        raise StructuredModelInputRequiredError()
    return compatibility_text_from_assembled_messages(messages)
