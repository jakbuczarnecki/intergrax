# © Artur Czarnecki. All rights reserved.

"""Graph-node helpers for DefaultNexusContextEngine (CE-3.7)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
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
    return ContextAssemblyRequest(
        trace_id=trace_id or task.task_id,
        run_id=task.task_id,
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


def text_from_assembled_messages(messages: tuple[ChatMessage, ...]) -> str:
    """Extract final agent-facing text from assembled chat messages."""
    if not messages:
        return ""
    return messages[-1].content or ""
