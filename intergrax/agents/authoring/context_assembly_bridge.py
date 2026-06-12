# © Artur Czarnecki. All rights reserved.

"""ACP / UAEP helpers for ``ContextAssemblyRequest`` (CE-4.1, CE-4.2)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragmentSource,
)
from intergrax.contracts.agent_context_hints import AgentContextHints
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions


def _sources_from_hints(hints: AgentContextHints | None) -> tuple[frozenset[ContextFragmentSource], frozenset[ContextFragmentSource]]:
    if hints is None:
        return frozenset(), frozenset()
    required: set[ContextFragmentSource] = set()
    excluded: set[ContextFragmentSource] = set()
    for raw in hints.required_sources:
        try:
            required.add(ContextFragmentSource(raw.strip().lower()))
        except ValueError:
            continue
    for raw in hints.excluded_sources:
        try:
            excluded.add(ContextFragmentSource(raw.strip().lower()))
        except ValueError:
            continue
    return frozenset(required), frozenset(excluded)


def build_acp_assembly_request(
    step_ctx: AgentStepContext,
    *,
    hints: AgentContextHints | None = None,
    assembly_options: TaskContextAssemblyOptions | None = None,
    objective: str = "",
) -> ContextAssemblyRequest:
    """Populate step-aware ``ContextAssemblyRequest`` for one ACP step (CE-4.1)."""
    required, excluded = _sources_from_hints(hints)
    options = assembly_options or TaskContextAssemblyOptions()
    step_kind = hints.step_kind if hints and hints.step_kind else step_ctx.step_kind
    return ContextAssemblyRequest(
        trace_id=step_ctx.run_id or step_ctx.task_id,
        run_id=step_ctx.run_id or step_ctx.task_id,
        task_id=step_ctx.task_id,
        tenant_id=step_ctx.tenant_id,
        assembly_scope="acp_step",
        objective=objective or step_ctx.message,
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_chars=max(options.max_prior_chars, 4000)),
        assembly_options=options,
        step_index=step_ctx.step_index,
        step_kind=step_kind,
        required_sources=required,
        excluded_sources=excluded,
    )
