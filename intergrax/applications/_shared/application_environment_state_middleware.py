# © Artur Czarnecki. All rights reserved.

"""Nexus lifecycle sync for ``ApplicationEnvironmentState`` (APP-CON-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.applications._shared.workspace_cleanup_wiring import sync_isolation_refs_for_hook
from intergrax.applications.contracts.environment_state import (
    ApplicationEnvironmentState,
    EnvironmentHealthStatus,
    EnvironmentTaskPhase,
    HitlEscalationState,
    seed_application_environment_state,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware

if TYPE_CHECKING:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications.contracts.manifest import ApplicationManifest
    from intergrax.runtime.nexus.budget.budget_models import RunBudget

_HOOK_POINT_PHASE: dict[HookPoint, EnvironmentTaskPhase] = {
    HookPoint.BEFORE_TASK_INTAKE: EnvironmentTaskPhase.INTAKE,
    HookPoint.AFTER_TASK_INTAKE: EnvironmentTaskPhase.INTAKE,
    HookPoint.BEFORE_CLASSIFICATION: EnvironmentTaskPhase.CLASSIFICATION,
    HookPoint.AFTER_CLASSIFICATION: EnvironmentTaskPhase.CLASSIFICATION,
    HookPoint.BEFORE_PLANNING: EnvironmentTaskPhase.PLANNING,
    HookPoint.AFTER_PLANNING: EnvironmentTaskPhase.PLANNING,
    HookPoint.BEFORE_AGENT_SELECTION: EnvironmentTaskPhase.AGENT_SELECTION,
    HookPoint.AFTER_AGENT_SELECTION: EnvironmentTaskPhase.AGENT_SELECTION,
    HookPoint.BEFORE_CONTEXT_BUILD: EnvironmentTaskPhase.AGENT_RUN,
    HookPoint.AFTER_CONTEXT_BUILD: EnvironmentTaskPhase.AGENT_RUN,
    HookPoint.BEFORE_LLM_INFERENCE: EnvironmentTaskPhase.AGENT_RUN,
    HookPoint.AFTER_LLM_INFERENCE: EnvironmentTaskPhase.AGENT_RUN,
    HookPoint.BEFORE_LLM_OUTPUT: EnvironmentTaskPhase.AGENT_RUN,
    HookPoint.AFTER_LLM_OUTPUT: EnvironmentTaskPhase.AGENT_RUN,
    HookPoint.BEFORE_STEP: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.AFTER_STEP: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.BEFORE_TOOL_CALL: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.AFTER_TOOL_CALL: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.BEFORE_VALIDATION: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.AFTER_VALIDATION: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.BEFORE_DECISION: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.AFTER_DECISION: EnvironmentTaskPhase.GRAPH_EXECUTION,
    HookPoint.BEFORE_HUMAN_APPROVAL: EnvironmentTaskPhase.HITL,
    HookPoint.AFTER_HUMAN_APPROVAL: EnvironmentTaskPhase.HITL,
    HookPoint.BEFORE_FINALIZATION: EnvironmentTaskPhase.FINALIZATION,
    HookPoint.AFTER_FINALIZATION: EnvironmentTaskPhase.FINALIZATION,
}

_HITL_HOOK_POINTS = frozenset(
    {
        HookPoint.BEFORE_HUMAN_APPROVAL,
        HookPoint.AFTER_HUMAN_APPROVAL,
    }
)


def _phase_for_hook(point: HookPoint, ctx: HookContext) -> EnvironmentTaskPhase:
    task_state = ctx.runtime_state.get("task_state")
    if point == HookPoint.AFTER_FINALIZATION:
        if task_state == "completed":
            return EnvironmentTaskPhase.COMPLETED
        if task_state == "failed":
            return EnvironmentTaskPhase.FAILED
    mapped = _HOOK_POINT_PHASE.get(point)
    if mapped is not None:
        return mapped
    existing = ApplicationEnvironmentState.from_runtime_state(ctx.runtime_state)
    if existing is not None:
        return existing.phase
    return EnvironmentTaskPhase.INTAKE


def _graph_id_from_context(ctx: HookContext) -> str | None:
    for key in ("graph_id", "plan_id"):
        value = ctx.runtime_state.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def sync_application_environment_state_for_hook(
    point: HookPoint,
    ctx: HookContext,
    *,
    app_id: str,
    profile_id: str,
    execution_mode: ExecutionMode,
    run_budget: RunBudget | None = None,
) -> ApplicationEnvironmentState:
    """Load, seed, and update host-visible environment state for one lifecycle hook."""
    state = ApplicationEnvironmentState.from_runtime_state(ctx.runtime_state)
    if state is None:
        seeded = seed_application_environment_state(
            app_id=app_id,
            profile_id=profile_id,
            execution_mode=execution_mode,
            task_id=ctx.task_id,
            organization_id=_organization_id(ctx),
            profile_snapshot_id=profile_id,
        )
        state = ApplicationEnvironmentState.from_runtime_state(seeded)
    assert state is not None

    phase = _phase_for_hook(point, ctx)
    graph_id = _graph_id_from_context(ctx) or state.graph_id
    health = state.health
    hitl = state.hitl
    budget = state.budget

    if point in _HITL_HOOK_POINTS:
        if point == HookPoint.BEFORE_HUMAN_APPROVAL:
            ticket_id = ctx.runtime_state.get("hitl_ticket_id")
            hitl = HitlEscalationState(
                pending=True,
                ticket_id=ticket_id if isinstance(ticket_id, str) else None,
                escalation_reason=(
                    ctx.runtime_state.get("hitl_reason")
                    if isinstance(ctx.runtime_state.get("hitl_reason"), str)
                    else None
                ),
            )
            health = EnvironmentHealthStatus.HITL_PENDING
        else:
            hitl = HitlEscalationState()
            if health == EnvironmentHealthStatus.HITL_PENDING:
                health = EnvironmentHealthStatus.HEALTHY

    if run_budget is not None and run_budget.max_total_tokens is not None:
        budget = budget.model_copy(
            update={"environment_tokens_limit": run_budget.max_total_tokens}
        )

    state = state.model_copy(
        update={
            "task_id": ctx.task_id,
            "run_id": ctx.run_id,
            "graph_id": graph_id,
            "phase": phase,
            "health": health,
            "hitl": hitl,
            "budget": budget,
        }
    )
    state = sync_isolation_refs_for_hook(ctx, state)
    ctx.runtime_state.update(state.apply_to_runtime_state(dict(ctx.runtime_state)))
    return state


def _organization_id(ctx: HookContext) -> str | None:
    value = ctx.runtime_state.get("organization_id")
    return value if isinstance(value, str) else None


class ApplicationEnvironmentStateMiddleware(RuntimeMiddleware):
    """Keeps ``app_env_state.v1`` current on Nexus lifecycle hooks (priority 40)."""

    priority = 40
    name = "application_environment_state"

    def __init__(
        self,
        *,
        manifest: ApplicationManifest,
        environment: ApplicationEnvironmentProfile,
        run_budget: RunBudget | None = None,
    ) -> None:
        self._app_id = manifest.app_id
        self._profile_id = environment.profile_id
        self._execution_mode = environment.execution_mode
        self._run_budget = run_budget

    async def before(self, point: HookPoint, context: HookContext) -> HookResult:
        sync_application_environment_state_for_hook(
            point,
            context,
            app_id=self._app_id,
            profile_id=self._profile_id,
            execution_mode=self._execution_mode,
            run_budget=self._run_budget,
        )
        return HookResult()

    async def after(self, point: HookPoint, context: HookContext) -> HookResult:
        sync_application_environment_state_for_hook(
            point,
            context,
            app_id=self._app_id,
            profile_id=self._profile_id,
            execution_mode=self._execution_mode,
            run_budget=self._run_budget,
        )
        return HookResult()
