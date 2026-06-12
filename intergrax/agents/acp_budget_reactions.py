# © Artur Czarnecki. All rights reserved.

"""Budget reaction policies and runtime events (§25.5.3 · ACP-TOK-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from intergrax.contracts.acp_budget_enforcement import BudgetScope, HardBudgetViolation
from intergrax.contracts.agent_budget import (
    BudgetExceededReaction,
    BudgetNotifyChannel,
    BudgetReactionProfile,
)
from intergrax.contracts.budget_reaction_hook import BudgetReactionHook, CustomBudgetReactionHook
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.models import NotificationMessage

if TYPE_CHECKING:
    from intergrax.agents.authoring.step_outcome import StepOutcome
    from intergrax.contracts.agent_step_context import AgentStepContext
    from intergrax.runtime.kernel.step_kernel import StepKernelContext


def resolve_exceeded_reaction(
    violation: HardBudgetViolation,
    profile: BudgetReactionProfile | None,
) -> BudgetExceededReaction:
    if profile is None:
        return BudgetExceededReaction.ABORT
    if violation.scope == BudgetScope.AGENT:
        return profile.on_agent_limit_exceeded
    return profile.on_environment_limit_exceeded


def _budget_event_payload(
    violation: HardBudgetViolation,
    *,
    reaction: BudgetExceededReaction,
) -> dict[str, Any]:
    return {
        "scope": violation.scope.value,
        "tokens_total": violation.tokens_total,
        "tokens_limit": violation.tokens_limit,
        "limit_source": violation.limit_source,
        "reaction": reaction.value,
    }


def _threshold_payload(
    *,
    scope: BudgetScope,
    tokens_total: int,
    tokens_limit: int,
    ratio: float,
    limit_source: str,
) -> dict[str, Any]:
    return {
        "scope": scope.value,
        "tokens_total": tokens_total,
        "tokens_limit": tokens_limit,
        "ratio": ratio,
        "limit_source": limit_source,
    }


async def _notify_budget_event(
    kernel_ctx: StepKernelContext,  # noqa: F821
    profile: BudgetReactionProfile,
    *,
    subject: str,
    body: str,
    metadata: dict[str, Any],
) -> None:
    adapter = kernel_ctx.notification_adapter
    if adapter is None or not isinstance(adapter, NotificationAdapter):
        return
    for channel in profile.notify_channels:
        if channel == BudgetNotifyChannel.TRACE_ONLY:
            continue
        message = NotificationMessage(
            channel=channel.value,
            subject=subject,
            body=body,
            task_id=kernel_ctx.task_id or kernel_ctx.run_id,
            tenant_id=kernel_ctx.tenant_id,
            metadata=metadata,
        )
        await adapter.notify(message)


async def handle_hard_budget_violation(
    violation: HardBudgetViolation,
    step_ctx: AgentStepContext,  # noqa: F821
    kernel_ctx: StepKernelContext,  # noqa: F821
):
    """Apply ``BudgetReactionProfile`` and emit ``BUDGET_EXCEEDED``."""
    from intergrax.agents.authoring.step_outcome import StepOutcome
    from intergrax.contracts.agent_run import AgentRunError
    from intergrax.contracts.agent_run_enums import AgentRunErrorCode, TerminalReason
    from intergrax.runtime.kernel.step_kernel import HarnessKernel

    _ = step_ctx
    profile = kernel_ctx.budget_reaction
    reaction = resolve_exceeded_reaction(violation, profile)
    payload = _budget_event_payload(violation, reaction=reaction)
    await HarnessKernel.emit_runtime_event(
        kernel_ctx,
        RuntimeEventType.BUDGET_EXCEEDED,
        payload,
    )
    if profile is not None:
        body = profile.user_message_template or (
            f"Budget exceeded ({violation.scope.value}): "
            f"{violation.tokens_total}/{violation.tokens_limit}"
        )
        await _notify_budget_event(
            kernel_ctx,
            profile,
            subject="budget_exceeded",
            body=body,
            metadata=payload,
        )
    hook = kernel_ctx.budget_reaction_hook
    if hook is not None and isinstance(hook, BudgetReactionHook):
        await hook.on_budget_exceeded(payload)
    elif (
        hook is not None
        and profile is not None
        and reaction == BudgetExceededReaction.CUSTOM_HOOK
        and profile.custom_hook_id is not None
        and isinstance(hook, CustomBudgetReactionHook)
    ):
        await hook.on_custom_budget_hook(profile.custom_hook_id, payload)

    if reaction == BudgetExceededReaction.HITL:
        return StepOutcome.pause_hitl(
            f"budget exceeded ({violation.scope.value})",
            governance_snapshot=payload,
        )
    if reaction == BudgetExceededReaction.DEGRADE_MODEL:
        kernel_ctx.budget_degrade_active = True
    if reaction == BudgetExceededReaction.NOTIFY_ONLY:
        return StepOutcome.fail(
            [
                AgentRunError(
                    code=AgentRunErrorCode.BUDGET_EXCEEDED,
                    message=(
                        f"{violation.scope.value} token budget exceeded (notify_only): "
                        f"{violation.tokens_total}/{violation.tokens_limit}"
                    ),
                    details=payload,
                )
            ],
            terminal_reason=TerminalReason.BUDGET_EXCEEDED,
        )
    return StepOutcome.fail(
        [
            AgentRunError(
                code=AgentRunErrorCode.BUDGET_EXCEEDED,
                message=(
                    f"{violation.scope.value} token budget exceeded: "
                    f"{violation.tokens_total}/{violation.tokens_limit}"
                ),
                details=payload,
            )
        ],
        terminal_reason=TerminalReason.BUDGET_EXCEEDED,
    )


async def maybe_emit_budget_threshold(
    step_ctx: AgentStepContext,  # noqa: F821
    kernel_ctx: StepKernelContext,  # noqa: F821
) -> None:
    """Emit ``BUDGET_THRESHOLD`` once per scope when warn ratio is crossed."""
    from intergrax.runtime.kernel.step_kernel import HarnessKernel

    usage = step_ctx.invocation_usage
    profile = kernel_ctx.budget_reaction
    limits = kernel_ctx.resolved_budget_limits
    if usage is None or profile is None:
        return

    scopes: list[tuple[BudgetScope, int, int | None]] = [
        (BudgetScope.AGENT, usage.agent.tokens_total, limits.agent_tokens_limit),
        (
            BudgetScope.ENVIRONMENT,
            usage.environment.tokens_total,
            limits.environment_tokens_limit,
        ),
    ]
    for scope, tokens_total, tokens_limit in scopes:
        if tokens_limit is None or tokens_limit <= 0:
            continue
        ratio = tokens_total / tokens_limit
        if ratio < profile.warn_threshold_ratio:
            continue
        key = f"{scope.value}:threshold"
        if key in kernel_ctx.budget_threshold_emitted:
            continue
        kernel_ctx.budget_threshold_emitted.add(key)
        payload = _threshold_payload(
            scope=scope,
            tokens_total=tokens_total,
            tokens_limit=tokens_limit,
            ratio=ratio,
            limit_source=limits.limit_source,
        )
        await HarnessKernel.emit_runtime_event(
            kernel_ctx,
            RuntimeEventType.BUDGET_THRESHOLD,
            payload,
        )
        hook = kernel_ctx.budget_reaction_hook
        if hook is not None and isinstance(hook, BudgetReactionHook):
            await hook.on_budget_threshold(payload)
        exceeded_reaction = (
            profile.on_agent_limit_exceeded
            if scope == BudgetScope.AGENT
            else profile.on_environment_limit_exceeded
        )
        if exceeded_reaction == BudgetExceededReaction.DEGRADE_MODEL:
            kernel_ctx.budget_degrade_active = True
