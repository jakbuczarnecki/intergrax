# © Artur Czarnecki. All rights reserved.

"""Compensation enqueue after step failure with committed side effects (ACP-PROD-3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from intergrax.agents.persistence.compensation_queue_store import (
    CompensationJob,
    CompensationQueueStore,
)
from intergrax.agents.persistence.declarative_tool_executor import (
    DeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.side_effect import CompensationRequest, SideEffectRecord
from intergrax.tools.tool_execution_profile import (
    ToolExecutionProfile,
    ToolMutability,
    ToolReversibility,
)


def build_compensation_idempotency_key(original_key: str) -> str:
    return f"comp:{original_key}"


@dataclass(frozen=True)
class CompensationActionResult:
    request: CompensationRequest
    status: Literal["enqueued", "compensated", "failed", "manual_required", "skipped"]
    error: str | None = None


@dataclass
class CompensationEnqueueResult:
    actions: list[CompensationActionResult] = field(default_factory=list)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "compensation_enqueue": [
                {
                    "original_side_effect_id": item.request.original_side_effect_id,
                    "compensation_tool_id": item.request.compensation_tool_id,
                    "status": item.status,
                    "idempotency_key": item.request.idempotency_key,
                    "error": item.error,
                }
                for item in self.actions
            ],
        }


def _compensation_args(
    record: SideEffectRecord,
    *,
    action_args: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    original_args = {}
    if action_args is not None:
        original_args = dict(action_args.get(record.target, {}))
    payload = dict(original_args)
    if record.external_ref is not None:
        payload["original_external_ref"] = record.external_ref
    payload["original_side_effect_id"] = record.side_effect_id
    return payload


def _persist_enqueued_compensation(
    *,
    compensation_queue: CompensationQueueStore,
    request: CompensationRequest,
    run_id: str,
    tenant_id: str,
    agent_id: str,
    step_index: int,
) -> None:
    if compensation_queue.get_by_idempotency_key(tenant_id, request.idempotency_key) is not None:
        return
    compensation_queue.enqueue(
        CompensationJob(
            run_id=run_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            step_index=step_index,
            request=request,
        ),
    )


async def enqueue_compensations_for_step_failure(
    *,
    ledger: SideEffectLedger | None,
    tool_profiles: dict[str, ToolExecutionProfile],
    step_index: int,
    invoker: DeclarativeToolInvoker | None = None,
    action_args: dict[str, dict[str, Any]] | None = None,
    compensation_queue: CompensationQueueStore | None = None,
    run_id: str = "",
    tenant_id: str = "default",
    agent_id: str = "",
) -> CompensationEnqueueResult:
    """
    Apply §40.3.2 compensation policy for committed effects in ``step_index``.

    * ``compensatable`` + registered compensation tool → invoke (when invoker set) or enqueue.
    * ``manual`` → mark side effect failed; operator HITL follow-up.
    * ``read_only`` / ``none`` → no action.
    """
    result = CompensationEnqueueResult()
    if ledger is None:
        return result

    for record in ledger.committed_for_step(step_index):
        profile = tool_profiles.get(record.target)
        if profile is None:
            profile = ToolExecutionProfile(tool_id=record.target)
        if profile.mutability == ToolMutability.READ_ONLY:
            continue
        if profile.reversibility == ToolReversibility.NONE:
            continue

        if profile.reversibility == ToolReversibility.MANUAL:
            ledger.mark_failed(record.idempotency_key)
            result.actions.append(
                CompensationActionResult(
                    request=CompensationRequest(
                        original_side_effect_id=record.side_effect_id,
                        compensation_tool_id=record.target,
                        args=_compensation_args(record, action_args=action_args),
                        idempotency_key=build_compensation_idempotency_key(record.idempotency_key),
                    ),
                    status="manual_required",
                )
            )
            continue

        compensation_tool_id = profile.compensation_tool_id
        if not compensation_tool_id:
            ledger.mark_failed(record.idempotency_key)
            result.actions.append(
                CompensationActionResult(
                    request=CompensationRequest(
                        original_side_effect_id=record.side_effect_id,
                        compensation_tool_id=record.target,
                        args=_compensation_args(record, action_args=action_args),
                        idempotency_key=build_compensation_idempotency_key(record.idempotency_key),
                    ),
                    status="skipped",
                    error="compensation_tool_not_registered",
                )
            )
            continue

        request = CompensationRequest(
            original_side_effect_id=record.side_effect_id,
            compensation_tool_id=compensation_tool_id,
            args=_compensation_args(record, action_args=action_args),
            idempotency_key=build_compensation_idempotency_key(record.idempotency_key),
        )

        if invoker is None:
            if compensation_queue is not None:
                _persist_enqueued_compensation(
                    compensation_queue=compensation_queue,
                    request=request,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    step_index=step_index,
                )
            result.actions.append(
                CompensationActionResult(request=request, status="enqueued"),
            )
            continue

        invoke_result = await invoker.invoke(
            tool_id=compensation_tool_id,
            args=request.args,
            idempotency_key=request.idempotency_key,
        )
        if invoke_result.status == "success":
            ledger.mark_compensated(record.idempotency_key)
            result.actions.append(
                CompensationActionResult(request=request, status="compensated"),
            )
            continue

        ledger.mark_failed(record.idempotency_key)
        result.actions.append(
            CompensationActionResult(
                request=request,
                status="failed",
                error=invoke_result.error or invoke_result.status,
            ),
        )
    return result
