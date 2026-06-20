# © Artur Czarnecki. All rights reserved.

"""Build and emit unsigned harness-step boundary events from ``HarnessKernel``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from intergrax.contracts.agent_run_trace import AgentStepStatus, PolicyCheckPhase
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.step_execution import StepExecutionRecord
from intergrax.runtime.attestation.boundary_emitter import _runtime_version
from intergrax.runtime.attestation.canonical_json import stable_payload_hash
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.execution_boundary_event import (
    ExecutionBoundaryEventV1,
    ExecutionBoundaryLineageV1,
    ExecutionBoundaryRuntimeRefV1,
    HarnessBoundaryPolicyVerdictV1,
    HarnessBoundaryStepOutcomeV1,
)
from intergrax.runtime.attestation.settings import ExecutionBoundaryExportRuntimeSettings
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.runtime.kernel.step_kernel import StepKernelContext

_LOG = logging.getLogger(__name__)


def _harness_action_status(record: StepExecutionRecord) -> str:
    if record.error_code is not None:
        if record.policy_pre is not None and record.policy_pre.action == PolicyAction.DENY:
            return "denied"
        if record.policy_post is not None and record.policy_post.action == PolicyAction.DENY:
            return "denied"
        return "failed"
    step_record = record.step_record
    if step_record is None:
        return "completed"
    if step_record.status == AgentStepStatus.PAUSED:
        return "paused"
    if step_record.status == AgentStepStatus.FAILED:
        return "failed"
    return "completed"


def _step_outcome(record: StepExecutionRecord) -> HarnessBoundaryStepOutcomeV1:
    step_record = record.step_record
    status = _harness_action_status(record)
    next_action = ""
    error_code: str | None = None
    if step_record is not None:
        next_action = step_record.next_action.value
        if step_record.error_code is not None:
            error_code = step_record.error_code.value
    elif record.error_code is not None:
        error_code = record.error_code.value
    return HarnessBoundaryStepOutcomeV1(
        status=status,  # type: ignore[arg-type]
        next_action=next_action,
        error_code=error_code,
        outcome_applied=record.outcome_applied,
    )


def _policy_verdicts(record: StepExecutionRecord) -> list[HarnessBoundaryPolicyVerdictV1]:
    step_record = record.step_record
    if step_record is not None and step_record.policy_verdicts:
        return [
            HarnessBoundaryPolicyVerdictV1(
                phase=verdict.phase.value,  # type: ignore[arg-type]
                action=verdict.action.value,
                reason=verdict.reason,
                policy_rule_id=verdict.policy_rule_id,
            )
            for verdict in step_record.policy_verdicts
        ]
    verdicts: list[HarnessBoundaryPolicyVerdictV1] = []
    if record.policy_pre is not None:
        verdicts.append(
            HarnessBoundaryPolicyVerdictV1(
                phase=PolicyCheckPhase.PRE.value,  # type: ignore[arg-type]
                action=record.policy_pre.action.value,
                reason=record.policy_pre.reason,
                policy_rule_id=record.policy_pre.policy_rule_id,
            )
        )
    if record.policy_post is not None:
        verdicts.append(
            HarnessBoundaryPolicyVerdictV1(
                phase=PolicyCheckPhase.POST.value,  # type: ignore[arg-type]
                action=record.policy_post.action.value,
                reason=record.policy_post.reason,
                policy_rule_id=record.policy_post.policy_rule_id,
            )
        )
    return verdicts


class HarnessBoundaryEmitter:
    """Non-blocking emitter — sink failures must not fail kernel step execution."""

    @staticmethod
    def maybe_emit(
        *,
        kernel_ctx: StepKernelContext,
        step_ctx: AgentStepContext,
        record: StepExecutionRecord,
    ) -> None:
        export_settings: ExecutionBoundaryExportRuntimeSettings | None = (
            kernel_ctx.execution_boundary_export
        )
        if export_settings is None or not export_settings.enabled:
            return
        if not export_settings.step_level_enabled:
            return

        event = HarnessBoundaryEmitter._build_event(
            kernel_ctx=kernel_ctx,
            step_ctx=step_ctx,
            record=record,
            export_settings=export_settings,
        )
        HarnessBoundaryEmitter._persist(kernel_ctx, event)

    @staticmethod
    def _build_event(
        *,
        kernel_ctx: StepKernelContext,
        step_ctx: AgentStepContext,
        record: StepExecutionRecord,
        export_settings: ExecutionBoundaryExportRuntimeSettings,
    ) -> ExecutionBoundaryEventV1:
        step_id = str(step_ctx.metadata.get("step_id") or "")
        if not step_id and record.step_record is not None and record.step_record.step_id:
            step_id = record.step_record.step_id
        if not step_id:
            step_id = f"step-{step_ctx.step_index:04d}"
        run_id = kernel_ctx.run_id
        input_payload = {
            "step_index": step_ctx.step_index,
            "step_id": step_id,
        }
        step_outcome = _step_outcome(record)
        output_payload = step_outcome.model_dump(mode="json")
        policy_verdicts = _policy_verdicts(record)
        action_status = _harness_action_status(record)
        error_message: str | None = None
        if action_status in {"failed", "denied"} and step_outcome.error_code:
            error_message = step_outcome.error_code

        input_hash = stable_payload_hash(input_payload)
        output_hash = stable_payload_hash(output_payload)
        if not export_settings.include_canonical_io:
            input_payload = {}
            output_payload = {}

        return ExecutionBoundaryEventV1(
            event_id=str(uuid4()),
            event_sequence=0,
            boundary_type="harness_step",
            agent_id=kernel_ctx.agent_id,
            run_id=run_id,
            step_id=step_id,
            task_id=kernel_ctx.task_id,
            tenant_id=kernel_ctx.tenant_id,
            action_status=action_status,  # type: ignore[arg-type]
            input=input_payload,
            output=output_payload,
            input_hash=input_hash,
            output_hash=output_hash,
            error_message=error_message,
            policy_verdicts=policy_verdicts,
            step_outcome=step_outcome,
            occurred_at=SystemTimeProvider.utc_now().isoformat(),
            lineage=ExecutionBoundaryLineageV1(
                ref=f"{run_id}:{step_id}:harness_step",
            ),
            runtime_ref=ExecutionBoundaryRuntimeRefV1(runtime_version=_runtime_version()),
        )

    @staticmethod
    def _persist(kernel_ctx: StepKernelContext, event: ExecutionBoundaryEventV1) -> None:
        buffer: BoundaryEventBuffer | None = kernel_ctx.boundary_event_buffer
        if buffer is None:
            return
        try:
            buffer.append(kernel_ctx.run_id, event)
        except Exception:
            _LOG.exception(
                "harness_boundary_export_buffer_failed run_id=%s step_id=%s",
                kernel_ctx.run_id,
                event.step_id,
            )
