# © Artur Czarnecki. All rights reserved.

"""Build and emit unsigned execution boundary events from ``RuntimeToolInvoker``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel

from intergrax.runtime.attestation.canonical_json import stable_payload_hash
from intergrax.runtime.attestation.attestation_policy import (
    AttestationCaptureMode,
    should_emit_boundary_event,
)
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.execution_boundary_event import (
    ExecutionBoundaryEventV1,
    ExecutionBoundaryLineageV1,
    ExecutionBoundaryRuntimeRefV1,
)
from intergrax.runtime.attestation.settings import ExecutionBoundaryExportRuntimeSettings
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

_LOG = logging.getLogger(__name__)


def _runtime_version() -> str:
    try:
        from importlib.metadata import version

        return version("intergrax")
    except Exception:
        return "unknown"


class ExecutionBoundaryEmitter:
    """Non-blocking emitter — sink failures must not fail tool invocation."""

    @staticmethod
    def maybe_emit(
        *,
        state: RuntimeState,
        agent_id: str,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        result: ToolExecutionResult[BaseModel],
    ) -> None:
        config = state.context.config
        export_settings: ExecutionBoundaryExportRuntimeSettings | None = (
            config.execution_boundary_export
        )
        if export_settings is None or not export_settings.enabled:
            return
        if not should_emit_boundary_event(
            contract=contract,
            result=result,
            capture_mode=export_settings.capture_mode,
            allowlist=export_settings.allowlist,
        ):
            return

        event = ExecutionBoundaryEmitter._build_event(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
            result=result,
            export_settings=export_settings,
        )
        ExecutionBoundaryEmitter._persist(state, event)

    @staticmethod
    def _build_event(
        *,
        state: RuntimeState,
        agent_id: str,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        result: ToolExecutionResult[BaseModel],
        export_settings: ExecutionBoundaryExportRuntimeSettings,
    ) -> ExecutionBoundaryEventV1:
        step_id = str(request.step_id)
        run_id = state.run_id
        task_id = str(state.request.metadata.get("task_id", state.request.session_id or ""))
        tenant_id = state.request.tenant_id or ""
        input_payload = request.input.model_dump(mode="json")
        output_payload: dict[str, object] = {}
        error_message: str | None = None
        action_status = "executed"
        if result.success and result.output is not None:
            output_payload = result.output.model_dump(mode="json")
        else:
            action_status = "failed"
            if result.error is not None:
                error_message = result.error.error_message

        input_hash = stable_payload_hash(input_payload) if input_payload else None
        output_hash = stable_payload_hash(output_payload) if output_payload else None
        if not export_settings.include_canonical_io:
            input_payload = {}
            output_payload = {}

        return ExecutionBoundaryEventV1(
            event_id=str(uuid4()),
            event_sequence=0,
            boundary_type="tool_execution",
            tool_id=contract.tool_id,
            agent_id=agent_id,
            run_id=run_id,
            step_id=step_id,
            task_id=task_id,
            tenant_id=tenant_id,
            action_status=action_status,
            side_effects=contract.side_effects,
            risk_level=contract.risk_level.value,
            input=input_payload,
            output=output_payload,
            input_hash=input_hash,
            output_hash=output_hash,
            error_message=error_message,
            occurred_at=SystemTimeProvider.utc_now().isoformat(),
            lineage=ExecutionBoundaryLineageV1(ref=f"{run_id}:{step_id}"),
            runtime_ref=ExecutionBoundaryRuntimeRefV1(runtime_version=_runtime_version()),
        )

    @staticmethod
    def _persist(state: RuntimeState, event: ExecutionBoundaryEventV1) -> None:
        buffer: BoundaryEventBuffer | None = state.context.config.boundary_event_buffer
        if buffer is None:
            return
        try:
            buffer.append(state.run_id, event)
        except Exception:
            _LOG.exception(
                "execution_boundary_export_buffer_failed run_id=%s tool_id=%s",
                state.run_id,
                event.tool_id,
            )
