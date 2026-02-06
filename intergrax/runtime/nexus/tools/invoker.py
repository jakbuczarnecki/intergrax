# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Protocol, Type, runtime_checkable

from pydantic import BaseModel, ValidationError

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.tools.tool_invocation import ToolInvocationEndDiagV1, ToolInvocationErrorDiagV1, ToolInvocationStartDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.tools.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor


@runtime_checkable
class TraceEmitter(Protocol):
    def trace_event(
        self,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel,
        payload: Optional[object] = None,
        artifact_refs: Optional[list] = None,
    ) -> None: ...


class RuntimeToolInvoker:
    """
    Nexus-owned enforcement wrapper for tool invocation.

    Enforces:
    - tool registry presence
    - input schema type
    - output schema validation
    - error mapping -> RuntimeErrorCode
    - trace start/end/error
    """

    def __init__(self, *, registry: ToolRegistry, executor: ToolExecutor) -> None:
        self._registry = registry
        self._executor = executor

    def invoke(
        self,
        *,
        state: TraceEmitter,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolExecutionResult[BaseModel]:
        # 1) registry check + contract bind
        try:
            reg = self._registry.get(request.tool_id)
        except KeyError as exc:
            msg = str(exc)
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_invocation_error",
                message="Tool not registered.",
                level=TraceLevel.ERROR,
                payload=ToolInvocationErrorDiagV1(
                    tool_id=request.tool_id,
                    step_id=str(request.step_id),
                    error_code=RuntimeErrorCode.TOOL_ERROR,
                    error_message=msg,
                ),
            )
            return ToolExecutionResult.fail(RuntimeErrorCode.TOOL_ERROR, msg)

        contract = reg.contract

        # 2) input type enforcement
        if not isinstance(request.input, contract.input_schema):
            msg = (
                f"Tool input must be {contract.input_schema.__name__} "
                f"(got {type(request.input).__name__})."
            )
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_invocation_error",
                message="Tool input validation failed.",
                level=TraceLevel.ERROR,
                payload=ToolInvocationErrorDiagV1(
                    tool_id=request.tool_id,
                    step_id=str(request.step_id),
                    error_code=RuntimeErrorCode.VALIDATION_ERROR,
                    error_message=msg,
                ),
            )
            return ToolExecutionResult.fail(RuntimeErrorCode.VALIDATION_ERROR, msg)

        # 3) trace start
        state.trace_event(
            component=TraceComponent.TOOLS,
            step="tool_invocation_start",
            message="Tool invocation started.",
            level=TraceLevel.INFO,
            payload=ToolInvocationStartDiagV1(
                tool_id=contract.tool_id,
                step_id=str(request.step_id),
                side_effects=contract.side_effects,
                input_payload=request.input.model_dump(),
            ),
        )

        # 4) execute + normalize
        try:
            raw_out = self._executor.execute(request)

            # output enforcement (typed)
            out = self._validate_output(contract.output_schema, raw_out)

            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_invocation_end",
                message="Tool invocation finished.",
                level=TraceLevel.INFO,
                payload=ToolInvocationEndDiagV1(
                    tool_id=contract.tool_id,
                    step_id=str(request.step_id),
                    success=True,
                    output_preview=self._preview_output(out),
                ),
            )
            return ToolExecutionResult.ok(out)

        except Exception as exc:
            code = self._map_error(contract, exc)
            msg = f"{type(exc).__name__}: {exc}"

            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_invocation_error",
                message="Tool invocation failed.",
                level=TraceLevel.ERROR,
                payload=ToolInvocationErrorDiagV1(
                    tool_id=contract.tool_id,
                    step_id=str(request.step_id),
                    error_code=code,
                    error_message=msg,
                ),
            )
            return ToolExecutionResult.fail(code, msg)

    @staticmethod
    def _map_error(contract: ToolContract, exc: Exception) -> RuntimeErrorCode:
        # contract.error_mapping: Mapping[type[Exception], RuntimeErrorCode]
        for exc_type, code in contract.error_mapping.items():
            if isinstance(exc, exc_type):
                return code
        return RuntimeErrorCode.TOOL_ERROR


    @staticmethod
    def _validate_output(schema: Type[BaseModel], raw_out: BaseModel) -> BaseModel:
        """
        Strict runtime boundary check.

        Tool handler MUST return exactly the declared output_schema type.
        """
        if not isinstance(raw_out, schema):
            raise TypeError(
                f"Tool returned invalid output type. "
                f"Expected {schema.__name__}, got {type(raw_out).__name__}."
            )
        return raw_out


    @staticmethod
    def _preview_output(out: BaseModel, *, limit: int = 300) -> str:
        s = out.model_dump_json()
        if len(s) <= limit:
            return s
        return s[:limit] + "…"
