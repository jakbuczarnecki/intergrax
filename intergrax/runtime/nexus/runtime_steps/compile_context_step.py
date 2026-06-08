# © Artur Czarnecki. All rights reserved.

"""Context Compiler runtime step (Phase MEM-DEPTH-1.1)."""

from __future__ import annotations

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads import ContextAssemblyPayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.nexus.context.context_budget import ContextTrimResult
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class CompileContextStep(RuntimeStep):
    """Apply unified ContextCompiler before core LLM invocation."""

    def execution_kind(self) -> ExecutionKind | None:
        return None

    async def run(self, state: RuntimeState) -> None:
        compiler = ContextCompiler()
        result = compiler.compile(
            list(state.messages_for_llm),
            state.context.config,
            max_output_tokens=state.request.max_output_tokens,
        )
        state.messages_for_llm = result.messages

        bus: RuntimeEventBus | None = state.context.event_bus
        if result.trimmed and bus is not None:
            trim = ContextTrimResult(
                message="",
                trimmed=True,
                original_chars=result.bytes_removed + sum(len(m.content or "") for m in result.messages),
                final_chars=sum(len(m.content or "") for m in result.messages),
            )
            bus.record(
                runtime_event_with_payload(
                    RuntimeEvent(
                        tenant_id=state.context.config.tenant_id,
                        task_id=state.request.session_id or state.run_id,
                        run_id=state.run_id,
                        event_type=RuntimeEventType.CONTEXT_TRIMMED,
                        phase=ExecutionPhase.CONTEXT_BUILDING,
                        correlation_id=state.run_id,
                        payload={"degradation_steps": list(result.degradation_steps)},
                    ),
                    ContextAssemblyPayloadV1(
                        node_id="chat",
                        context_original_chars=trim.original_chars,
                        context_final_chars=trim.final_chars,
                        trimmed=True,
                    ),
                )
            )

        state.trace_event(
            component=TraceComponent.ENGINE,
            step="context_compiler",
            message="Context compiler pass completed.",
            level=TraceLevel.INFO,
            payload={
                "schema_id": "context_compiler_diag.v1",
                "total_tokens": result.total_tokens,
                "budget_tokens": result.budget_tokens,
                "trimmed": result.trimmed,
                "degradation_steps": list(result.degradation_steps),
                "bytes_removed": result.bytes_removed,
            },
        )
