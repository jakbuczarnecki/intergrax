# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UAEP adapter: project ``RuntimeExecutionContext`` into tool invocation wiring."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.invocation_wiring import (
    ToolInvocationContext,
    ToolInvocationWiringResolver,
    ToolWiringOverlay,
)
from intergrax.tools.registry.wiring import ToolWiringContext


def build_uaep_wiring_overlay(exec_ctx: RuntimeExecutionContext) -> ToolWiringOverlay:
    workspace = exec_ctx.metadata.get("shadow_workspace")
    shadow: ShadowWorkspace | None = workspace if isinstance(workspace, ShadowWorkspace) else None
    trace_reader: RunTraceReader | None = None
    for key in ("trace_reader", "trace_store"):
        candidate = exec_ctx.metadata.get(key)
        if isinstance(candidate, RunTraceReader):
            trace_reader = candidate
            break
    run_budget = exec_ctx.metadata.get("run_budget")
    budget: RunBudget | None = run_budget if isinstance(run_budget, RunBudget) else None
    cost_envelopes = exec_ctx.metadata.get("cost_envelopes", ())
    cost_quotas = exec_ctx.metadata.get("cost_quotas", ())
    sandbox_session = exec_ctx.metadata.get("sandbox_session")
    task_metadata: dict[str, str] | None = None
    request = exec_ctx.request
    if request is not None and request.metadata:
        task_metadata = {str(k): str(v) for k, v in request.metadata.items()}
    return ToolWiringOverlay(
        shadow_workspace=shadow,
        memory_view=exec_ctx.memory_view,
        trace_reader=trace_reader,
        run_budget=budget,
        cost_envelopes=tuple(cost_envelopes) if isinstance(cost_envelopes, (list, tuple)) else (),
        cost_quotas=tuple(cost_quotas) if isinstance(cost_quotas, (list, tuple)) else (),
        sandbox_session=sandbox_session,
        task_metadata=task_metadata,
    )


class UAEPToolInvocationWiringResolver:
    """Composition-owned UAEP resolver (not imported by shared Tool Engine contracts)."""

    __slots__ = ("_exec_ctx",)

    def __init__(self, exec_ctx: RuntimeExecutionContext) -> None:
        self._exec_ctx = exec_ctx

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolWiringContext,
    ) -> ToolWiringOverlay:
        return build_uaep_wiring_overlay(self._exec_ctx)
