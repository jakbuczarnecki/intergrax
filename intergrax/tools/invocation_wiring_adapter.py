# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Legacy ``ToolWiringContext`` adapter (private composition seam, not resolver ABI)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.tools.invocation_wiring import (
    ToolInvocationWiring,
    ToolRegistrationWiringView,
)
from intergrax.tools.registry.wiring import ToolWiringContext


def registration_wiring_for_handler(handler: object) -> ToolWiringContext:
    from intergrax.tools.core.handler import WiringContextToolHandler

    if isinstance(handler, WiringContextToolHandler):
        return handler.registration_wiring
    return ToolWiringContext()


def registration_wiring_view_from_context(
    registration: ToolWiringContext,
) -> ToolRegistrationWiringView:
    return ToolRegistrationWiringView(
        workspace=registration.shadow_workspace,
        memory_view=registration.memory_view,
        trace_reader=registration.trace_reader,
        run_budget=registration.run_budget,
        cost_envelopes=tuple(registration.cost_envelopes),
        cost_quotas=tuple(registration.cost_quotas),
        sandbox_session=registration.sandbox_session,
    )


def registration_wiring_view_for_handler(handler: object) -> ToolRegistrationWiringView:
    return registration_wiring_view_from_context(registration_wiring_for_handler(handler))


def merge_invocation_into_handler_context(
    registration: ToolWiringContext,
    invocation: ToolInvocationWiring,
) -> ToolWiringContext:
    """Deterministic merge into handler-visible legacy context."""
    effective = registration
    if invocation.workspace is not None:
        effective = replace(effective, shadow_workspace=invocation.workspace)
    if invocation.memory_view is not None:
        effective = replace(effective, memory_view=invocation.memory_view)
    if invocation.trace_reader is not None:
        effective = replace(effective, trace_reader=invocation.trace_reader)
    if invocation.run_budget is not None:
        effective = replace(effective, run_budget=invocation.run_budget)
    if invocation.cost_envelopes is not None:
        effective = replace(effective, cost_envelopes=invocation.cost_envelopes)
    if invocation.cost_quotas is not None:
        effective = replace(effective, cost_quotas=invocation.cost_quotas)
    if invocation.sandbox_session is not None:
        effective = replace(effective, sandbox_session=invocation.sandbox_session)
        if "effective_environment_profile" not in effective.extras:
            from intergrax.runtime.sandbox.runtime_host_wiring import (
                runtime_host_sandbox_isolation_profile,
            )

            merged_extras = dict(effective.extras)
            merged_extras["effective_environment_profile"] = (
                runtime_host_sandbox_isolation_profile()
            )
            effective = replace(effective, extras=merged_extras)
    if invocation.task_metadata is not None:
        merged_extras = dict(registration.extras)
        merged_extras["task_metadata"] = dict(invocation.task_metadata)
        effective = replace(effective, extras=merged_extras)
    return effective


def merge_invocation_wiring(
    registration: ToolWiringContext,
    invocation: ToolInvocationWiring,
) -> ToolWiringContext:
    return merge_invocation_into_handler_context(registration, invocation)


def effective_wiring_for_request(
    request: object,
    registration: ToolWiringContext,
) -> ToolWiringContext:
    from intergrax.tools.execution_models import ToolExecutionRequest

    if not isinstance(request, ToolExecutionRequest):
        return registration
    if request.effective_wiring is not None:
        return request.effective_wiring
    return registration
