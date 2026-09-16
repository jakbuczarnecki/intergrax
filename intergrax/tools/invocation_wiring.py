# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Per-invocation tool wiring contracts (TOOL-ENG-RX / TOOL-ENG-RX-C1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from intergrax.runtime.architecture.cost_budget import BudgetEnvelope
from intergrax.runtime.architecture.cost_quota import ResourceQuota
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.invocation_wiring_requirements import ToolInvocationWiringRequirements
from intergrax.tools.registry.runtime_bindings import RunTraceReaderBinding, TaskMemoryViewBinding
from intergrax.tools.registry.wiring import ToolWiringContext

# Wiring merge ownership (registration vs invocation overlay):
# | Field            | Owner              | Registration | Invocation override | Required by |
# | shadow_workspace | runtime workspace  | yes          | yes                 | contract    |
# | memory_view      | task memory port   | yes          | yes                 | contract    |
# | trace_reader     | trace read port    | yes          | yes                 | contract    |
# | run_budget       | execution budget   | yes          | yes                 | contract    |
# | cost_envelopes   | cost governance    | yes          | yes                 | cost tools  |
# | cost_quotas      | cost governance    | yes          | yes                 | cost tools  |
# | sandbox_session  | sandbox exec port  | yes          | yes                 | contract    |
# | task_metadata    | invocation identity| no (extras)  | yes → extras        | legacy seam |


class ToolWiringResolutionError(Exception):
    """Read-only wiring resolution failure (sanitized at invoker boundary)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ToolWiringOverlay:
    """Invocation-scoped wiring values; unset fields do not override registration wiring."""

    shadow_workspace: ShadowWorkspace | None = None
    memory_view: TaskMemoryViewBinding | None = None
    trace_reader: RunTraceReaderBinding | None = None
    run_budget: RunBudget | None = None
    cost_envelopes: tuple[BudgetEnvelope, ...] | None = None
    cost_quotas: tuple[ResourceQuota, ...] | None = None
    sandbox_session: SandboxExecCapable | None = None
    task_metadata: Mapping[str, str] | None = None

    @classmethod
    def empty(cls) -> ToolWiringOverlay:
        return cls()


@runtime_checkable
class ToolInvocationWiringResolver(Protocol):
    """Read-only dependency projection for a single tool invocation."""

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: "ToolInvocationContext",
        registration_wiring: ToolWiringContext,
    ) -> ToolWiringOverlay:
        ...


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """Narrow invocation identity for wiring resolution (no arbitrary metadata bag)."""

    run_id: str
    step_id: str
    tool_id: str
    agent_id: str | None = None
    tenant_id: str | None = None
    correlation_request_id: str | None = None
    wiring_resolver: ToolInvocationWiringResolver | None = None


class DelegatingToolInvocationWiringResolver:
    """Default invoker resolver: optional per-request delegate, else empty overlay."""

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolWiringContext,
    ) -> ToolWiringOverlay:
        delegate = invocation_context.wiring_resolver
        if delegate is None:
            return ToolWiringOverlay.empty()
        overlay = delegate.resolve(
            tool_id=tool_id,
            invocation_context=invocation_context,
            registration_wiring=registration_wiring,
        )
        return ensure_tool_wiring_overlay(overlay)


def ensure_tool_wiring_overlay(overlay: ToolWiringOverlay) -> ToolWiringOverlay:
    """Fail closed when an external resolver returns an invalid overlay type."""
    if not isinstance(overlay, ToolWiringOverlay):
        raise ToolWiringResolutionError(
            "wiring_overlay_invalid_type",
            "tool invocation wiring resolver must return ToolWiringOverlay",
        )
    return overlay


def registration_wiring_for_handler(handler: object) -> ToolWiringContext:
    from intergrax.tools.core.handler import WiringContextToolHandler

    if isinstance(handler, WiringContextToolHandler):
        return handler.registration_wiring
    return ToolWiringContext()


def merge_invocation_wiring(
    registration: ToolWiringContext,
    overlay: ToolWiringOverlay,
) -> ToolWiringContext:
    """Deterministic merge: invocation-scoped overlay fields override registration when set."""
    effective = registration
    if overlay.shadow_workspace is not None:
        effective = replace(effective, shadow_workspace=overlay.shadow_workspace)
    if overlay.memory_view is not None:
        effective = replace(effective, memory_view=overlay.memory_view)
    if overlay.trace_reader is not None:
        effective = replace(effective, trace_reader=overlay.trace_reader)
    if overlay.run_budget is not None:
        effective = replace(effective, run_budget=overlay.run_budget)
    if overlay.cost_envelopes is not None:
        effective = replace(effective, cost_envelopes=overlay.cost_envelopes)
    if overlay.cost_quotas is not None:
        effective = replace(effective, cost_quotas=overlay.cost_quotas)
    if overlay.sandbox_session is not None:
        effective = replace(effective, sandbox_session=overlay.sandbox_session)
    if overlay.task_metadata is not None:
        merged_extras = dict(registration.extras)
        merged_extras["task_metadata"] = dict(overlay.task_metadata)
        effective = replace(effective, extras=merged_extras)
    return effective


def validate_invocation_wiring(
    requirements: ToolInvocationWiringRequirements,
    effective: ToolWiringContext,
) -> None:
    if requirements.shadow_workspace and effective.shadow_workspace is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "shadow_workspace required for this tool invocation",
        )
    if requirements.memory_view and effective.memory_view is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "memory_view required for this tool invocation",
        )
    if requirements.trace_reader and effective.trace_reader is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "trace_reader required for this tool invocation",
        )
    if requirements.run_budget and effective.run_budget is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "run_budget required for this tool invocation",
        )
    if requirements.sandbox_session and effective.sandbox_session is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "sandbox_session required for this tool invocation",
        )


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
