# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Per-invocation tool wiring contracts (TOOL-ENG-RX)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from intergrax.tools.invocation_wiring_requirements import ToolInvocationWiringRequirements
from intergrax.tools.registry.wiring import ToolWiringContext


class ToolWiringResolutionError(Exception):
    """Read-only wiring resolution failure (sanitized at invoker boundary)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ToolWiringOverlay:
    """Invocation-scoped wiring values; unset fields do not override registration wiring."""

    shadow_workspace: Any | None = None
    memory_view: object | None = None
    trace_reader: object | None = None
    run_budget: Any | None = None
    cost_envelopes: tuple[Any, ...] | None = None
    cost_quotas: tuple[Any, ...] | None = None
    sandbox_session: Any | None = None
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
        return delegate.resolve(
            tool_id=tool_id,
            invocation_context=invocation_context,
            registration_wiring=registration_wiring,
        )


def registration_wiring_for_handler(handler: object) -> ToolWiringContext:
    from intergrax.tools.core.handler import WiringContextToolHandler

    if isinstance(handler, WiringContextToolHandler):
        return handler._ctx
    return ToolWiringContext()


def merge_invocation_wiring(
    registration: ToolWiringContext,
    overlay: ToolWiringOverlay,
) -> ToolWiringContext:
    """Deterministic merge: invocation-scoped overlay fields override registration when set."""
    updates: dict[str, Any] = {}
    if overlay.shadow_workspace is not None:
        updates["shadow_workspace"] = overlay.shadow_workspace
    if overlay.memory_view is not None:
        updates["memory_view"] = overlay.memory_view
    if overlay.trace_reader is not None:
        updates["trace_reader"] = overlay.trace_reader
    if overlay.run_budget is not None:
        updates["run_budget"] = overlay.run_budget
    if overlay.cost_envelopes is not None:
        updates["cost_envelopes"] = overlay.cost_envelopes
    if overlay.cost_quotas is not None:
        updates["cost_quotas"] = overlay.cost_quotas
    if overlay.sandbox_session is not None:
        updates["sandbox_session"] = overlay.sandbox_session
    if overlay.task_metadata is not None:
        merged_extras = dict(registration.extras)
        merged_extras["task_metadata"] = dict(overlay.task_metadata)
        updates["extras"] = merged_extras
    if not updates:
        return registration
    return replace(registration, **updates)


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
