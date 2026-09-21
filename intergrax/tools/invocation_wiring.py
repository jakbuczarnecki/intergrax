# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical per-invocation tool wiring contracts (TOOL-ENG-RX-C2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.run_budget import RunBudget
from intergrax.runtime.architecture.cost_budget import BudgetEnvelope
from intergrax.runtime.architecture.cost_quota import ResourceQuota
from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.runtime.workspace.execution_port import WorkspaceExecutionPort
from intergrax.tools.invocation_wiring_requirements import (
    ToolInvocationWiringRequirements,
)
from intergrax.tools.registry.runtime_bindings import (
    RunTraceReaderBinding,
    TaskMemoryViewBinding,
)


class ToolWiringResolutionError(Exception):
    """Read-only wiring resolution failure (sanitized at invoker boundary)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ToolRegistrationWiringView:
    """Minimal registration-time wiring visible to invocation resolvers."""

    workspace: WorkspaceExecutionPort | None = None
    memory_view: TaskMemoryViewBinding | None = None
    trace_reader: RunTraceReaderBinding | None = None
    run_budget: RunBudget | None = None
    cost_envelopes: tuple[BudgetEnvelope, ...] = ()
    cost_quotas: tuple[ResourceQuota, ...] = ()
    sandbox_session: SandboxExecCapable | None = None


@dataclass(frozen=True, slots=True)
class ToolInvocationWiring:
    """Immutable invocation-scoped wiring; unset fields do not override registration."""

    workspace: WorkspaceExecutionPort | None = None
    memory_view: TaskMemoryViewBinding | None = None
    trace_reader: RunTraceReaderBinding | None = None
    run_budget: RunBudget | None = None
    cost_envelopes: tuple[BudgetEnvelope, ...] | None = None
    cost_quotas: tuple[ResourceQuota, ...] | None = None
    sandbox_session: SandboxExecCapable | None = None
    task_metadata: Mapping[str, str] | None = None

    @classmethod
    def empty(cls) -> ToolInvocationWiring:
        return cls()


@runtime_checkable
class ToolInvocationWiringResolver(Protocol):
    """Read-only dependency projection for a single tool invocation."""

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolRegistrationWiringView,
    ) -> ToolInvocationWiring: ...


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
    """Default invoker resolver: optional per-request delegate, else empty wiring."""

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolRegistrationWiringView,
    ) -> ToolInvocationWiring:
        delegate = invocation_context.wiring_resolver
        if delegate is None:
            return ToolInvocationWiring.empty()
        wiring = delegate.resolve(
            tool_id=tool_id,
            invocation_context=invocation_context,
            registration_wiring=registration_wiring,
        )
        return ensure_tool_invocation_wiring(wiring)


def ensure_tool_invocation_wiring(wiring: ToolInvocationWiring) -> ToolInvocationWiring:
    """Fail closed when an external resolver returns an invalid wiring type."""
    if not isinstance(wiring, ToolInvocationWiring):
        raise ToolWiringResolutionError(
            "wiring_invocation_invalid_type",
            "tool invocation wiring resolver must return ToolInvocationWiring",
        )
    return wiring


def compose_effective_invocation_wiring(
    registration: ToolRegistrationWiringView,
    invocation: ToolInvocationWiring,
) -> ToolInvocationWiring:
    """Overlay invocation fields onto registration defaults."""
    return ToolInvocationWiring(
        workspace=(
            invocation.workspace
            if invocation.workspace is not None
            else registration.workspace
        ),
        memory_view=(
            invocation.memory_view
            if invocation.memory_view is not None
            else registration.memory_view
        ),
        trace_reader=(
            invocation.trace_reader
            if invocation.trace_reader is not None
            else registration.trace_reader
        ),
        run_budget=(
            invocation.run_budget
            if invocation.run_budget is not None
            else registration.run_budget
        ),
        cost_envelopes=(
            invocation.cost_envelopes
            if invocation.cost_envelopes is not None
            else registration.cost_envelopes
        ),
        cost_quotas=(
            invocation.cost_quotas
            if invocation.cost_quotas is not None
            else registration.cost_quotas
        ),
        sandbox_session=(
            invocation.sandbox_session
            if invocation.sandbox_session is not None
            else registration.sandbox_session
        ),
        task_metadata=invocation.task_metadata,
    )


def validate_invocation_wiring(
    requirements: ToolInvocationWiringRequirements,
    wiring: ToolInvocationWiring,
) -> None:
    if requirements.workspace and wiring.workspace is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "workspace required for this tool invocation",
        )
    if requirements.memory_view and wiring.memory_view is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "memory_view required for this tool invocation",
        )
    if requirements.trace_reader and wiring.trace_reader is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "trace_reader required for this tool invocation",
        )
    if requirements.run_budget and wiring.run_budget is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "run_budget required for this tool invocation",
        )
    if requirements.sandbox_session and wiring.sandbox_session is None:
        raise ToolWiringResolutionError(
            "wiring_requirement_missing",
            "sandbox_session required for this tool invocation",
        )


@dataclass(frozen=True, slots=True)
class FixedSandboxSessionWiringResolver:
    """Per-invocation sandbox overlay for catalog tools requiring isolation."""

    sandbox_session: SandboxExecCapable

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolRegistrationWiringView,
    ) -> ToolInvocationWiring:
        _ = tool_id, invocation_context, registration_wiring
        return ToolInvocationWiring(sandbox_session=self.sandbox_session)


# TOOL-ENG-RX-C1 compatibility aliases (internal/tests); not canonical ABI names.
ToolWiringOverlay = ToolInvocationWiring
ensure_tool_wiring_overlay = ensure_tool_invocation_wiring
