# © Artur Czarnecki. All rights reserved.

"""Neutral host-orchestration wiring capability ports (Tier-0, no runtime imports)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol, runtime_checkable

from intergrax.contracts.event_taxonomy import EventCategory
from intergrax.contracts.execution_budget_ledger_port import ExecutionBudgetLedgerFactoryPort
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.run_trace_store import RunTraceReader
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.middleware_hook_point import HookPoint
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType


@runtime_checkable
class HostOrchestrationMiddlewareHookAction(Protocol):
    """Hook action token returned by executable orchestration middleware."""

    @property
    def value(self) -> str: ...


@runtime_checkable
class HostOrchestrationMiddlewareHookContext(Protocol):
    """Minimal hook context required for orchestration middleware execution."""

    task_id: str
    run_id: str
    node_id: str | None
    agent_id: str | None
    step_id: str | None
    phase: ExecutionPhase

    @property
    def runtime_state(self) -> Mapping[str, object]: ...


@runtime_checkable
class HostOrchestrationMiddlewareHookResult(Protocol):
    """Hook outcome consumed by the orchestration middleware pipeline."""

    @property
    def action(self) -> HostOrchestrationMiddlewareHookAction: ...

    @property
    def reason(self) -> str | None: ...


@runtime_checkable
class HostOrchestrationRuntimeMiddlewareRegistration(Protocol):
    """Executable middleware registration mounted on the orchestration pipeline."""

    name: str
    priority: int

    async def before(
        self,
        point: HookPoint,
        ctx: HostOrchestrationMiddlewareHookContext,
    ) -> HostOrchestrationMiddlewareHookResult: ...

    async def after(
        self,
        point: HookPoint,
        ctx: HostOrchestrationMiddlewareHookContext,
    ) -> HostOrchestrationMiddlewareHookResult: ...


@runtime_checkable
class HostOrchestrationHookRegistryPort(Protocol):
    """Hook registration surface required by runtime plugin bootstrap."""

    def register(
        self,
        point: str,
        handler: Callable[..., object],
        *,
        priority: int = 100,
        name: str | None = None,
        hook_id: str | None = None,
    ) -> str: ...


@runtime_checkable
class HostOrchestrationRuntimeEventPort(Protocol):
    """Runtime event spine subscription surface for Tier-3 host wiring."""

    def subscribe(
        self,
        handler: Callable[[RuntimeEvent], None | Awaitable[None]],
        *,
        event_types: set[RuntimeEventType] | None = None,
        categories: set[EventCategory] | None = None,
        kind_prefix: str | None = None,
        ops_hints: set[str] | None = None,
        priority: int = 100,
        subscription_id: str | None = None,
    ) -> str: ...

    def unsubscribe(self, subscription_id: str) -> None: ...

    async def publish(self, event: RuntimeEvent) -> None: ...


@runtime_checkable
class HostOrchestrationMiddlewarePipelinePort(Protocol):
    """Middleware attachment and hook runtime configuration for orchestration hosts."""

    @property
    def hooks(self) -> HostOrchestrationHookRegistryPort: ...

    @property
    def hook_timeout_seconds(self) -> float | None: ...

    def configure_hook_runtime(
        self,
        *,
        hook_timeout_seconds: float | None,
        event_bus: HostOrchestrationRuntimeEventPort | None,
    ) -> None: ...

    def registered_middleware_names(self) -> frozenset[str]: ...

    def attach_runtime_middleware_if_absent(
        self,
        middleware: HostOrchestrationRuntimeMiddlewareRegistration,
    ) -> None: ...


@runtime_checkable
class HostOrchestrationTraceEmitterPort(Protocol):
    """Trace emitter exposing persisted trace read access for platform plugins."""

    @property
    def trace_store(self) -> RunTraceReader: ...


__all__ = [
    "HostOrchestrationHookRegistryPort",
    "HostOrchestrationMiddlewareHookAction",
    "HostOrchestrationMiddlewareHookContext",
    "HostOrchestrationMiddlewareHookResult",
    "HostOrchestrationMiddlewarePipelinePort",
    "HostOrchestrationRuntimeEventPort",
    "HostOrchestrationRuntimeMiddlewareRegistration",
    "HostOrchestrationTraceEmitterPort",
]
