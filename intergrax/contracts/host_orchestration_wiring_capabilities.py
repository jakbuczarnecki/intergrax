# © Artur Czarnecki. All rights reserved.

"""Neutral host-orchestration wiring capability ports (Tier-0, no runtime imports)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from intergrax.contracts.event_taxonomy import EventCategory
from intergrax.contracts.execution_budget_ledger_port import ExecutionBudgetLedgerFactoryPort
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.run_trace_store import RunTraceReader
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType


@runtime_checkable
class HostOrchestrationRuntimeMiddlewareRegistration(Protocol):
    """One middleware registration mounted on the orchestration pipeline."""

    name: str
    priority: int


@runtime_checkable
class HostOrchestrationPolicyRuleRegistration(Protocol):
    """One policy rule registered by a runtime plugin at bootstrap."""

    @property
    def rule_id(self) -> str: ...


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
class HostOrchestrationPluginPolicyEnginePort(Protocol):
    """Minimal policy registration surface for runtime plugins."""

    def register_rule(self, rule: HostOrchestrationPolicyRuleRegistration) -> None: ...


@runtime_checkable
class HostOrchestrationTraceEmitterPort(Protocol):
    """Trace emitter exposing persisted trace read access for platform plugins."""

    @property
    def trace_store(self) -> RunTraceReader: ...


__all__ = [
    "HostOrchestrationHookRegistryPort",
    "HostOrchestrationMiddlewarePipelinePort",
    "HostOrchestrationPluginPolicyEnginePort",
    "HostOrchestrationPolicyRuleRegistration",
    "HostOrchestrationRuntimeEventPort",
    "HostOrchestrationRuntimeMiddlewareRegistration",
    "HostOrchestrationTraceEmitterPort",
]
