# © Artur Czarnecki. All rights reserved.

"""Neutral host-orchestration wiring surface (EE applies; Nexus implements internally)."""

from __future__ import annotations

from typing import Any, Protocol

from intergrax.contracts.agent_execution_result import AgentExecutionResult


class HostOrchestrationApplicationWiringTarget(Protocol):
    """Execution-semantic orchestration host target for Tier-3 wiring (not Nexus-public)."""

    @property
    def middleware(self) -> object: ...

    @property
    def event_bus(self) -> object: ...

    def apply_decision_flow_gate(
        self,
        gate: object,
        *,
        verify_uaep_step: bool,
        verify_graph_final: bool,
    ) -> None: ...

    def apply_decision_exposure_selection(self, selection: object) -> None: ...

    def apply_validation_engine(self, validation_engine: object | None) -> None: ...

    def set_hold_persisted_trace_finalize(self, hold: bool) -> None: ...

    def take_deferred_persisted_trace_finalize(self) -> object | None: ...

    def attach_terminal_diagnostic_trigger(self, port: object) -> None: ...


class HostOrchestrationPluginBootstrapTarget(Protocol):
    """Plugin and platform bootstrap surface without Nexus types."""

    @property
    def event_bus(self) -> object: ...

    @property
    def middleware(self) -> object: ...

    @property
    def policy_engine(self) -> object: ...

    @property
    def trace_store(self) -> object: ...

    @property
    def trace_emitter(self) -> object: ...

    @property
    def runtime_event_store(self) -> object: ...

    @property
    def execution_budget_ledger_factory(self) -> object | None: ...


class HostOrchestrationAssemblyInspectionTarget(Protocol):
    """Narrow read surface for assembly validation without Nexus types."""

    @property
    def middleware(self) -> object: ...

    def peek_decision_flow_gate(self) -> object | None: ...

    @property
    def policy_engine(self) -> Any: ...


__all__ = [
    "HostOrchestrationApplicationWiringTarget",
    "HostOrchestrationAssemblyInspectionTarget",
    "HostOrchestrationPluginBootstrapTarget",
]
