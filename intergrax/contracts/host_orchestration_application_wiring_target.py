# © Artur Czarnecki. All rights reserved.

"""Neutral host-orchestration wiring surface (EE applies; Nexus implements internally)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.agent_execution_validation_engine import (
    AgentExecutionValidationEnginePort,
)
from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureSelectionHostBinding,
)
from intergrax.contracts.decision_flow_gate import DecisionFlowGate
from intergrax.contracts.deferred_persisted_trace_finalize_port import (
    DeferredPersistedTraceFinalizePort,
)
from intergrax.contracts.diagnostics.terminal_execution_diagnostic_port import (
    TerminalExecutionDiagnosticPort,
)
from intergrax.contracts.execution_budget_ledger_port import ExecutionBudgetLedgerFactoryPort
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationMiddlewarePipelinePort,
    HostOrchestrationRuntimeEventPort,
    HostOrchestrationTraceEmitterPort,
)
from intergrax.contracts.run_trace_store import RunTraceReader


class HostOrchestrationApplicationWiringTarget(Protocol):
    """Execution-semantic orchestration host target for Tier-3 wiring (not Nexus-public)."""

    @property
    def middleware(self) -> HostOrchestrationMiddlewarePipelinePort: ...

    @property
    def event_bus(self) -> HostOrchestrationRuntimeEventPort: ...

    def apply_decision_flow_gate(
        self,
        gate: DecisionFlowGate[AgentExecutionResult] | None,
        *,
        verify_uaep_step: bool,
        verify_graph_final: bool,
    ) -> None: ...

    def apply_decision_exposure_selection(
        self,
        selection: DecisionExposureSelectionHostBinding,
    ) -> None: ...

    def apply_validation_engine(
        self,
        validation_engine: AgentExecutionValidationEnginePort | None,
    ) -> None: ...

    def set_hold_persisted_trace_finalize(self, hold: bool) -> None: ...

    def take_deferred_persisted_trace_finalize(
        self,
    ) -> DeferredPersistedTraceFinalizePort | None: ...

    def attach_terminal_diagnostic_trigger(
        self,
        port: TerminalExecutionDiagnosticPort,
    ) -> None: ...


class HostOrchestrationPluginBootstrapTarget(Protocol):
    """Plugin and platform bootstrap surface without Nexus types."""

    @property
    def event_bus(self) -> HostOrchestrationRuntimeEventPort: ...

    @property
    def middleware(self) -> HostOrchestrationMiddlewarePipelinePort: ...

    @property
    def trace_store(self) -> RunTraceReader | None: ...

    @property
    def trace_emitter(self) -> HostOrchestrationTraceEmitterPort | None: ...

    @property
    def runtime_event_store(self) -> EvidencePersistencePort | None: ...

    @property
    def execution_budget_ledger_factory(self) -> ExecutionBudgetLedgerFactoryPort | None: ...


class HostOrchestrationAssemblyInspectionTarget(Protocol):
    """Narrow read surface for assembly validation without Nexus types."""

    @property
    def middleware(self) -> HostOrchestrationMiddlewarePipelinePort: ...


__all__ = [
    "HostOrchestrationApplicationWiringTarget",
    "HostOrchestrationAssemblyInspectionTarget",
    "HostOrchestrationPluginBootstrapTarget",
]
