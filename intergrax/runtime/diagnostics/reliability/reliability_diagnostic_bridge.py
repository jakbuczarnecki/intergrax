# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Runtime bridge from ERL reliability observations to central diagnostics (ERL-DIAG-001B)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.enterprise_reliability.diagnostics.emitter import (
    ExternalEffectReliabilityDiagnosticEmitter,
)
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    ExternalEffectReliabilityObservation,
)
from intergrax.logging import IntergraxLogging
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationRequest,
    DiagnosticOrchestrationResult,
    DiagnosticSignalSubjectScope,
)
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingStrategyId
from intergrax.runtime.diagnostics.reliability.observation_to_problem_signal import (
    ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID,
    map_handoff_to_platform_problem_signal,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_handoff import (
    ReliabilityDiagnosticHandoffIntegrityError,
    map_observation_to_handoff,
)


class ReliabilityDiagnosticOrchestrationPort(Protocol):
    """Narrow orchestration boundary — callers inject DiagnosticOrchestrator or test double."""

    def run(self, request: DiagnosticOrchestrationRequest) -> DiagnosticOrchestrationResult: ...


class ReliabilityDiagnosticBridge:
    """
    Maps validated ERL observations into non-execution diagnostic orchestration.

    Does not write persistence directly; does not mutate ERL or execution state.
    """

    def __init__(
        self,
        orchestration: ReliabilityDiagnosticOrchestrationPort,
        *,
        grouping_strategy_id: ProblemGroupingStrategyId,
    ) -> None:
        self._orchestration = orchestration
        self._grouping_strategy_id = grouping_strategy_id
        self._logger = IntergraxLogging.get_logger(__name__, component="diagnostics")

    def on_observation(self, observation: ExternalEffectReliabilityObservation) -> None:
        """Best-effort handoff into central diagnostics; failures are contained."""
        try:
            handoff = map_observation_to_handoff(observation)
            signal = map_handoff_to_platform_problem_signal(handoff)
            contract_id = handoff.correlation.external_effect_contract_id
            scope = DiagnosticSignalSubjectScope(
                tenant_id=handoff.tenant_id,
                application_id=ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID,
                instance_id=handoff.reliability_case_id,
                problem_signals=(signal,),
            )
            request = DiagnosticOrchestrationRequest(
                tenant_id=handoff.tenant_id,
                grouping_strategy_id=self._grouping_strategy_id,
                observed_at=handoff.recorded_at,
                signal_subjects=(scope,),
            )
            result = self._orchestration.run(request)
            self._logger.info(
                "ERL reliability diagnostic bridge completed",
                extra={
                    "tenant_id": handoff.tenant_id,
                    "observation_id": handoff.observation_id,
                    "reliability_case_id": handoff.reliability_case_id,
                    "external_effect_contract_id": contract_id,
                    "signal_kind": handoff.signal_kind.value,
                    "problems_created": len(result.lifecycle_result.created),
                    "problems_updated": len(result.lifecycle_result.updated),
                },
            )
        except ReliabilityDiagnosticHandoffIntegrityError:
            self._logger.exception(
                "ERL reliability observation rejected at bridge mapping",
                extra={"observation_id": observation.observation_id},
            )
        except Exception:
            self._logger.exception(
                "ERL reliability diagnostic bridge failed",
                extra={
                    "tenant_id": observation.tenant_id,
                    "observation_id": observation.observation_id,
                    "reliability_case_id": observation.reliability_case_id,
                },
            )


class RuntimeExternalEffectReliabilityDiagnosticEmitter:
    """Public emitter port backed by ReliabilityDiagnosticBridge."""

    def __init__(self, bridge: ReliabilityDiagnosticBridge) -> None:
        self._bridge = bridge

    def emit(self, observation: ExternalEffectReliabilityObservation) -> None:
        self._bridge.on_observation(observation)


def build_reliability_diagnostic_emitter(
    orchestration: ReliabilityDiagnosticOrchestrationPort,
    *,
    grouping_strategy_id: ProblemGroupingStrategyId,
) -> ExternalEffectReliabilityDiagnosticEmitter:
    bridge = ReliabilityDiagnosticBridge(
        orchestration,
        grouping_strategy_id=grouping_strategy_id,
    )
    return RuntimeExternalEffectReliabilityDiagnosticEmitter(bridge)


__all__ = [
    "ReliabilityDiagnosticBridge",
    "ReliabilityDiagnosticOrchestrationPort",
    "RuntimeExternalEffectReliabilityDiagnosticEmitter",
    "build_reliability_diagnostic_emitter",
]
