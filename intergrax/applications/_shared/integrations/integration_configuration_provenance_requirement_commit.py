# © Artur Czarnecki. All rights reserved.

"""Runtime-backed requirement commit port adapter (TRACE-X-P5-R2-P4-R2)."""

from __future__ import annotations

from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    ExecutionIntegrationConfigurationProvenanceRequirementCommitPort,
    ExecutionIntegrationConfigurationProvenanceRequirementCommitResult,
    ExecutionIntegrationConfigurationProvenanceRequirementFact,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.integration_configuration_provenance_requirement_recorder import (
    RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder,
)


class RuntimeEventBusIntegrationConfigurationProvenanceRequirementCommitPort(
    ExecutionIntegrationConfigurationProvenanceRequirementCommitPort,
):
    def __init__(self, bus: RuntimeEventBus) -> None:
        self._recorder = RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder(bus)

    def commit_configured_adopted_requirement(
        self,
        fact: ExecutionIntegrationConfigurationProvenanceRequirementFact,
    ) -> ExecutionIntegrationConfigurationProvenanceRequirementCommitResult:
        return self._recorder.commit(fact)


__all__ = ["RuntimeEventBusIntegrationConfigurationProvenanceRequirementCommitPort"]
