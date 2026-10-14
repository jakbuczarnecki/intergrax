# © Artur Czarnecki. All rights reserved.

"""Adapter-scoped physical execution gate for sanctioned composition (PROD-Q-R1-R7-D1-R6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.colocated_serving_unit_execution_admission import (
    ColocatedUnitPhysicalExecutionGate,
)


@dataclass(frozen=True, slots=True)
class VenvBundleColocatedUnitPhysicalExecutionGate:
    """Routes physical in-flight accounting through the deployment adapter only."""

    adapter: VenvBundleColocatedRuntimeDeploymentAdapter

    def admit_execution(self, serving_unit_ref: str) -> None:
        self.adapter.admit_unit_execution(serving_unit_ref)

    def release_execution(self, serving_unit_ref: str) -> None:
        self.adapter.release_unit_execution(serving_unit_ref)


def as_colocated_unit_physical_gate(
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
) -> ColocatedUnitPhysicalExecutionGate:
    return VenvBundleColocatedUnitPhysicalExecutionGate(adapter=adapter)


__all__ = [
    "VenvBundleColocatedUnitPhysicalExecutionGate",
    "as_colocated_unit_physical_gate",
]
