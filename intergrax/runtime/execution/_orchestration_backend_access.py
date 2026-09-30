# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution Engine owner-zone-only orchestration backend access (EBH-4-R1-R1)."""

from __future__ import annotations

from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def orchestration_backend_for_execution_engine(
    materialization: EnvironmentOrchestrationMaterialization,
) -> NexusLoop:
    """Return private Nexus backend — callable only from ``runtime/execution/**``."""
    return materialization._backend  # noqa: SLF001 — EE owner-zone gate


__all__ = ["orchestration_backend_for_execution_engine"]
