# © Artur Czarnecki. All rights reserved.

"""Shared EE-B4-A operational fact fixtures."""

from __future__ import annotations

from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityOverloadMode,
)
from testing_support.execution_operational_readiness.assessment import (
    ExecutionOperationalFacts,
    ExecutionOperationalScope,
)


def baseline_operational_facts(**overrides: object) -> ExecutionOperationalFacts:
    base = {
        "scope": ExecutionOperationalScope.GLOBAL,
        "process_alive": True,
        "startup_complete": True,
        "shutdown_phase": None,
        "active_root_executions": 0,
        "capacity_limit": 4,
        "overload_mode": ExecutionCapacityOverloadMode.REJECT,
        "diagnostics_required": True,
        "diagnostics_attached": True,
        "runtime_event_persistence_available": True,
        "diagnostic_read_side_required": False,
        "diagnostic_read_side_ready": True,
        "mandatory_evidence_persistence_available": True,
        "best_effort_observability_export_available": True,
        "worker_pool_degraded": False,
        "dependency_degraded": False,
        "recovery_subsystem_degraded": False,
    }
    base.update(overrides)
    return ExecutionOperationalFacts(**base)  # type: ignore[arg-type]
