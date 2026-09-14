# © Artur Czarnecki. All rights reserved.

"""Execution Engine SLI catalog and SLO contract shape (EE-B4-A, provider-neutral)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SloTargetOwner(StrEnum):
    DEPLOYMENT_OPERATOR = "deployment_operator"
    EXECUTION_ENGINE_FACTS = "execution_engine_facts"
    OPERATIONAL_LAYER = "operational_layer"


@dataclass(frozen=True, slots=True)
class ExecutionEngineSliDefinition:
    sli_id: str
    description: str
    numerator: str
    denominator: str
    measurement_window: str
    fact_owner: SloTargetOwner
    scope: str


@dataclass(frozen=True, slots=True)
class SloContractShape:
    """Deployment-configurable SLO target envelope (no hardcoded availability %)."""

    sli_id: str
    target_description: str
    measurement_window: str
    target_owner: SloTargetOwner
    error_budget_model: str


EXECUTION_ENGINE_SLI_CATALOG: tuple[ExecutionEngineSliDefinition, ...] = (
    ExecutionEngineSliDefinition(
        sli_id="execution_success_rate",
        description="Share of started attempts that reach successful terminal outcome.",
        numerator="terminal_success_attempts",
        denominator="terminal_attempts",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="tenant_or_global",
    ),
    ExecutionEngineSliDefinition(
        sli_id="execution_failure_rate",
        description="Share of started attempts that reach failed terminal outcome.",
        numerator="terminal_failed_attempts",
        denominator="terminal_attempts",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="tenant_or_global",
    ),
    ExecutionEngineSliDefinition(
        sli_id="admission_reject_defer_rate",
        description="Pre-admission capacity rejections and deferrals vs admission attempts.",
        numerator="capacity_reject_or_defer",
        denominator="admission_attempts",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="global_root_capacity",
    ),
    ExecutionEngineSliDefinition(
        sli_id="execution_latency",
        description="Attempt duration from admission to terminal event (p50/p95/p99 operator choice).",
        numerator="attempt_duration_ms",
        denominator="terminal_attempts",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="per_attempt",
    ),
    ExecutionEngineSliDefinition(
        sli_id="admission_wait_latency",
        description="Time spent waiting on capacity port acquire (bounded wait mode only).",
        numerator="capacity_wait_ms",
        denominator="admitted_attempts",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="global_root_capacity",
    ),
    ExecutionEngineSliDefinition(
        sli_id="capacity_utilization",
        description="active_root_executions / capacity_limit snapshot ratio.",
        numerator="active_root_executions",
        denominator="capacity_limit",
        measurement_window="point_in_time_or_rolling_max",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="global_root_capacity",
    ),
    ExecutionEngineSliDefinition(
        sli_id="worker_failure_rate",
        description="Worker-isolated failures vs worker dispatches (EE-B1.3 containment plane).",
        numerator="worker_isolated_failures",
        denominator="worker_dispatches",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="worker_pool",
    ),
    ExecutionEngineSliDefinition(
        sli_id="dependency_failure_rate",
        description="Tool/provider failures vs dependency invocations (scoped per provider).",
        numerator="dependency_failures",
        denominator="dependency_invocations",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="dependency",
    ),
    ExecutionEngineSliDefinition(
        sli_id="mandatory_evidence_persistence_failure_rate",
        description="Mandatory runtime event / evidence persistence failures vs mandatory writes.",
        numerator="mandatory_persistence_failures",
        denominator="mandatory_persistence_writes",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="global_or_profile",
    ),
    ExecutionEngineSliDefinition(
        sli_id="recovery_success_rate",
        description="Recovery operations reaching allowed terminal disposition vs recovery starts.",
        numerator="recovery_success",
        denominator="recovery_starts",
        measurement_window="operator_configured_rolling_window",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="execution_or_subsystem",
    ),
    ExecutionEngineSliDefinition(
        sli_id="shutdown_drain_duration",
        description="Elapsed time from STOP_ACCEPTING to drain completion signal.",
        numerator="shutdown_drain_ms",
        denominator="shutdown_cycles",
        measurement_window="per_shutdown_cycle",
        fact_owner=SloTargetOwner.EXECUTION_ENGINE_FACTS,
        scope="global",
    ),
)

SLO_CONTRACT_SHAPE_EXAMPLES: tuple[SloContractShape, ...] = (
    SloContractShape(
        sli_id="execution_success_rate",
        target_description="Operator-defined minimum success ratio (not hardcoded in core).",
        measurement_window="30d_rolling",
        target_owner=SloTargetOwner.DEPLOYMENT_OPERATOR,
        error_budget_model="allowed_bad_terminal_attempts_over_window",
    ),
)
