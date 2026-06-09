# © Artur Czarnecki. All rights reserved.

"""Harness SLO catalog types — canonical mirror of HARNESS_ENVIRONMENT § SLO (IDEAL-21.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class HarnessSloDomain(str, Enum):
    RUNTIME = "runtime"
    QUALITY = "quality"
    GOVERNANCE = "governance"
    COST = "cost"


class HarnessSloDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slo_id: str
    domain: HarnessSloDomain
    description: str
    target: str
    measurement: str


HARNESS_SLO_CATALOG: tuple[HarnessSloDefinition, ...] = (
    HarnessSloDefinition(
        slo_id="harness.run.availability",
        domain=HarnessSloDomain.RUNTIME,
        description="Nexus run completion without unhandled internal error",
        target=">= 99.5% over 30d",
        measurement="TASK_COMPLETED / (TASK_COMPLETED + TASK_FAILED internal)",
    ),
    HarnessSloDefinition(
        slo_id="harness.run.p95_latency",
        domain=HarnessSloDomain.RUNTIME,
        description="End-to-end run latency P95",
        target="<= 120s lab / product-specific",
        measurement="trace run duration histogram",
    ),
    HarnessSloDefinition(
        slo_id="harness.quality.contract_pass",
        domain=HarnessSloDomain.QUALITY,
        description="Critic L0 contract pass rate on reference scenarios",
        target=">= 95%",
        measurement="CRIT-V L0 gateway outcomes",
    ),
    HarnessSloDefinition(
        slo_id="harness.governance.policy_block_rate",
        domain=HarnessSloDomain.GOVERNANCE,
        description="Policy blocks with typed rationale",
        target="100% traced",
        measurement="POLICY_DECISION events with payload",
    ),
    HarnessSloDefinition(
        slo_id="harness.cost.budget_breach",
        domain=HarnessSloDomain.COST,
        description="Runs exceeding token budget without prior warn",
        target="0 unexpected breaches",
        measurement="budget enforcer + COST events",
    ),
)


def list_harness_slos() -> list[HarnessSloDefinition]:
    return list(HARNESS_SLO_CATALOG)


def slo_ids() -> frozenset[str]:
    return frozenset(s.slo_id for s in HARNESS_SLO_CATALOG)
