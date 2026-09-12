"""Harness input for one full scenario execution — variant selects dataset slice only."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.runtime.platform_lab_runtime import (
    SYNTHETIC_SCENARIO_TENANT_ID,
)


@dataclass(frozen=True, slots=True)
class EnterprisePaymentScenarioExecutionRequest:
    """Single enterprise proof run — business semantics come from provisioned data."""

    variant_id: str
    run_id: str
    tenant_id: str = SYNTHETIC_SCENARIO_TENANT_ID
