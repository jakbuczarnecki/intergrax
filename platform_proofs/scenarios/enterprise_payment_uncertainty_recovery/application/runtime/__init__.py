"""Runtime integration with Integrax lab baseline (future ERL wiring)."""

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.runtime.platform_lab_runtime import (
    SYNTHETIC_SCENARIO_TENANT_ID,
    build_scenario_runtime,
)

__all__ = ["SYNTHETIC_SCENARIO_TENANT_ID", "build_scenario_runtime"]
