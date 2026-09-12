"""Scenario runtime composition via platform scenario runtime baseline."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeComposition,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    build_scenario_lab_runtime,
)

SYNTHETIC_SCENARIO_TENANT_ID = "synthetic-scenario-enterprise_payment_uncertainty_recovery"


def build_scenario_runtime(
    *,
    tenant_id: str = SYNTHETIC_SCENARIO_TENANT_ID,
    workspace_root: Path | None = None,
) -> ScenarioRuntimeComposition:
    return build_scenario_lab_runtime(
        tenant_id=tenant_id,
        scenario_slug="enterprise_payment_uncertainty_recovery",
        workspace_root=workspace_root,
    )
