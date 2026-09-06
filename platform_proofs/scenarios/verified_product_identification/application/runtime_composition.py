"""Scenario runtime composition via platform scenario runtime baseline."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioLabAgentRegistration,
    ScenarioRuntimeComposition,
    build_scenario_lab_agent_registry,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    build_scenario_lab_runtime,
)

from platform_proofs.scenarios.verified_product_identification.application.agent import VerifiedProductIdentificationAgent

SYNTHETIC_SCENARIO_TENANT_ID = "synthetic-scenario-verified_product_identification"


def build_scenario_runtime(
    *,
    tenant_id: str = SYNTHETIC_SCENARIO_TENANT_ID,
    workspace_root: Path | None = None,
) -> ScenarioRuntimeComposition:
    agent = VerifiedProductIdentificationAgent()
    registry = build_scenario_lab_agent_registry(
        ScenarioLabAgentRegistration(agent=agent),
    )
    return build_scenario_lab_runtime(
        registry=registry,
        tenant_id=tenant_id,
        scenario_slug="verified_product_identification",
        workspace_root=workspace_root,
    )
