"""Scenario runtime composition via platform scenario runtime baseline."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeComposition,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    build_scenario_lab_runtime,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry

from platform_proofs.scenarios.indirect_prompt_injection.application.agent import IndirectPromptInjectionAgent

SYNTHETIC_SCENARIO_TENANT_ID = "synthetic-scenario-indirect_prompt_injection"


def build_scenario_runtime(
    *,
    tenant_id: str = SYNTHETIC_SCENARIO_TENANT_ID,
    workspace_root: Path | None = None,
) -> ScenarioRuntimeComposition:
    registry = AgentRegistry()
    registry.register(IndirectPromptInjectionAgent())
    return build_scenario_lab_runtime(
        registry=registry,
        tenant_id=tenant_id,
        scenario_slug="indirect_prompt_injection",
        workspace_root=workspace_root,
    )
