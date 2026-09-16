# © Artur Czarnecki. All rights reserved.

"""OBS-UNIVERSAL-SPINE-E2E — initialized scenario surfaces on canonical spine."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEventType
from scripts.proof.scenario_architecture_conformance import (
    assert_scenario_application_architecture,
    discover_initialized_scenario_slugs,
)
from testing_support.obs_universal_spine.scenario_runtime_proof import (
    prove_initialized_scenario_runtime,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gate,
    pytest.mark.obs_coverage_p1,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INITIALIZED_SLUGS = discover_initialized_scenario_slugs(_REPO_ROOT)


def test_initialized_scenario_inventory_is_current() -> None:
    assert len(_INITIALIZED_SLUGS) >= 1
    assert all(slug.strip() for slug in _INITIALIZED_SLUGS)


@pytest.mark.parametrize("scenario_slug", _INITIALIZED_SLUGS)
def test_initialized_scenario_application_architecture_gate(scenario_slug: str) -> None:
    assert_scenario_application_architecture(
        repo_root=_REPO_ROOT,
        scenario_slug=scenario_slug,
    )


@pytest.mark.parametrize("scenario_slug", _INITIALIZED_SLUGS)
def test_initialized_scenario_delegates_through_execute_scenario_task(scenario_slug: str) -> None:
    scenario_py = (
        _REPO_ROOT
        / "platform_proofs"
        / "scenarios"
        / scenario_slug
        / "application"
        / "scenario.py"
    )
    source = scenario_py.read_text(encoding="utf-8")
    assert "execute_scenario_task" in source


@pytest.mark.parametrize("scenario_slug", _INITIALIZED_SLUGS)
@pytest.mark.asyncio
async def test_initialized_scenario_runtime_spine_proof(
    scenario_slug: str,
    tmp_path: Path,
) -> None:
    proof = await prove_initialized_scenario_runtime(
        scenario_slug,
        workspace_root=tmp_path / scenario_slug,
    )
    assert proof.slug == scenario_slug
    assert proof.has_runtime_events
    assert proof.has_terminal_diagnostic_trigger
    assert proof.reconstruction_complete
    assert proof.terminal_state.name in {"COMPLETED", "FAILED", "PARTIALLY_COMPLETED"}


def test_enterprise_payment_lab_execution_uses_platform_erl_spine() -> None:
    from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution import (
        EnterprisePaymentScenarioExecutionRequest,
        build_lab_execution_composition,
    )

    executor = build_lab_execution_composition()
    result = executor.execute(
        EnterprisePaymentScenarioExecutionRequest(
            variant_id="payment_completed_after_unknown",
            run_id="obs-spine-erl-lab",
        ),
    )
    assert result.evidence_ref is not None
    assert result.lifecycle_outcome is not None


@pytest.mark.asyncio
async def test_indirect_prompt_injection_execute_scenario_task_persists_terminal_events() -> None:
    from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import build_safe_read_fixture
    from platform_proofs.scenarios.indirect_prompt_injection.fixtures.runtime_bundle import (
        build_fixture_runtime_bundle,
    )
    from platform_proofs.scenarios.indirect_prompt_injection.proof.harness import execute_fixture_scenario_run
    from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.in_process_order_provider import (
        InProcessOrderProviderClient,
    )
    from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.llm_doubles import (
        SummaryOnlyOrderLLM,
    )

    fixture = build_safe_read_fixture()
    provider = InProcessOrderProviderClient()
    bundle = build_fixture_runtime_bundle(
        fixture,
        order_operations=provider,
        llm_adapter_override=SummaryOnlyOrderLLM(),
    )
    platform = bundle.run_bundle.runtime_composition.platform
    observed = await execute_fixture_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=provider,
        provider_control=provider,
        fixture=fixture,
    )
    store = platform.observability.runtime_event_store
    assert store is not None
    events = store.list_for_run(observed.run_id, tenant_id=observed.tenant_id)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)
