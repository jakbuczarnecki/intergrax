# © Artur Czarnecki. All rights reserved.

"""Live platform scenario qualification helpers for DS-E2E-12/13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import llm_profile_from_env
from intergrax.runtime.decision_flow import DecisionFlowScope

from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    resolve_scenario_llm_adapter,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.proof.reproduction import (
    canonical_reproduction_shell_command,
)
from testing_support.decision_e2e.qualification_evidence import ScenarioExecutionEvidence


CANONICAL_DECISION_RUNTIME_MODULES: frozenset[str] = frozenset(
    {
        "intergrax.runtime.decision_flow",
        "intergrax.applications._shared.decision_wiring",
        "intergrax.runtime.execution.decision_lifecycle_host",
    },
)

AI_INCIDENT_SCENARIO_ID = "ai_incident_investigation"


@dataclass(frozen=True, slots=True)
class ScenarioQualificationAttempt:
    evidence: ScenarioExecutionEvidence
    evaluation_passed: bool
    error: str | None = None


def _is_mock_adapter_type(adapter: object) -> bool:
    module = type(adapter).__module__
    name = type(adapter).__name__
    if "fixture" in module.lower():
        return True
    if "fake" in name.lower() or "fixture" in name.lower() or "lab" in name.lower():
        return True
    return name == "FixtureDrivenIncidentInvestigationLLM"


def _provider_model_from_env() -> tuple[str, str | None]:
    profile = llm_profile_from_env()
    provider = (
        profile.provider.value
        if isinstance(profile.provider, LLMProvider)
        else str(profile.provider)
    )
    return provider, profile.model


async def run_ai_incident_live_qualification(
    *,
    variant: ScenarioVariant = ScenarioVariant.RESOLVED,
) -> ScenarioQualificationAttempt:
    invocation = canonical_reproduction_shell_command()
    provider, model = _provider_model_from_env()
    try:
        fixture_bundle = build_fixture_runtime_bundle(variant=variant)
        bundle = fixture_bundle.bundle
        composition = bundle.runtime_composition
        adapter = resolve_scenario_llm_adapter(composition.environment)
        if _is_mock_adapter_type(adapter):
            return ScenarioQualificationAttempt(
                evidence=ScenarioExecutionEvidence(
                    scenario_id=AI_INCIDENT_SCENARIO_ID,
                    invocation=invocation,
                    provider=provider,
                    model=model,
                    executed=False,
                    decision_path_exercised=None,
                    used_mock_provider=True,
                    block_reason="Scenario resolved to mock/fixture LLM adapter",
                ),
                evaluation_passed=False,
            )

        decision_gate = composition.platform.nexus_loop.peek_decision_flow_gate()
        decision_path = (
            decision_gate is not None
            and decision_gate.supports_scope(DecisionFlowScope.GRAPH_FINAL)
        )
        if not decision_path:
            return ScenarioQualificationAttempt(
                evidence=ScenarioExecutionEvidence(
                    scenario_id=AI_INCIDENT_SCENARIO_ID,
                    invocation=invocation,
                    provider=provider,
                    model=model,
                    executed=False,
                    decision_path_exercised=False,
                    used_mock_provider=False,
                    block_reason="Scenario runtime has no canonical Decision flow gate",
                ),
                evaluation_passed=False,
            )

        result = await execute_resolved_skeleton(bundle)
        evaluation = evaluate_scenario_run(result, fixture_bundle.fixture)
    except Exception as exc:
        return ScenarioQualificationAttempt(
            evidence=ScenarioExecutionEvidence(
                scenario_id=AI_INCIDENT_SCENARIO_ID,
                invocation=invocation,
                provider=provider,
                model=model,
                executed=False,
                decision_path_exercised=None,
                used_mock_provider=False,
                block_reason=f"scenario execution failed: {type(exc).__name__}",
            ),
            evaluation_passed=False,
            error=str(exc),
        )

    return ScenarioQualificationAttempt(
        evidence=ScenarioExecutionEvidence(
            scenario_id=AI_INCIDENT_SCENARIO_ID,
            invocation=invocation,
            provider=provider,
            model=model,
            executed=True,
            decision_path_exercised=True,
            used_mock_provider=False,
            outcome=result.outcome,
            runtime_modules=CANONICAL_DECISION_RUNTIME_MODULES,
        ),
        evaluation_passed=evaluation.passed,
        error=None if evaluation.passed else "; ".join(evaluation.failures),
    )


def discover_decision_scenario_roots(repo_root: Path) -> tuple[Path, ...]:
    scenarios_root = repo_root / "platform_proofs" / "scenarios"
    roots: list[Path] = []
    for candidate in (
        scenarios_root / "ai_incident_investigation",
        scenarios_root / "verified_product_identification",
        scenarios_root / "indirect_prompt_injection",
    ):
        if candidate.is_dir():
            roots.append(candidate)
    return tuple(roots)


def scenario_exercises_decision_runtime(scenario_dir: Path) -> bool:
    runtime_composition = scenario_dir / "application" / "runtime_composition.py"
    if not runtime_composition.is_file():
        return False
    source = runtime_composition.read_text(encoding="utf-8")
    return "DecisionProfile" in source or "peek_decision_flow_gate" in source
