# © Artur Czarnecki. All rights reserved.

"""Live platform scenario qualification helpers for DS-E2E-12/13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeComposition as PlatformScenarioRuntimeComposition,
)
from intergrax.llm_adapters.registry.profile import llm_profile_from_env
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.tools.registry import ToolRegistry

from intergrax.runtime.decision_flow import DecisionFlowScope
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    ScenarioRuntimeComposition,
    build_scenario_environment_profile,
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
from testing_support.decision_e2e.provider_binding import (
    QualificationProviderBinding,
    bind_qualification_llm_profile,
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
MIN_CROSS_SCENARIO_DECISION_SCENARIOS = 2


def resolve_canonical_runtime_modules(
    composition: PlatformScenarioRuntimeComposition,
) -> frozenset[str]:
    decision_gate = composition.nexus_loop.peek_decision_flow_gate()
    if decision_gate is None:
        return frozenset()
    if not decision_gate.supports_scope(DecisionFlowScope.GRAPH_FINAL):
        return frozenset()
    return CANONICAL_DECISION_RUNTIME_MODULES


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


def _provider_model_from_binding(binding: QualificationProviderBinding) -> tuple[str, str | None]:
    return binding.resolved_provider, binding.resolved_model


def _scenario_evidence_base(
    *,
    scenario_id: str,
    invocation: str,
    binding: QualificationProviderBinding,
    **kwargs: object,
) -> ScenarioExecutionEvidence:
    return ScenarioExecutionEvidence(
        scenario_id=scenario_id,
        invocation=invocation,
        provider=binding.resolved_provider,
        model=binding.resolved_model,
        requested_provider=binding.requested_provider,
        requested_model=binding.requested_model,
        resolved_provider=binding.resolved_provider,
        resolved_model=binding.resolved_model,
        binding_source=binding.binding_source,
        **kwargs,
    )


async def run_ai_incident_live_qualification(
    *,
    variant: ScenarioVariant = ScenarioVariant.RESOLVED,
) -> ScenarioQualificationAttempt:
    invocation = canonical_reproduction_shell_command()
    scenario_id = AI_INCIDENT_SCENARIO_ID
    environment = build_scenario_environment_profile()
    binding, block_reason = bind_qualification_llm_profile(environment)
    if block_reason is not None or binding is None:
        profile = llm_profile_from_env()
        provider = (
            profile.provider.value
            if isinstance(profile.provider, LLMProvider)
            else str(profile.provider)
        )
        return ScenarioQualificationAttempt(
            evidence=ScenarioExecutionEvidence(
                scenario_id=scenario_id,
                invocation=invocation,
                provider=provider,
                model=profile.model,
                executed=False,
                decision_path_exercised=None,
                used_mock_provider=False,
                block_reason=block_reason or "qualification provider binding failed",
            ),
            evaluation_passed=False,
            error=block_reason,
        )

    provider, model = _provider_model_from_binding(binding)
    try:
        runtime_composition = ScenarioRuntimeComposition(
            environment=environment,
            tool_registry=ToolRegistry(),
        )
        fixture_bundle = build_fixture_runtime_bundle(
            variant=variant,
            runtime_composition=runtime_composition,
        )
        bundle = fixture_bundle.bundle
        composition = bundle.runtime_composition
        runtime_modules = resolve_canonical_runtime_modules(composition.platform)
        adapter = resolve_scenario_llm_adapter(composition.environment)
        if _is_mock_adapter_type(adapter):
            return ScenarioQualificationAttempt(
                evidence=_scenario_evidence_base(
                    scenario_id=scenario_id,
                    invocation=invocation,
                    binding=binding,
                    executed=False,
                    decision_path_exercised=None,
                    used_mock_provider=True,
                    block_reason="Scenario resolved to mock/fixture LLM adapter",
                ),
                evaluation_passed=False,
            )

        if not runtime_modules:
            return ScenarioQualificationAttempt(
                evidence=_scenario_evidence_base(
                    scenario_id=scenario_id,
                    invocation=invocation,
                    binding=binding,
                    executed=False,
                    decision_path_exercised=False,
                    used_mock_provider=False,
                    block_reason="Scenario runtime has no canonical Decision flow gate",
                ),
                evaluation_passed=False,
            )

        try:
            result = await execute_resolved_skeleton(bundle)
            evaluation = evaluate_scenario_run(result, fixture_bundle.fixture)
        except Exception as exc:
            return ScenarioQualificationAttempt(
                evidence=_scenario_evidence_base(
                    scenario_id=scenario_id,
                    invocation=invocation,
                    binding=binding,
                    executed=True,
                    decision_path_exercised=True,
                    used_mock_provider=False,
                    runtime_modules=runtime_modules,
                    block_reason=f"scenario evaluation failed: {type(exc).__name__}",
                ),
                evaluation_passed=False,
                error=str(exc),
            )
    except Exception as exc:
        return ScenarioQualificationAttempt(
            evidence=_scenario_evidence_base(
                scenario_id=scenario_id,
                invocation=invocation,
                binding=binding,
                executed=False,
                decision_path_exercised=None,
                used_mock_provider=False,
                block_reason=f"scenario execution failed: {type(exc).__name__}",
            ),
            evaluation_passed=False,
            error=str(exc),
        )

    return ScenarioQualificationAttempt(
        evidence=_scenario_evidence_base(
            scenario_id=scenario_id,
            invocation=invocation,
            binding=binding,
            executed=True,
            decision_path_exercised=True,
            used_mock_provider=False,
            outcome=result.outcome,
            runtime_modules=runtime_modules,
        ),
        evaluation_passed=evaluation.passed,
        error=None if evaluation.passed else "; ".join(evaluation.failures),
    )


def discover_decision_scenario_roots(repo_root: Path) -> tuple[Path, ...]:
    scenarios_root = repo_root / "platform_proofs" / "scenarios"
    if not scenarios_root.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in scenarios_root.iterdir()
            if path.is_dir() and scenario_exercises_decision_runtime(path)
        )
    )


def discover_decision_scenario_slugs(repo_root: Path) -> tuple[str, ...]:
    return tuple(path.name for path in discover_decision_scenario_roots(repo_root))


def scenario_exercises_decision_runtime(scenario_dir: Path) -> bool:
    runtime_composition = scenario_dir / "application" / "runtime_composition.py"
    if not runtime_composition.is_file():
        return False
    source = runtime_composition.read_text(encoding="utf-8")
    return "DecisionProfile" in source or "peek_decision_flow_gate" in source
