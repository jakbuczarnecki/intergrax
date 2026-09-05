# © Artur Czarnecki. All rights reserved.

"""DS-E2E-13 — cross-scenario no special casing gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.environment import qualification_required
from testing_support.decision_e2e.requirements import qualify_cross_scenario_dual
from testing_support.decision_e2e.scenario_qualification import (
    AI_INCIDENT_SCENARIO_ID,
    CANONICAL_DECISION_RUNTIME_MODULES,
    MIN_CROSS_SCENARIO_DECISION_SCENARIOS,
    ScenarioQualificationAttempt,
    discover_decision_scenario_slugs,
    run_ai_incident_live_qualification,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.no_ci,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DECISION_RUNTIME_ROOT = _REPO_ROOT / "intergrax" / "runtime"
_FORBIDDEN_SCENARIO_TOKENS = (
    "ai_incident_investigation",
    "verified_product_identification",
    "scenario_name",
    "proof_id",
    "if E2E",
    "if TESTING",
)
_SECOND_SCENARIO = _REPO_ROOT / "platform_proofs" / "scenarios" / "verified_product_identification"


def _iter_decision_runtime_sources() -> list[Path]:
    return [
        path
        for path in _DECISION_RUNTIME_ROOT.rglob("*.py")
        if path.is_file() and "decision" in path.name
    ]


def _contains_scenario_branch(source: str) -> list[str]:
    hits: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in (*node.comparators, node.left):
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                    for token in _FORBIDDEN_SCENARIO_TOKENS:
                        if token in operand.value:
                            hits.append(operand.value)
    return hits


def test_ds_e2e_13_ast_gate_no_scenario_branching() -> None:
    scenario_roots = (
        _REPO_ROOT / "platform_proofs" / "scenarios" / "ai_incident_investigation",
        _SECOND_SCENARIO,
    )
    for root in scenario_roots:
        assert root.is_dir()

    violations: list[str] = []
    for path in _iter_decision_runtime_sources():
        source = path.read_text(encoding="utf-8")
        hits = _contains_scenario_branch(source)
        if hits:
            violations.append(f"{path}: {sorted(set(hits))}")
    assert not violations


@pytest.mark.asyncio
async def test_ds_e2e_13_cross_scenario_qualification(
    decision_e2e_report_collector,
) -> None:
    violations: list[str] = []
    for path in _iter_decision_runtime_sources():
        source = path.read_text(encoding="utf-8")
        hits = _contains_scenario_branch(source)
        if hits:
            violations.append(f"{path}: {sorted(set(hits))}")

    if violations:
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_13,
                disposition=QualificationDisposition.FAILED,
                evidence=(),
                reason="; ".join(violations),
            ),
        )
        pytest.fail("; ".join(violations))

    decision_scenarios = discover_decision_scenario_slugs(_REPO_ROOT)
    if not decision_scenarios:
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_13,
                disposition=QualificationDisposition.BLOCKED,
                evidence=(),
                reason=(
                    "Cross-scenario qualification requires a live platform proof exercising "
                    "canonical Decision runtime"
                ),
            ),
        )
        return

    if len(decision_scenarios) < MIN_CROSS_SCENARIO_DECISION_SCENARIOS:
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_13,
                disposition=QualificationDisposition.BLOCKED,
                evidence=(),
                reason=(
                    "Cross-scenario qualification requires "
                    f"{MIN_CROSS_SCENARIO_DECISION_SCENARIOS} distinct live Decision scenarios; "
                    f"found {len(decision_scenarios)} ({', '.join(decision_scenarios)})"
                ),
            ),
        )
        return

    if not qualification_required():
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_13,
                disposition=QualificationDisposition.BLOCKED,
                evidence=(),
                reason="INTERGRAX_DECISION_E2E_QUALIFICATION not enabled for live scenario runs",
            ),
        )
        return

    scenario_a_slug, scenario_b_slug = decision_scenarios[0], decision_scenarios[1]

    async def _run_live_qualification(slug: str) -> ScenarioQualificationAttempt | None:
        if slug == AI_INCIDENT_SCENARIO_ID:
            return await run_ai_incident_live_qualification()
        return None

    scenario_a_attempt = await _run_live_qualification(scenario_a_slug)
    scenario_b_attempt = await _run_live_qualification(scenario_b_slug)
    if scenario_a_attempt is None or scenario_b_attempt is None:
        missing = [
            slug
            for slug, attempt in (
                (scenario_a_slug, scenario_a_attempt),
                (scenario_b_slug, scenario_b_attempt),
            )
            if attempt is None
        ]
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_13,
                disposition=QualificationDisposition.BLOCKED,
                evidence=(),
                reason=(
                    "Cross-scenario qualification missing live runner for: "
                    + ", ".join(missing)
                ),
            ),
        )
        return

    scenario_a = scenario_a_attempt
    scenario_b = scenario_b_attempt
    result = qualify_cross_scenario_dual(
        scenario_a=scenario_a.evidence,
        scenario_b=scenario_b.evidence,
        reason=(
            f"scenario_a={scenario_a.evidence.scenario_id}; "
            f"scenario_b={scenario_b.evidence.scenario_id}; "
            f"runtime_modules={','.join(sorted(CANONICAL_DECISION_RUNTIME_MODULES))}"
        ),
    )
    if result.disposition is QualificationDisposition.PASSED and (
        not scenario_a.evaluation_passed or not scenario_b.evaluation_passed
    ):
        failures = []
        if not scenario_a.evaluation_passed:
            failures.append(
                f"{scenario_a.evidence.scenario_id}: {scenario_a.error or 'evaluation failed'}",
            )
        if not scenario_b.evaluation_passed:
            failures.append(
                f"{scenario_b.evidence.scenario_id}: {scenario_b.error or 'evaluation failed'}",
            )
        result = DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_13,
            disposition=QualificationDisposition.FAILED,
            evidence=result.evidence,
            reason="; ".join(failures),
        )
    if result.disposition is QualificationDisposition.FAILED:
        decision_e2e_report_collector.record(result)
        pytest.fail(result.reason or "cross-scenario qualification failed")
    if result.disposition is QualificationDisposition.BLOCKED:
        decision_e2e_report_collector.record(result)
        pytest.fail(result.reason or "cross-scenario qualification blocked")
    decision_e2e_report_collector.record(result)
