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
from testing_support.decision_e2e.qualification_evidence import ScenarioExecutionEvidence
from testing_support.decision_e2e.requirements import qualify_live_scenario
from testing_support.decision_e2e.scenario_qualification import (
    AI_INCIDENT_SCENARIO_ID,
    CANONICAL_DECISION_RUNTIME_MODULES,
    discover_decision_scenario_roots,
    run_ai_incident_live_qualification,
    scenario_exercises_decision_runtime,
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

    decision_scenarios = tuple(
        path
        for path in discover_decision_scenario_roots(_REPO_ROOT)
        if scenario_exercises_decision_runtime(path)
    )
    if len(decision_scenarios) < 2:
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_13,
                disposition=QualificationDisposition.BLOCKED,
                evidence=(),
                reason=(
                    "Cross-scenario qualification requires two live platform proofs exercising "
                    "canonical Decision runtime; only ai_incident_investigation currently qualifies"
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

    scenario_a = await run_ai_incident_live_qualification()
    scenario_b_evidence = ScenarioExecutionEvidence(
        scenario_id=decision_scenarios[1].name,
        invocation="pending-second-scenario-runner",
        provider=scenario_a.evidence.provider,
        model=scenario_a.evidence.model,
        executed=False,
        decision_path_exercised=True,
        used_mock_provider=False,
        block_reason="Second Decision platform proof runner not yet implemented",
        runtime_modules=CANONICAL_DECISION_RUNTIME_MODULES,
    )
    result = qualify_live_scenario(
        proof_id=DecisionE2EProofId.DS_E2E_13,
        scenario_evidence=scenario_a.evidence,
        reason=(
            f"scenario_a={AI_INCIDENT_SCENARIO_ID}; "
            f"runtime_modules={','.join(sorted(CANONICAL_DECISION_RUNTIME_MODULES))}"
        ),
    )
    if result.disposition is QualificationDisposition.PASSED:
        result = DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_13,
            disposition=QualificationDisposition.BLOCKED,
            evidence=result.evidence,
            reason=scenario_b_evidence.block_reason,
        )
    decision_e2e_report_collector.record(result)
