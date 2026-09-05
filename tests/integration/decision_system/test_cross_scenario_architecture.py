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


def test_ds_e2e_13_cross_scenario_no_special_casing(
    decision_e2e_report_collector,
) -> None:
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

    disposition = (
        QualificationDisposition.PASSED
        if not violations
        else QualificationDisposition.FAILED
    )
    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_13,
            disposition=disposition,
            evidence=(),
            reason="; ".join(violations) if violations else "no scenario branching in decision runtime",
        ),
    )
    assert not violations
