# © Artur Czarnecki. All rights reserved.

"""UE-11G-C1-R4 functional diagnosis gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    CHECK_C1_SELECTION,
    CHECK_C1_VALIDATION,
)
from tests.system.unified_execution.proof_runner.functional_diagnosis import (
    DiagnosticCheckProjection,
    FunctionalDiagnosisReport,
    evaluate_r4_result,
    failure_stage_for_check,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERIC_DIAGNOSTIC_DIR = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_FORBIDDEN_IMPORT_PREFIXES = (
    "applications.local_workspace_application",
    "agents.local_search",
    "tests.",
    "testing_support",
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


@pytest.mark.parametrize(
    "module_name",
    (
        "functional_diagnostic_analyzer.py",
        "functional_diagnostic_specification.py",
        "functional_evidence.py",
        "functional_evidence_persistence.py",
    ),
)
def test_generic_functional_diagnostics_have_no_application_imports(module_name: str) -> None:
    imports = _module_imports(_GENERIC_DIAGNOSTIC_DIR / module_name)
    for module in imports:
        for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
            assert not module.startswith(forbidden), f"{module_name} imports {module}"


def test_failure_stage_mapping_is_typed() -> None:
    assert failure_stage_for_check(CHECK_C1_SELECTION) == "SELECTION"
    assert failure_stage_for_check(CHECK_C1_VALIDATION) == "OUTPUT VALIDATION"
    assert failure_stage_for_check(None) is None


def test_r4_pass_when_oracle_fails_and_diagnosis_proven() -> None:
    diagnosis = FunctionalDiagnosisReport(
        invocation_status="PASS",
        persistence_backend="DocumentStore via qualification API",
        durable=True,
        evidence_kinds=("selection",),
        evidence_count=3,
        validation_id="validation-1",
        functional_expected="2026-08-17",
        functional_actual_bounded="unknown date",
        diagnostic_specification_id="spec-1",
        diagnostic_specification_version=1,
        diagnostic_first_proven_failure=str(CHECK_C1_SELECTION),
        diagnostic_check_results=(
            DiagnosticCheckProjection(
                check_id=str(CHECK_C1_SELECTION),
                status="proven_fail",
                factual_claim="Wrong retrieval selection was recorded.",
            ),
        ),
        diagnostic_supporting_evidence_refs=("evidence-1",),
        diagnostic_limitations=(),
        failure_stage="SELECTION",
        confidence="PROVEN",
    )
    assert (
        evaluate_r4_result(
            search_completed=True,
            oracle_pass=False,
            diagnosis=diagnosis,
        )
        == "PASS"
    )


def test_r4_partial_when_oracle_passes() -> None:
    diagnosis = FunctionalDiagnosisReport(
        invocation_status="PASS",
        persistence_backend="DocumentStore via qualification API",
        durable=True,
        evidence_kinds=("selection",),
        evidence_count=1,
        validation_id="validation-1",
        functional_expected="2026-08-17",
        functional_actual_bounded="2026-08-17",
        diagnostic_specification_id="spec-1",
        diagnostic_specification_version=1,
        diagnostic_first_proven_failure=None,
        diagnostic_check_results=(),
        diagnostic_supporting_evidence_refs=(),
        diagnostic_limitations=(),
        failure_stage=None,
        confidence="INSUFFICIENT",
    )
    assert (
        evaluate_r4_result(
            search_completed=True,
            oracle_pass=True,
            diagnosis=diagnosis,
        )
        == "PARTIAL"
    )
