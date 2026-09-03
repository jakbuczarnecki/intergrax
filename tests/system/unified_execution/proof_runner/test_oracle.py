# © Artur Czarnecki. All rights reserved.

"""Unit tests for UE-11G-C1 expected-fact functional oracle."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.system.unified_execution.proof_runner.contracts import (
    LkwEvidenceSlice,
    LkwRunResponse,
)
from tests.system.unified_execution.proof_runner.oracle import (
    c1_expected_fact_oracle,
    expected_fact,
    functional_oracle_passes,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ORACLE_MODULE = _REPO_ROOT / "tests" / "system" / "unified_execution" / "proof_runner" / "oracle.py"
_FORBIDDEN_IMPORT_PREFIXES = (
    "agents.local_search",
    "intergrax.runtime.diagnostics",
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


def _response(
    *,
    answer: str | None,
    source_refs: list[str] | None = None,
) -> LkwRunResponse:
    diagnostics: dict[str, object] = {}
    if source_refs is not None:
        diagnostics["lkw.search_summary.v1"] = {
            "evidence_count": len(source_refs),
            "num_results": len(source_refs),
            "source_refs": source_refs,
        }
    evidence = LkwEvidenceSlice(
        schema_version="lkw.evidence.v1",
        diagnostics=diagnostics,
    )
    return LkwRunResponse(
        task_id="task-1",
        run_id="run-1",
        state="completed",
        answer=answer,
        lkw_evidence=evidence if source_refs is not None else None,
    )


def test_expected_fact_constant() -> None:
    assert expected_fact() == "2026-08-17"


def test_oracle_passes_when_answer_contains_exact_fact() -> None:
    assert functional_oracle_passes(_response(answer="Incident Orion occurred on 2026-08-17."))


def test_oracle_fails_when_answer_mentions_orion_without_date() -> None:
    assert not functional_oracle_passes(
        _response(answer="Incident Orion was an operational incident."),
    )


def test_oracle_fails_when_incident_source_evidence_present_without_date() -> None:
    assert not functional_oracle_passes(
        _response(
            answer="Incident Orion was an operational incident.",
            source_refs=["/fixtures/incident-report.md"],
        ),
    )


def test_oracle_fails_when_answer_empty() -> None:
    assert not functional_oracle_passes(_response(answer=None))
    assert not functional_oracle_passes(_response(answer=""))


def test_oracle_fails_when_answer_contains_wrong_date() -> None:
    assert not functional_oracle_passes(
        _response(answer="Incident Orion occurred on 2026-07-01."),
    )


def test_evidence_does_not_affect_oracle_outcome() -> None:
    incorrect = "Incident Orion was an operational incident."
    without_evidence = functional_oracle_passes(_response(answer=incorrect))
    with_evidence = functional_oracle_passes(
        _response(
            answer=incorrect,
            source_refs=["/fixtures/incident-report.md"],
        ),
    )
    assert without_evidence is False
    assert with_evidence is False
    assert without_evidence == with_evidence


def test_expected_fact_oracle_is_pluginable() -> None:
    oracle = c1_expected_fact_oracle()
    assert oracle.passes(answer="On 2026-08-17 the incident happened.")
    assert not oracle.passes(answer="No date here.")


def test_oracle_module_has_no_forbidden_imports() -> None:
    imports = _module_imports(_ORACLE_MODULE)
    for module in imports:
        for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
            assert not module.startswith(forbidden), f"oracle imports {module}"
