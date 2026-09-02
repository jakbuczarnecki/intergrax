# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q3 evidence fidelity and decision-independence gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceKind
from intergrax.runtime.diagnostics.functional_evidence_persistence import FunctionalEvidenceQueryRequest
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.observability.functional_evidence_recorder import (
    FunctionalEvidenceRecorder,
    attach_functional_evidence_recorder,
)
from web_search_qualifier.web_functional_evidence import emit_web_search_functional_evidence
from web_search_qualifier.web_search import WebSearchCandidate

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_INSTRUMENTATION = (
    _REPO_ROOT / "agents" / "web_search_qualifier" / "web_functional_evidence.py",
    _REPO_ROOT / "agents" / "web_search_qualifier" / "steps" / "web_search_job.py",
)


def _collect_import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_web_search_job_has_no_qualification_force_hooks() -> None:
    source = (_REPO_ROOT / "agents" / "web_search_qualifier" / "steps" / "web_search_job.py").read_text(
        encoding="utf-8",
    )
    assert "qualification_force_source" not in source
    assert "qualification_force_selected_url" not in source
    assert "qualification_expected_source" not in source


@pytest.mark.parametrize("path", _PRODUCTION_INSTRUMENTATION)
def test_production_instrumentation_does_not_import_qualification_oracle(path: Path) -> None:
    for module in _collect_import_modules(path):
        assert "functional_diagnostics_q3" not in module
        assert "qualification.oracle" not in module


def test_emit_web_search_functional_evidence_records_generic_kinds() -> None:
    persistence = InMemoryFunctionalEvidencePersistence()
    recorder = FunctionalEvidenceRecorder(
        persistence,
        producer_component="agents.web_search_qualifier",
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="web_search_qualifier",
        request=RuntimeRequest(
            agent_id="web_search_qualifier",
            tenant_id="tenant-q3",
            user_id="user-q3",
            session_id="session-q3",
            task_id=task_id,
            run_id=run_id,
            message="python release",
            metadata={},
        ),
    )
    attach_functional_evidence_recorder(exec_ctx, recorder)
    candidates = (
        WebSearchCandidate(
            rank=1,
            url="https://www.python.org/downloads/release/python-3120/",
            title="Python 3.12.0",
            snippet="Released Oct 2, 2023",
            provider="tavily",
        ),
    )
    emit_web_search_functional_evidence(
        exec_ctx,
        metadata={},
        actual_query="Python 3.12.0 release date site:python.org",
        search_succeeded=True,
        candidates=candidates,
        selected_url="https://www.python.org/downloads/release/python-3120/",
        extracted_fact="2023-10-02",
    )
    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id="tenant-q3",
            task_id=task_id,
            run_id=run_id,
        ),
    )
    kinds = {item.kind for item in page.items}
    assert PipelineEvidenceKind.CANDIDATE_RANK in kinds
    assert PipelineEvidenceKind.SELECTION in kinds
    assert PipelineEvidenceKind.OPERATION_OUTCOME in kinds
    assert PipelineEvidenceKind.OUTPUT_RELATION in kinds
