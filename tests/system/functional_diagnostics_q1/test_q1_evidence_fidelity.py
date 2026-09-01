# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q1-R2 evidence fidelity and decision-independence gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceKind
from intergrax.runtime.diagnostics.functional_evidence_persistence import FunctionalEvidenceQueryRequest
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    attach_functional_evidence_recorder,
    FunctionalEvidenceRecorder,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from local_search.rag_functional_evidence import emit_search_functional_evidence
from local_search.retrieval_selection import (
    SearchRetrievalCandidate,
    artifact_ref_from_candidate,
    candidates_from_formatted_evidence,
    select_top_ranked_candidate,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_INSTRUMENTATION = (
    _REPO_ROOT / "agents" / "local_search" / "rag_functional_evidence.py",
    _REPO_ROOT / "agents" / "local_search" / "steps" / "search_job.py",
    _REPO_ROOT / "agents" / "local_synthesizer" / "rag_functional_evidence.py",
    _REPO_ROOT / "agents" / "local_synthesizer" / "steps" / "synthesize_job.py",
)
_SEARCH_SELECTION_MODULE = _REPO_ROOT / "agents" / "local_search" / "retrieval_selection.py"
_DIAGNOSTICS_DIR = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_FORBIDDEN_DIAG_IMPORT_PREFIXES = (
    "intergrax.runtime.diagnostics",
    "functional_diagnostic",
    "functional_diagnostics_q1",
    "qualification.oracle",
)
_FORBIDDEN_SELECTION_EXPORTS = frozenset(
    {
        "top_ranked_artifact_ref",
        "select_top_ranked_candidate",
        "parse_retrieval_evidence_items",
        "candidates_from_formatted_evidence",
    }
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


def test_artifact_ref_uses_chunk_id_not_path_heuristic() -> None:
    candidate = SearchRetrievalCandidate(
        chunk_id="qdrant-chunk-42",
        source_path="/cert-fixtures/workspace/operations-decoy.md",
        score=0.9,
    )
    assert artifact_ref_from_candidate(candidate) == "chunk:qdrant-chunk-42"


def test_top_ranked_selection_is_first_candidate() -> None:
    candidates = candidates_from_formatted_evidence(
        [
            {"chunk_id": "decoy-1", "source_path": "/x/operations-decoy.md", "score": 0.99},
            {"chunk_id": "incident-1", "source_path": "/x/incident-report.md", "score": 0.5},
        ],
    )
    selection = select_top_ranked_candidate(candidates)
    assert selection.selected_artifact_ref == "chunk:decoy-1"


def test_emit_search_records_actual_selection_not_heuristic() -> None:
    persistence = InMemoryFunctionalEvidencePersistence(cursor_secret=b"x" * 32)
    recorder = FunctionalEvidenceRecorder(persistence)
    task_id = mint_task_id()
    run_id = mint_run_id()
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="local_search",
        request=RuntimeRequest(
            agent_id="local_search",
            tenant_id="tenant-q1",
            user_id="u1",
            session_id="s1",
            task_id=task_id,
            run_id=run_id,
            message="q",
            metadata={},
        ),
    )
    attach_functional_evidence_recorder(exec_ctx, recorder)
    candidates = candidates_from_formatted_evidence(
        [
            {"chunk_id": "decoy-1", "source_path": "/x/operations-decoy.md"},
            {"chunk_id": "incident-1", "source_path": "/x/incident-report.md"},
        ],
    )
    emit_search_functional_evidence(
        exec_ctx,
        metadata={},
        candidates=candidates,
        selected_artifact_ref="chunk:decoy-1",
        retrieve_succeeded=True,
    )
    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id="tenant-q1",
            task_id=exec_ctx.task_id,
            run_id=exec_ctx.run_id,
            page_size=50,
        ),
    )
    selected = next(
        item.selection.selected_artifact_ref.artifact_ref
        for item in page.items
        if item.kind is PipelineEvidenceKind.SELECTION and item.selection is not None
    )
    assert selected == "chunk:decoy-1"


@pytest.mark.parametrize("path", _PRODUCTION_INSTRUMENTATION)
def test_production_instrumentation_has_no_qualification_force_hooks(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert "qualification_force_selection" not in source
    assert "qualification_selected_artifact" not in source
    assert "qualification_draft_override" not in source


def test_production_instrumentation_does_not_import_qualification_oracle() -> None:
    for path in _PRODUCTION_INSTRUMENTATION:
        for module in _collect_import_modules(path):
            assert "functional_diagnostics_q1" not in module
            assert "qualification.oracle" not in module


def test_search_selection_module_has_no_diagnostics_dependency() -> None:
    for module in _collect_import_modules(_SEARCH_SELECTION_MODULE):
        for prefix in _FORBIDDEN_DIAG_IMPORT_PREFIXES:
            assert not module.startswith(prefix), f"{_SEARCH_SELECTION_MODULE.name} imports {module}"


def test_search_job_does_not_import_diagnostics_selection_helpers() -> None:
    search_job = _REPO_ROOT / "agents" / "local_search" / "steps" / "search_job.py"
    for module in _collect_import_modules(search_job):
        assert module != "intergrax.runtime.diagnostics.c1_retrieval_evidence"
        assert "top_ranked_artifact_ref" not in search_job.read_text(encoding="utf-8")


def test_synthesizer_does_not_reconstruct_selection_via_diagnostics() -> None:
    synthesize_job = _REPO_ROOT / "agents" / "local_synthesizer" / "steps" / "synthesize_job.py"
    source = synthesize_job.read_text(encoding="utf-8")
    for module in _collect_import_modules(synthesize_job):
        assert module != "intergrax.runtime.diagnostics.c1_retrieval_evidence"
    assert "top_ranked_artifact_ref" not in source
    assert "parse_retrieval_evidence_items" not in source


def test_diagnostics_does_not_export_search_selection_policy() -> None:
    violations: list[str] = []
    for path in sorted(_DIAGNOSTICS_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_SELECTION_EXPORTS:
            if f"def {symbol}" in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)} defines {symbol}")
    assert violations == [], "; ".join(violations)


def test_decision_diagnostics_independence() -> None:
    """Search selection can run without importing or calling DIAG modules."""
    for module in _collect_import_modules(_SEARCH_SELECTION_MODULE):
        for prefix in _FORBIDDEN_DIAG_IMPORT_PREFIXES:
            assert not module.startswith(prefix), f"search selection imports {module}"
    candidates = candidates_from_formatted_evidence([{"chunk_id": "incident-1"}])
    selection = select_top_ranked_candidate(candidates)
    assert selection.selected_artifact_ref == "chunk:incident-1"
