# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q1-R1 evidence fidelity and architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.diagnostics.c1_retrieval_evidence import (
    RetrievalEvidenceItem,
    artifact_ref_from_retrieval_item,
    parse_retrieval_evidence_items,
    top_ranked_artifact_ref,
)
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

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_INSTRUMENTATION = (
    _REPO_ROOT / "agents" / "local_search" / "rag_functional_evidence.py",
    _REPO_ROOT / "agents" / "local_search" / "steps" / "search_job.py",
    _REPO_ROOT / "agents" / "local_synthesizer" / "rag_functional_evidence.py",
    _REPO_ROOT / "agents" / "local_synthesizer" / "steps" / "synthesize_job.py",
)


def test_artifact_ref_uses_chunk_id_not_path_heuristic() -> None:
    item = RetrievalEvidenceItem(
        chunk_id="qdrant-chunk-42",
        source_path="/cert-fixtures/workspace/operations-decoy.md",
        score=0.9,
    )
    assert artifact_ref_from_retrieval_item(item) == "chunk:qdrant-chunk-42"


def test_top_ranked_selection_is_first_candidate() -> None:
    items = parse_retrieval_evidence_items(
        [
            {"chunk_id": "decoy-1", "source_path": "/x/operations-decoy.md", "score": 0.99},
            {"chunk_id": "incident-1", "source_path": "/x/incident-report.md", "score": 0.5},
        ],
    )
    assert top_ranked_artifact_ref(items) == "chunk:decoy-1"


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
    items = parse_retrieval_evidence_items(
        [
            {"chunk_id": "decoy-1", "source_path": "/x/operations-decoy.md"},
            {"chunk_id": "incident-1", "source_path": "/x/incident-report.md"},
        ],
    )
    emit_search_functional_evidence(
        exec_ctx,
        metadata={},
        evidence_items=items,
        actual_selected_artifact_ref="chunk:decoy-1",
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
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "functional_diagnostics_q1" not in alias.name
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "functional_diagnostics_q1" not in node.module
                assert "qualification.oracle" not in node.module
