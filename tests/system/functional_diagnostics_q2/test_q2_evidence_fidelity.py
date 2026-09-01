# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q2 evidence fidelity and decision-independence gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.functional_evidence_persistence import FunctionalEvidenceQueryRequest
from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceKind
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.observability.functional_evidence_recorder import (
    FunctionalEvidenceRecorder,
    attach_functional_evidence_recorder,
)
from tool_selection_qualifier.tool_functional_evidence import emit_tool_selection_functional_evidence
from tool_selection_qualifier.tool_selection import candidates_from_tool_ids

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_INSTRUMENTATION = (
    _REPO_ROOT / "agents" / "tool_selection_qualifier" / "tool_functional_evidence.py",
    _REPO_ROOT / "agents" / "tool_selection_qualifier" / "steps" / "tool_selection_job.py",
)
_FORBIDDEN_DIAG_IMPORT_PREFIXES = (
    "intergrax.runtime.diagnostics",
    "functional_diagnostic",
    "functional_diagnostics_q2",
    "qualification.oracle",
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


def test_tool_selection_job_has_no_qualification_force_hooks() -> None:
    source = (_REPO_ROOT / "agents" / "tool_selection_qualifier" / "steps" / "tool_selection_job.py").read_text(
        encoding="utf-8",
    )
    assert "qualification_force_tool" not in source
    assert "qualification_selected_tool" not in source


@pytest.mark.parametrize("path", _PRODUCTION_INSTRUMENTATION)
def test_production_instrumentation_does_not_import_qualification_oracle(path: Path) -> None:
    for module in _collect_import_modules(path):
        assert "functional_diagnostics_q2" not in module
        assert "qualification.oracle" not in module


def test_emit_tool_selection_functional_evidence_records_generic_kinds() -> None:
    persistence = InMemoryFunctionalEvidencePersistence()
    recorder = FunctionalEvidenceRecorder(
        persistence,
        producer_component="agents.tool_selection_qualifier",
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="tool_selection_qualifier",
        request=RuntimeRequest(
            agent_id="tool_selection_qualifier",
            tenant_id="tenant-q2",
            user_id="user-q2",
            session_id="session-q2",
            task_id=task_id,
            run_id=run_id,
            message="find incident",
            metadata={},
        ),
    )
    attach_functional_evidence_recorder(exec_ctx, recorder)
    candidates = candidates_from_tool_ids(("workspace.search", "workspace.write_file"))
    emit_tool_selection_functional_evidence(
        exec_ctx,
        metadata={},
        candidates=candidates,
        selected_tool_id="workspace.search",
        invoke_succeeded=True,
    )
    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id="tenant-q2",
            task_id=task_id,
            run_id=run_id,
        ),
    )
    kinds = {item.kind for item in page.items}
    assert PipelineEvidenceKind.CANDIDATE_RANK in kinds
    assert PipelineEvidenceKind.SELECTION in kinds
    assert PipelineEvidenceKind.OPERATION_OUTCOME in kinds
