# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q4 evidence fidelity and decision-independence gates."""

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
from model_routing_qualifier.model_routing import build_profile_a, candidates_from_profiles
from model_routing_qualifier.model_routing_functional_evidence import emit_model_routing_functional_evidence
from model_routing_qualifier.routing_profile import build_q4_qualification_routing_profile
from model_routing_qualifier.model_routing import build_profile_b, build_invoke_fail_profile, artifact_ref_for_profile

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_INSTRUMENTATION = (
    _REPO_ROOT / "agents" / "model_routing_qualifier" / "model_routing_functional_evidence.py",
    _REPO_ROOT / "agents" / "model_routing_qualifier" / "steps" / "model_routing_job.py",
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


def test_model_routing_job_has_no_qualification_force_hooks() -> None:
    source = (_REPO_ROOT / "agents" / "model_routing_qualifier" / "steps" / "model_routing_job.py").read_text(
        encoding="utf-8",
    )
    assert "qualification_force_profile" not in source
    assert "qualification_selected_profile" not in source


@pytest.mark.parametrize("path", _PRODUCTION_INSTRUMENTATION)
def test_production_instrumentation_does_not_import_qualification_oracle(path: Path) -> None:
    for module in _collect_import_modules(path):
        assert "functional_diagnostics_q4" not in module
        assert "qualification.oracle" not in module


def test_emit_model_routing_functional_evidence_records_generic_kinds() -> None:
    persistence = InMemoryFunctionalEvidencePersistence()
    recorder = FunctionalEvidenceRecorder(
        persistence,
        producer_component="agents.model_routing_qualifier",
    )
    profile_a = build_profile_a()
    profile_b = build_profile_b()
    routing = build_q4_qualification_routing_profile(
        profile_a=profile_a,
        profile_b=profile_b,
        invoke_fail_profile=build_invoke_fail_profile(),
    )
    candidates = candidates_from_profiles(routing.allowed_profiles)
    selected_ref = artifact_ref_for_profile(profile_a)
    task_id = mint_task_id()
    run_id = mint_run_id()
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="model_routing_qualifier",
        request=RuntimeRequest(
            agent_id="model_routing_qualifier",
            user_id="user",
            session_id="sess",
            tenant_id="tenant",
            task_id=task_id,
            run_id=run_id,
            message="test",
        ),
    )
    attach_functional_evidence_recorder(exec_ctx, recorder)
    emit_model_routing_functional_evidence(
        exec_ctx,
        metadata={},
        candidates=candidates,
        selected_profile_ref=selected_ref,
        invoke_succeeded=True,
        raw_model_output="42",
    )
    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id="tenant",
            task_id=task_id,
            run_id=run_id,
        ),
    )
    kinds = {item.kind for item in page.items}
    assert PipelineEvidenceKind.CANDIDATE_RANK in kinds
    assert PipelineEvidenceKind.SELECTION in kinds
    assert PipelineEvidenceKind.OPERATION_OUTCOME in kinds
    assert PipelineEvidenceKind.OUTPUT_RELATION in kinds
