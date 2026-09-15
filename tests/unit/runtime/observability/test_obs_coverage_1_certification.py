# © Artur Czarnecki. All rights reserved.

"""OBS-COVERAGE-1 — platform evidence coverage certification (gates + proof map)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.functional_evidence.models import PipelineEvidenceScope
from intergrax.runtime.observability.causal_evidence import RuntimeExecutionRef
from intergrax.runtime.events.runtime_event import RuntimeEvent
from scripts.proof.scenario_architecture_conformance import (
    assert_all_initialized_scenario_architectures,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NEXUS_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_RUNTIME_ROOT = _REPO_ROOT / "intergrax" / "runtime"
_EVENT_OBS_MINT_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime" / "events",
    _REPO_ROOT / "intergrax" / "runtime" / "observability",
)
_CONFORMANCE_EXCLUDED = frozenset({"persistence_conformance.py", "functional_evidence_persistence_conformance.py"})
_EXCLUDED_PARTS = frozenset({"__pycache__", "tests"})


@dataclass(frozen=True, slots=True)
class CoveragePathProof:
    path_id: str
    semantic_scope: str
    required_identity: str
    proof_modules: tuple[str, ...]
    verdict: str


P1_PATHS = frozenset(
    {
        "P1_root_terminal_success",
        "P1_root_terminal_failure",
        "P2_graph_child_execution",
        "P3_retry_multiple_attempts",
        "P4_delegated_child_execution",
        "P5_background_async_execution",
        "P6_transport_causal_evidence",
        "P7_scenario_canonical_execution",
        "P8_functional_evidence_execution_scope",
        "P9_terminal_before_diagnostic_dispatch",
        "P10_failure_factual_evidence",
    }
)

COVERAGE_PATH_PROOFS: tuple[CoveragePathProof, ...] = (
    CoveragePathProof(
        path_id="P1_root_terminal_success",
        semantic_scope="execution-scoped",
        required_identity="tenant, task, run, attempt, execution",
        proof_modules=(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "tests/integration/runtime/test_terminal_diagnostic_production_e2e.py",
        ),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P1_root_terminal_failure",
        semantic_scope="execution-scoped",
        required_identity="tenant, task, run, attempt, execution",
        proof_modules=(
            "tests/unit/runtime/execution/test_execution_failure_evidence_r2.py",
            "tests/unit/runtime/execution/test_execution_failure_evidence_r2_correction.py",
        ),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P2_graph_child_execution",
        semantic_scope="execution-scoped",
        required_identity="distinct ExecutionId per graph node",
        proof_modules=(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "tests/unit/runtime/execution/lineage/test_execution_lineage_admission_order.py",
        ),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P3_retry_multiple_attempts",
        semantic_scope="execution-scoped",
        required_identity="attempt + execution per redelivery",
        proof_modules=("tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P4_delegated_child_execution",
        semantic_scope="execution-scoped + lineage",
        required_identity="parent and child ExecutionId",
        proof_modules=(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "tests/unit/runtime/execution/lineage/test_host_task_resume_lineage_identity.py",
            "tests/unit/runtime/execution/test_delegated_execution_status.py",
        ),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P5_background_async_execution",
        semantic_scope="execution-scoped",
        required_identity="BackgroundExecutionIdentity five-ID",
        proof_modules=(
            "tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py",
            "tests/unit/queueing/test_document_store_task_queue.py",
        ),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P6_transport_causal_evidence",
        semantic_scope="execution-scoped causal target",
        required_identity="RuntimeExecutionRef five-ID",
        proof_modules=("tests/unit/runtime/observability/test_causal_evidence_contract.py",),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P7_scenario_canonical_execution",
        semantic_scope="host-scoped / execution via baseline",
        required_identity="no scenario-local execution authority",
        proof_modules=("scripts/proof/scenario_architecture_conformance.py",),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P8_functional_evidence_execution_scope",
        semantic_scope="execution-scoped",
        required_identity="PipelineEvidenceScope five-ID",
        proof_modules=(
            "tests/unit/runtime/architecture/test_obs_functional_evidence_contract_boundary.py",
            "tests/unit/runtime/observability/test_functional_evidence_r1_wiring.py",
        ),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P9_terminal_before_diagnostic_dispatch",
        semantic_scope="ordering invariant",
        required_identity="terminal RuntimeEvent then DIAG port",
        proof_modules=(
            "tests/unit/runtime/architecture/test_obs_diag_port_1_gates.py",
            "tests/unit/runtime/observability/test_obs_coverage_1_certification.py",
        ),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P10_failure_factual_evidence",
        semantic_scope="execution-scoped",
        required_identity="failure request carries canonical IDs",
        proof_modules=("tests/unit/runtime/execution/test_execution_failure_evidence_r2.py",),
        verdict="PROVEN",
    ),
    CoveragePathProof(
        path_id="P11_long_running_checkpoint",
        semantic_scope="execution-scoped",
        required_identity="resume plan + lineage (same or new execution per UEA)",
        proof_modules=(
            "tests/unit/runtime/execution/lineage/test_host_task_resume_lineage_identity.py",
        ),
        verdict="PARTIAL",
    ),
    CoveragePathProof(
        path_id="P12_hitl_interrupt",
        semantic_scope="execution-scoped",
        required_identity="pause within attempt unless UEA defines fork",
        proof_modules=("tests/unit/applications/test_idt_fix_c_hitl_approver_provenance.py",),
        verdict="PARTIAL",
    ),
    CoveragePathProof(
        path_id="DG005_cross_topology_runtime_event",
        semantic_scope="transport-scoped qualification",
        required_identity="shared persistence across HTTP host and worker",
        proof_modules=("docs/project/maintainers/qualification/DIAGNOSTIC_GAP_LEDGER.md",),
        verdict="NOT PROVEN",
    ),
    CoveragePathProof(
        path_id="NON_EXECUTION_host_bootstrap_failure",
        semantic_scope="non-execution",
        required_identity="explicit subject without ExecutionId",
        proof_modules=("docs/project/maintainers/qualification/DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md",),
        verdict="PROVEN",
    ),
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _event_obs_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _EVENT_OBS_MINT_ROOTS:
        for path in root.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            if path.name in _CONFORMANCE_EXCLUDED:
                continue
            files.append(path)
    return files


def _collect_forbidden_mint_calls() -> list[str]:
    violations: list[str] = []
    for path in _event_obs_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name == "mint_execution_id":
                violations.append(f"{rel}:{node.lineno} calls mint_execution_id")
    return violations


def _collect_runtime_event_without_execution_id() -> list[str]:
    violations: list[str] = []
    for path in _RUNTIME_ROOT.rglob("*.py"):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node.func) != "RuntimeEvent":
                continue
            has_execution_id = any(kw.arg == "execution_id" for kw in node.keywords if kw.arg)
            if not has_execution_id:
                for kw in node.keywords:
                    if kw.arg is None and isinstance(kw.value, ast.Call):
                        if _call_name(kw.value.func) == "runtime_event_identity_kwargs":
                            has_execution_id = True
                            break
            if not has_execution_id:
                violations.append(f"{rel}:{node.lineno} RuntimeEvent(...) missing execution_id")
    return violations


@pytest.mark.parametrize("entry", COVERAGE_PATH_PROOFS, ids=lambda e: e.path_id)
def test_coverage_proof_modules_exist(entry: CoveragePathProof) -> None:
    for rel in entry.proof_modules:
        path = _REPO_ROOT / rel
        assert path.is_file(), f"missing proof module for {entry.path_id}: {rel}"


def test_p1_critical_paths_are_proven() -> None:
    by_id = {entry.path_id: entry for entry in COVERAGE_PATH_PROOFS}
    missing = sorted(P1_PATHS - by_id.keys())
    assert missing == []
    not_proven = sorted(
        path_id
        for path_id in P1_PATHS
        if by_id[path_id].verdict != "PROVEN"
    )
    assert not_proven == []


def test_gate_obs_does_not_mint_execution_id() -> None:
    assert _collect_forbidden_mint_calls() == []


def test_gate_runtime_event_constructions_include_execution_id() -> None:
    assert _collect_runtime_event_without_execution_id() == []


def test_gate_runtime_event_execution_id_field_required() -> None:
    assert RuntimeEvent.model_fields["execution_id"].is_required()


def test_gate_pipeline_evidence_scope_requires_execution_id() -> None:
    assert "execution_id" in PipelineEvidenceScope.model_fields
    assert PipelineEvidenceScope.model_fields["execution_id"].is_required()


def test_gate_runtime_execution_ref_requires_execution_id() -> None:
    assert "execution_id" in RuntimeExecutionRef.model_fields
    assert RuntimeExecutionRef.model_fields["execution_id"].is_required()


def test_gate_initialized_scenarios_do_not_bypass_canonical_execution() -> None:
    assert_all_initialized_scenario_architectures(_REPO_ROOT)


def test_gate_terminal_runtime_event_published_before_diagnostic_dispatch() -> None:
    source = _NEXUS_LOOP.read_text(encoding="utf-8")
    publish_idx = source.index("terminal_event = await self._events.publish_terminal(task)")
    dispatch_idx = source.index(
        "self._terminal_diagnostic_trigger.dispatch_terminal_execution(",
        publish_idx,
    )
    assert publish_idx < dispatch_idx


def test_gate_event_bus_persists_before_handlers() -> None:
    bus_path = _REPO_ROOT / "intergrax" / "runtime" / "events" / "event_bus.py"
    source = bus_path.read_text(encoding="utf-8")
    publish_block = source.split("async def publish", 1)[1].split("\n    @property", 1)[0]
    commit_idx = publish_block.index("self._commit_durable_evidence(event)")
    dispatch_idx = publish_block.index("await self._dispatch_handlers_async(event)")
    assert commit_idx < dispatch_idx
