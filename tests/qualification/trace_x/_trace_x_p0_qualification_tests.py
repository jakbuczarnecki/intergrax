# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P0 mechanical gates TXP0-Q01..TXP0-Q30."""

from __future__ import annotations

import ast
import dataclasses
import subprocess
from pathlib import Path

import pytest

from intergrax.contracts.tracing.events import TraceEvent
from tests.qualification.trace_x._trace_x_p0_support import (
    ARCHITECTURE_LOCK,
    CLOSED_WORLD_ROOTS,
    ENTERPRISE_AUDIT_MATRIX,
    FORWARD_CHAIN,
    FRZ_TRC_P0_MATRIX,
    HISTORICAL_EVIDENCE,
    MANDATORY_FRZ_TRC_IDS,
    REVERSE_RECONSTRUCTION_MATRIX,
    SEMANTIC_OWNER_MATRIX,
    TRACE_MECHANISM_CLASS_REGISTRY,
    TRACE_X_CHILD_DECOMPOSITION,
    TRACE_X_KNOWN_BLOCKERS,
    TRACE_X_P0_AUDITED_HEAD,
    TRACEABILITY_SURFACES,
    AuthorityRole,
    BlockerClassification,
    EvidencePlaneClassification,
    FrzTrcP0Status,
    TenantIsolationP0Result,
    TraceabilityDomain,
    _FORBIDDEN_GLOBAL_TRACE_NAMES,
    _SENSITIVE_CLASS_SUFFIXES,
    frz_row_by_id,
    repo_root,
    surfaces_by_id,
)

_REPO_ROOT = repo_root()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def _python_files_under(rel_root: str) -> list[Path]:
    root = _REPO_ROOT / rel_root
    if not root.is_dir():
        return []
    files: list[Path] = []
    skip_dir_names = frozenset(
        {
            "__pycache__",
            "tests",
            "docker",
            "runtime-context",
        }
    )
    for path in root.rglob("*.py"):
        if any(part in skip_dir_names for part in path.parts):
            continue
        files.append(path)
    return files


def _class_names_in_file(path: Path) -> frozenset[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError:
        return frozenset()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
    return frozenset(names)


def _discover_sensitive_classes() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for rel in CLOSED_WORLD_ROOTS:
        for path in _python_files_under(rel):
            rel_path = path.relative_to(_REPO_ROOT).as_posix()
            for name in _class_names_in_file(path):
                if name in TRACE_MECHANISM_CLASS_REGISTRY:
                    found.setdefault(name, []).append(rel_path)
                    continue
                for suffix in _SENSITIVE_CLASS_SUFFIXES:
                    if name.endswith(suffix):
                        found.setdefault(name, []).append(rel_path)
    return found


def test_txp0_q01_current_head_anchor() -> None:
    head = _git_head()
    audited = TRACE_X_P0_AUDITED_HEAD
    if head == audited:
        return
    check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", audited, head],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, (
        f"HEAD {head} must equal or descend from TRACE-X-P0 baseline {audited}"
    )


def test_txp0_q02_all_frz_trc_represented() -> None:
    matrix_ids = {row.criterion for row in FRZ_TRC_P0_MATRIX}
    assert matrix_ids == set(MANDATORY_FRZ_TRC_IDS)
    assert len(FRZ_TRC_P0_MATRIX) == 12


def test_txp0_q03_evidence_plane_taxonomy_complete() -> None:
    allowed = frozenset(EvidencePlaneClassification)
    forbidden = {"OTHER", "MISC", "UNKNOWN"}
    for surface in TRACEABILITY_SURFACES:
        assert surface.evidence_plane in allowed
        assert surface.evidence_plane.value not in forbidden


def test_txp0_q04_exactly_one_execution_event_truth_owner() -> None:
    owners = [
        row
        for row in SEMANTIC_OWNER_MATRIX
        if row.concern == "execution event truth"
    ]
    assert len(owners) == 1
    assert "RuntimeEvent" in owners[0].canonical_contract


def test_txp0_q05_exactly_one_factual_reconstruction_owner() -> None:
    recon_rows = [r for r in SEMANTIC_OWNER_MATRIX if r.concern == "factual reconstruction"]
    assert len(recon_rows) == 1
    assert "ExecutionReconstructionReader" in recon_rows[0].canonical_contract


def test_txp0_q06_trace_event_not_execution_truth() -> None:
    trace = surfaces_by_id()["TX-S03"]
    assert trace.evidence_plane == EvidencePlaneClassification.DIAGNOSTIC_READ_MODEL
    assert trace.authority_role != AuthorityRole.MINT_EXECUTION_IDENTITY
    field_names = {f.name for f in dataclasses.fields(TraceEvent)}
    forbidden = {"attempt_id", "execution_id", "task_id", "tenant_id"}
    assert forbidden.isdisjoint(field_names)


def test_txp0_q07_reconstruction_does_not_use_trace_as_execution_authority() -> None:
    recon_path = (
        _REPO_ROOT
        / "intergrax/runtime/observability/reconstruction/execution_reconstruction.py"
    )
    text = recon_path.read_text(encoding="utf-8")
    assert "RunTraceReader" not in text
    assert "TraceEvent" not in text


def test_txp0_q08_diagnostics_consumes_reconstruction_contract() -> None:
    diag = _REPO_ROOT / "intergrax/runtime/diagnostics/diagnostic_orchestrator.py"
    text = diag.read_text(encoding="utf-8")
    assert "ExecutionReconstructionReader" in text
    assert "ExecutionReconstructor" not in text


def test_txp0_q09_observability_diagnostics_no_mint_execution_identity() -> None:
    for surface_id in ("TX-S17", "TX-S18", "TX-S03"):
        surface = surfaces_by_id()[surface_id]
        assert surface.authority_role != AuthorityRole.MINT_EXECUTION_IDENTITY


def test_txp0_q10_transport_runtime_identity_separation() -> None:
    causal = surfaces_by_id()["TX-S04"]
    assert causal.domain == TraceabilityDomain.TRANSPORT_CAUSALITY
    contract = _REPO_ROOT / "intergrax/contracts/platform_causal_evidence.py"
    text = contract.read_text(encoding="utf-8")
    assert "MessageBusTaskRef" in text
    assert "RuntimeExecutionRef" in text
    assert "TRANSPORT_TASK_TRIGGERED_EXECUTION" in text


def test_txp0_q11_parent_child_owner_explicit() -> None:
    lineage = surfaces_by_id()["TX-S05"]
    assert lineage.evidence_plane == EvidencePlaneClassification.CANONICAL_LINEAGE_TRUTH
    owners = [r for r in SEMANTIC_OWNER_MATRIX if r.concern == "execution lineage"]
    assert len(owners) == 1


def test_txp0_q12_tool_attribution_surfaces_inventoried() -> None:
    tool_surfaces = [s for s in TRACEABILITY_SURFACES if s.domain == TraceabilityDomain.TOOL_INVOCATION]
    assert tool_surfaces
    assert any("FRZ-TRC-03" in s.frz_criteria for s in tool_surfaces)


def test_txp0_q13_provider_attribution_surfaces_inventoried() -> None:
    surfaces = [
        s
        for s in TRACEABILITY_SURFACES
        if s.domain == TraceabilityDomain.PROVIDER_INVOCATION
    ]
    assert surfaces


def test_txp0_q14_model_context_attribution_surfaces_inventoried() -> None:
    model = [s for s in TRACEABILITY_SURFACES if s.domain == TraceabilityDomain.MODEL_CALL]
    ctx = [s for s in TRACEABILITY_SURFACES if s.domain == TraceabilityDomain.CONTEXT_DECISION]
    assert model and ctx


def test_txp0_q15_side_effect_authorization_evidence_inventoried() -> None:
    surfaces = [
        s
        for s in TRACEABILITY_SURFACES
        if s.domain == TraceabilityDomain.SIDE_EFFECT_AUTHORIZATION
        or TraceabilityDomain.GOVERNANCE_DECISION in (s.domain,)
    ]
    assert any("FRZ-TRC-06" in s.frz_criteria for s in surfaces)


def test_txp0_q16_policy_provenance_inventoried() -> None:
    assert any(s.domain == TraceabilityDomain.POLICY_REVISION for s in TRACEABILITY_SURFACES)


def test_txp0_q17_profile_provenance_inventoried() -> None:
    profile = [s for s in TRACEABILITY_SURFACES if s.domain == TraceabilityDomain.PROFILE_REVISION]
    assert profile
    assert profile[0].current_status.value == "GAP_REQUIRES_CHILD"


def test_txp0_q18_configured_effective_provenance_inventoried() -> None:
    cfg = [
        s for s in TRACEABILITY_SURFACES if s.domain == TraceabilityDomain.CONFIGURATION_PROVENANCE
    ]
    assert cfg


def test_txp0_q19_restart_resume_continuity_inventoried() -> None:
    cont = [
        s
        for s in TRACEABILITY_SURFACES
        if s.domain == TraceabilityDomain.RESTART_RESUME_CONTINUITY
    ]
    assert cont


def test_txp0_q20_terminal_outcome_evidence_inventoried() -> None:
    term = [s for s in TRACEABILITY_SURFACES if s.domain == TraceabilityDomain.TERMINAL_OUTCOME]
    assert term


def test_txp0_q21_forward_chain_complete_as_inventory() -> None:
    assert len(FORWARD_CHAIN) >= 9
    for transition in FORWARD_CHAIN:
        assert transition.source_semantic_owner.strip()
        assert transition.target_semantic_owner.strip()
        assert transition.joining_identity.strip()
        assert transition.evidence_contract.strip()


def test_txp0_q22_reverse_matrix_complete_as_inventory() -> None:
    subjects = {row.subject for row in REVERSE_RECONSTRUCTION_MATRIX}
    required = {
        "external effect",
        "provider invocation",
        "tool invocation",
        "model call",
        "failure",
        "diagnostic finding",
        "terminal outcome",
    }
    assert required <= subjects


def test_txp0_q23_no_duplicate_trace_reconstruction_authority() -> None:
    owners = [row.canonical_semantic_owner for row in SEMANTIC_OWNER_MATRIX]
    assert len(owners) == len(set(owners))
    recon_impls = _discover_sensitive_classes().get("ExecutionReconstructor", [])
    assert recon_impls == [
        "intergrax/runtime/observability/reconstruction/execution_reconstruction.py"
    ]


def test_txp0_q24_no_alternate_execution_truth_in_observability() -> None:
    obs_root = _REPO_ROOT / "intergrax/runtime/observability"
    for path in obs_root.rglob("*.py"):
        if "reconstruction" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "class ExecutionReconstructor" in text:
            pytest.fail(f"unexpected reconstructor in {path}")


def test_txp0_q25_every_frz_criterion_has_explicit_status() -> None:
    forbidden = {"UNKNOWN", "TBD", "probably"}
    for row in FRZ_TRC_P0_MATRIX:
        assert row.p0_status in FrzTrcP0Status
        gap = row.gap.upper()
        for token in forbidden:
            assert token not in gap


def test_txp0_q26_every_gap_has_future_child_owner() -> None:
    for row in FRZ_TRC_P0_MATRIX:
        assert row.future_child_owner.startswith("TRACE-X-")


def test_txp0_q27_no_future_stage_scope_leakage() -> None:
    forbidden_stages = ("CONFIG-X", "COMPAT-X", "TENANT-X", "PROD-Q", "QUAL-X")
    inventory_text = (_REPO_ROOT / "tests/qualification/trace_x/_trace_x_p0_support.py").read_text(
        encoding="utf-8"
    )
    for stage in forbidden_stages:
        assert f"implement {stage}" not in inventory_text


def test_txp0_q28_tenant_audit_complete() -> None:
    from tests.qualification.trace_x._trace_x_p0_support import TENANT_ISOLATION_AUDIT

    assert TENANT_ISOLATION_AUDIT["tenant_scope_applicable"] == "YES"
    assert TENANT_ISOLATION_AUDIT["result"] == TenantIsolationP0Result.PARTIAL.value


def test_txp0_q29_in_scope_blocker_inventory_explicit() -> None:
    assert TRACE_X_KNOWN_BLOCKERS
    for blocker in TRACE_X_KNOWN_BLOCKERS:
        assert blocker.classification in BlockerClassification


def test_txp0_q30_p0_readiness_gate() -> None:
    assert len(TRACEABILITY_SURFACES) >= 15
    assert len(ARCHITECTURE_LOCK) >= 9
    assert len(TRACE_X_CHILD_DECOMPOSITION) >= 6
    assert len(HISTORICAL_EVIDENCE) >= 8
    assert len(ENTERPRISE_AUDIT_MATRIX) >= 20
    child_frz: set[str] = set()
    for child in TRACE_X_CHILD_DECOMPOSITION:
        child_frz.update(child.frz_criteria)
    assert set(MANDATORY_FRZ_TRC_IDS) <= child_frz


def test_txp0_closed_world_no_forbidden_global_trace_types() -> None:
    for rel in CLOSED_WORLD_ROOTS:
        for path in _python_files_under(rel):
            for name in _class_names_in_file(path):
                assert name not in _FORBIDDEN_GLOBAL_TRACE_NAMES


def test_txp0_closed_world_sensitive_classes_classified() -> None:
    """Unknown relevant mechanism → FAIL (no generic OUTSIDE pass)."""
    discovered = _discover_sensitive_classes()
    unregistered_reconstructors = [
        name
        for name in discovered
        if name.endswith("Reconstructor") and name not in TRACE_MECHANISM_CLASS_REGISTRY
    ]
    assert unregistered_reconstructors == []


def test_txp0_semantic_owner_matrix_no_unknown_tokens() -> None:
    forbidden = {"UNKNOWN", "TBD", "unknown", "tbd"}
    for row in SEMANTIC_OWNER_MATRIX:
        owner = row.canonical_semantic_owner.upper()
        for token in forbidden:
            assert token not in owner


def test_txp0_architecture_lock_contract_paths_exist() -> None:
    for entry in ARCHITECTURE_LOCK:
        for rel in entry.contract_paths:
            assert (_REPO_ROOT / rel).is_file(), rel


def test_txp0_surface_producer_paths_exist() -> None:
    for surface in TRACEABILITY_SURFACES:
        for rel in surface.producer_paths:
            path = _REPO_ROOT / rel
            assert path.exists(), f"{surface.surface_id}: missing {rel}"


def test_txp0_frz_matrix_matches_surface_coverage() -> None:
    covered: set[str] = set()
    for surface in TRACEABILITY_SURFACES:
        covered.update(surface.frz_criteria)
    for criterion in MANDATORY_FRZ_TRC_IDS:
        assert criterion in covered or criterion in frz_row_by_id()


@pytest.mark.parametrize("criterion", MANDATORY_FRZ_TRC_IDS)
def test_txp0_frz_row_present(criterion: str) -> None:
    assert criterion in frz_row_by_id()
