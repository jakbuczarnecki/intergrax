# © Artur Czarnecki. All rights reserved.

"""TENANT-X-FINAL-R1-R1 — authoritative execution provenance closure gates."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import pytest

from tests.qualification.tenant_x._final_certification import build_final_qualification_report
from tests.qualification.tenant_x._final_execution import run_final_adversarial_matrix
from tests.qualification.tenant_x._final_matrices import FINAL_ADVERSARIAL_MATRIX
from tests.qualification.tenant_x._final_production_delta import (
    TENANT_X_FINAL_R1_REJECTED_BASE_SHA,
    TENANT_X_FINAL_REJECTED_BASE_SHA,
    build_final_production_delta_evidence,
)
from tests.qualification.tenant_x._final_reconciliation import reconcile_final_adversarial_execution
from tests.qualification.tenant_x._final_types import (
    FinalAdversarialClass,
    FinalAdversarialExecutionResult,
    FinalAdversarialOutcome,
    FinalF9ExecutionEvidence,
    FinalF9WidenProof,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_FORBIDDEN_BUILDER_PARAMS = frozenset(
    {"execution", "results", "evidence", "adversarial_results", "runner_results"}
)


def _golden_execution() -> tuple[FinalAdversarialExecutionResult, ...]:
    return run_final_adversarial_matrix()


def _f9_row() -> FinalAdversarialExecutionResult:
    return next(
        r for r in _golden_execution() if r.adversarial_class == FinalAdversarialClass.F9_EXTENSION_PROVIDER_REWRITE
    )


def _with_f9_patch(
    execution: tuple[FinalAdversarialExecutionResult, ...],
    *,
    f9_evidence: FinalF9ExecutionEvidence | None,
    observed: FinalAdversarialOutcome = FinalAdversarialOutcome.REJECTED,
) -> tuple[FinalAdversarialExecutionResult, ...]:
    patched = list(execution)
    for index, item in enumerate(patched):
        if item.adversarial_class == FinalAdversarialClass.F9_EXTENSION_PROVIDER_REWRITE:
            patched[index] = replace(
                item,
                f9_evidence=f9_evidence,
                observed_outcome=observed,
                executed=True,
            )
            break
    return tuple(patched)


def test_tx_final_r1_r1_q01_builder_has_no_execution_injection_parameter() -> None:
    params = inspect.signature(build_final_qualification_report).parameters
    assert not _FORBIDDEN_BUILDER_PARAMS.intersection(params)


def test_tx_final_r1_r1_q02_synthetic_execution_cannot_be_passed_to_authoritative_builder() -> None:
    params = inspect.signature(build_final_qualification_report).parameters
    assert "execution" not in params


def test_tx_final_r1_r1_q03_reconcile_alone_has_no_authoritative_final_ready() -> None:
    recon = reconcile_final_adversarial_execution(_golden_execution())
    assert not hasattr(recon, "final_ready_for_audit")
    assert not hasattr(recon, "final_ready")


def test_tx_final_r1_r1_q04_f9_evidence_none_blocks() -> None:
    recon = reconcile_final_adversarial_execution(_with_f9_patch(_golden_execution(), f9_evidence=None))
    assert recon.adversarial_matrix_complete is False
    assert "F9_EVIDENCE_ABSENT" in recon.adversarial_specialized_evidence_failures


def test_tx_final_r1_r1_q05_f9_replace_false_blocks() -> None:
    base = _f9_row().f9_evidence
    assert base is not None
    partial = FinalF9ExecutionEvidence(
        replace_rejected=False,
        widen_proof=base.widen_proof,
        drop_rejected=True,
    )
    recon = reconcile_final_adversarial_execution(_with_f9_patch(_golden_execution(), f9_evidence=partial))
    assert recon.adversarial_matrix_complete is False
    assert "F9_REPLACE_NOT_REJECTED" in recon.adversarial_specialized_evidence_failures


def test_tx_final_r1_r1_q06_f9_drop_false_blocks() -> None:
    base = _f9_row().f9_evidence
    assert base is not None
    partial = FinalF9ExecutionEvidence(
        replace_rejected=True,
        widen_proof=base.widen_proof,
        drop_rejected=False,
    )
    recon = reconcile_final_adversarial_execution(_with_f9_patch(_golden_execution(), f9_evidence=partial))
    assert recon.adversarial_matrix_complete is False
    assert "F9_DROP_NOT_REJECTED" in recon.adversarial_specialized_evidence_failures


def test_tx_final_r1_r1_q07_f9_widen_none_blocks() -> None:
    partial = FinalF9ExecutionEvidence(
        replace_rejected=True,
        widen_proof=None,
        drop_rejected=True,
    )
    recon = reconcile_final_adversarial_execution(_with_f9_patch(_golden_execution(), f9_evidence=partial))
    assert recon.adversarial_matrix_complete is False
    assert "F9_WIDEN_PROOF_ABSENT" in recon.adversarial_specialized_evidence_failures


def test_tx_final_r1_r1_q08_f9_fully_valid_passes() -> None:
    recon = reconcile_final_adversarial_execution(_golden_execution())
    assert recon.f9_sub_evidence_complete
    assert recon.adversarial_specialized_evidence_failures == ()


def test_tx_final_r1_r1_q09_f10_preservation_remains_pass() -> None:
    row = next(
        r for r in FINAL_ADVERSARIAL_MATRIX if r.adversarial_class == FinalAdversarialClass.F10_EVIDENCE_TRACE_PRESERVATION
    )
    assert row.expected_outcome == FinalAdversarialOutcome.PRESERVED
    f10 = next(
        r for r in _golden_execution() if r.adversarial_class == FinalAdversarialClass.F10_EVIDENCE_TRACE_PRESERVATION
    )
    assert f10.observed_outcome == FinalAdversarialOutcome.PRESERVED


def test_tx_final_r1_r1_q10_authoritative_builder_invokes_canonical_runner() -> None:
    source = inspect.getsource(build_final_qualification_report)
    tree = ast.parse(source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    assert "run_final_adversarial_matrix" in called


def test_tx_final_r1_r1_q11_authoritative_current_head_matrix_zeros() -> None:
    report = build_final_qualification_report()
    assert report.adversarial_classes_expected == 10
    assert report.adversarial_classes_executed == 10
    assert report.adversarial_missing == 0
    assert report.adversarial_duplicates == 0
    assert report.adversarial_unknown == 0
    assert report.adversarial_not_executed == 0
    assert report.adversarial_failed == 0
    assert report.adversarial_outcome_mismatches == 0


def test_tx_final_r1_r1_production_delta_cumulative_and_corrective_zero() -> None:
    cumulative = build_final_production_delta_evidence(base_sha=TENANT_X_FINAL_REJECTED_BASE_SHA)
    corrective = build_final_production_delta_evidence(base_sha=TENANT_X_FINAL_R1_REJECTED_BASE_SHA)
    assert cumulative is not None and cumulative.production_delta_count == 0
    assert corrective is not None and corrective.production_delta_count == 0
    assert corrective.base_sha == TENANT_X_FINAL_R1_REJECTED_BASE_SHA


def test_tx_final_r1_r1_synthetic_f9_rejected_without_evidence_blocks() -> None:
    forged = replace(
        _f9_row(),
        f9_evidence=None,
        observed_outcome=FinalAdversarialOutcome.REJECTED,
        executed=True,
    )
    execution = tuple(
        forged if r.adversarial_class == FinalAdversarialClass.F9_EXTENSION_PROVIDER_REWRITE else r
        for r in _golden_execution()
    )
    recon = reconcile_final_adversarial_execution(execution)
    assert recon.adversarial_matrix_complete is False
    assert recon.adversarial_outcome_mismatches >= 1
