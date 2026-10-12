# © Artur Czarnecki. All rights reserved.

"""TENANT-X-FINAL-R1 qualification-framework integrity gates (executable evidence)."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from tests.qualification.tenant_x._final_certification import build_final_qualification_report
from tests.qualification.tenant_x._final_execution import (
    FINAL_ADVERSARIAL_EXECUTABLE_REGISTRY,
    registry_class_ids,
    run_final_adversarial_matrix,
)
from tests.qualification.tenant_x._final_matrices import FINAL_ADVERSARIAL_MATRIX
from tests.qualification.tenant_x._final_production_delta import (
    TENANT_X_FINAL_REJECTED_BASE_SHA,
    build_final_production_delta_evidence,
)
from tests.qualification.tenant_x._final_reconciliation import (
    WaveReadinessOnHead,
    build_frz_ten_reconciliation_rows,
    matrix_class_ids,
    reconcile_final_adversarial_execution,
)
from tests.qualification.tenant_x._final_types import (
    FinalAdversarialClass,
    FinalAdversarialExecutionResult,
    FinalAdversarialOutcome,
    FrzTenId,
    FrzTenReconciliationVerdict,
    REQUIRED_FRZ_TEN_IDS,
)
from tests.qualification.tenant_x._final_wave_evidence import accepted_wave_ancestry_violations
from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_certification import build_r1_qualification_report
from tests.qualification.tenant_x._r2_certification import build_r2_qualification_report
from tests.qualification.tenant_x._r3_certification import build_r3_qualification_report
from tests.qualification.tenant_x._r4_certification import build_r4_qualification_report

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_EXPECTED_F = frozenset(FinalAdversarialClass)


def _golden_execution() -> tuple[FinalAdversarialExecutionResult, ...]:
    return run_final_adversarial_matrix()


def _wave_readiness(adversarial: object) -> WaveReadinessOnHead:
    inventory = get_tenant_x_inventory()
    r1 = build_r1_qualification_report(inventory)
    r2 = build_r2_qualification_report()
    r3 = build_r3_qualification_report()
    r4 = build_r4_qualification_report()
    return WaveReadinessOnHead(
        r1_ready=r1.r1_ready_for_audit,
        r2_ready=r2.r2_ready_for_audit,
        r3_ready=r3.r3_ready_for_audit,
        r4_ready=r4.r4_ready_for_audit,
        accepted_ancestry_ok=accepted_wave_ancestry_violations() == (),
        adversarial_reconciliation=adversarial,  # type: ignore[arg-type]
    )


# --- R1 regression scenarios (must pass only after FINAL-R1 repair) ---


def test_r1_a_no_caller_controlled_readiness_boolean() -> None:
    params = inspect.signature(build_final_qualification_report).parameters
    assert "final_negatives_runnable" not in params
    forbidden = frozenset(
        {"execution", "results", "evidence", "adversarial_results", "runner_results"}
    )
    assert not forbidden.intersection(params)


def test_r1_b_executable_evidence_string_not_execution_proof() -> None:
    assert all(row.executable_evidence for row in FINAL_ADVERSARIAL_MATRIX)
    recon = reconcile_final_adversarial_execution(())
    assert recon.adversarial_matrix_complete is False


def test_r1_c_missing_f7_blocks_reconciliation() -> None:
    execution = tuple(r for r in _golden_execution() if r.adversarial_class != FinalAdversarialClass.F7_CHILD_ASYNC_ISOLATION)
    recon = reconcile_final_adversarial_execution(execution)
    assert recon.adversarial_matrix_complete is False
    assert FinalAdversarialClass.F7_CHILD_ASYNC_ISOLATION in recon.adversarial_missing


def test_r1_d_duplicate_f4_blocks_reconciliation() -> None:
    execution = list(_golden_execution())
    f4 = next(r for r in execution if r.adversarial_class == FinalAdversarialClass.F4_EVIDENCE_ISOLATION)
    execution.append(f4)
    recon = reconcile_final_adversarial_execution(tuple(execution))
    assert recon.adversarial_matrix_complete is False
    assert FinalAdversarialClass.F4_EVIDENCE_ISOLATION in recon.adversarial_duplicates


def test_r1_e_wrong_observed_outcome_blocks() -> None:
    execution = list(_golden_execution())
    for i, item in enumerate(execution):
        if item.adversarial_class == FinalAdversarialClass.F1_AUTHORITY_ISOLATION:
            execution[i] = replace(item, observed_outcome=FinalAdversarialOutcome.PRESERVED)
            break
    recon = reconcile_final_adversarial_execution(tuple(execution))
    assert recon.adversarial_matrix_complete is False
    assert recon.adversarial_outcome_mismatches >= 1


def test_r1_f_not_executed_blocks() -> None:
    execution = list(_golden_execution())
    for i, item in enumerate(execution):
        if item.adversarial_class == FinalAdversarialClass.F8_MISSING_TENANT_FAIL_CLOSED:
            execution[i] = replace(item, executed=False)
            break
    recon = reconcile_final_adversarial_execution(tuple(execution))
    assert recon.adversarial_matrix_complete is False
    assert recon.adversarial_not_executed >= 1


# --- Q01–Q20 ---


def test_tx_final_r1_q01_registry_identity_f1_f10() -> None:
    assert registry_class_ids() == _EXPECTED_F
    assert len(FINAL_ADVERSARIAL_EXECUTABLE_REGISTRY) == 10


def test_tx_final_r1_q02_matrix_identity_f1_f10() -> None:
    assert matrix_class_ids() == _EXPECTED_F


def test_tx_final_r1_q03_registry_equals_matrix() -> None:
    assert registry_class_ids() == matrix_class_ids()


def test_tx_final_r1_q04_missing_runner_blocks() -> None:
    test_r1_c_missing_f7_blocks_reconciliation()


def test_tx_final_r1_q05_duplicate_runner_blocks() -> None:
    test_r1_d_duplicate_f4_blocks_reconciliation()


def test_tx_final_r1_q06_not_executed_blocks() -> None:
    test_r1_f_not_executed_blocks()


def test_tx_final_r1_q07_wrong_outcome_blocks() -> None:
    test_r1_e_wrong_observed_outcome_blocks()


def test_tx_final_r1_q08_unknown_outcome_blocks() -> None:
    execution = list(_golden_execution())
    execution[0] = replace(execution[0], observed_outcome=FinalAdversarialOutcome.UNKNOWN)
    recon = reconcile_final_adversarial_execution(tuple(execution))
    assert recon.adversarial_matrix_complete is False
    assert recon.adversarial_unknown >= 1


def test_tx_final_r1_q09_execution_exception_blocks() -> None:
    execution = list(_golden_execution())
    execution[0] = replace(
        execution[0],
        observed_outcome=FinalAdversarialOutcome.FAILED,
        failure_detail="synthetic runner exception",
    )
    recon = reconcile_final_adversarial_execution(tuple(execution))
    assert recon.adversarial_matrix_complete is False
    assert recon.adversarial_failed >= 1


def test_tx_final_r1_q10_string_executable_evidence_alone_insufficient() -> None:
    test_r1_b_executable_evidence_string_not_execution_proof()


def test_tx_final_r1_q11_no_boolean_forces_readiness() -> None:
    test_r1_a_no_caller_controlled_readiness_boolean()


def test_tx_final_r1_q12_f9_dedicated_replace_drop_widen() -> None:
    f9 = next(r for r in _golden_execution() if r.adversarial_class == FinalAdversarialClass.F9_EXTENSION_PROVIDER_REWRITE)
    assert f9.f9_evidence is not None
    assert f9.f9_evidence.replace_rejected
    assert f9.f9_evidence.drop_rejected
    assert f9.f9_evidence.widen_proof is not None


def test_tx_final_r1_q13_f10_preservation_semantics() -> None:
    row = next(r for r in FINAL_ADVERSARIAL_MATRIX if r.adversarial_class == FinalAdversarialClass.F10_EVIDENCE_TRACE_PRESERVATION)
    assert row.expected_outcome == FinalAdversarialOutcome.PRESERVED
    f10 = next(r for r in _golden_execution() if r.adversarial_class == FinalAdversarialClass.F10_EVIDENCE_TRACE_PRESERVATION)
    assert f10.observed_outcome == FinalAdversarialOutcome.PRESERVED


def test_tx_final_r1_q14_current_head_executes_ten_classes() -> None:
    assert len(_golden_execution()) == 10


def test_tx_final_r1_q15_current_head_reconciliation_zeros() -> None:
    recon = reconcile_final_adversarial_execution(_golden_execution())
    assert recon.adversarial_classes_expected == 10
    assert recon.adversarial_classes_executed == 10
    assert len(recon.adversarial_missing) == 0
    assert len(recon.adversarial_duplicates) == 0
    assert recon.adversarial_unknown == 0
    assert recon.adversarial_not_executed == 0
    assert recon.adversarial_failed == 0
    assert recon.adversarial_outcome_mismatches == 0


def test_tx_final_r1_q16_frz_ten_twelve_exact_identities() -> None:
    recon = reconcile_final_adversarial_execution(_golden_execution())
    rows = build_frz_ten_reconciliation_rows(_wave_readiness(recon))
    assert len(rows) == 12
    assert [row.criterion_id for row in rows] == list(REQUIRED_FRZ_TEN_IDS)


def test_tx_final_r1_q17_frz_ten_10_fails_when_f8_fails() -> None:
    execution = list(_golden_execution())
    for i, item in enumerate(execution):
        if item.adversarial_class == FinalAdversarialClass.F8_MISSING_TENANT_FAIL_CLOSED:
            execution[i] = replace(item, observed_outcome=FinalAdversarialOutcome.FAILED)
            break
    recon = reconcile_final_adversarial_execution(tuple(execution))
    rows = build_frz_ten_reconciliation_rows(_wave_readiness(recon))
    frz10 = next(r for r in rows if r.criterion_id == FrzTenId.FRZ_TEN_10)
    assert frz10.verdict == FrzTenReconciliationVerdict.BLOCKED


def test_tx_final_r1_q18_frz_ten_12_fails_when_any_f_fails() -> None:
    execution = list(_golden_execution())
    execution[0] = replace(execution[0], observed_outcome=FinalAdversarialOutcome.FAILED)
    recon = reconcile_final_adversarial_execution(tuple(execution))
    rows = build_frz_ten_reconciliation_rows(_wave_readiness(recon))
    frz12 = next(r for r in rows if r.criterion_id == FrzTenId.FRZ_TEN_12)
    assert frz12.verdict == FrzTenReconciliationVerdict.BLOCKED


def test_tx_final_r1_q19_production_delta_corrective_range_zero() -> None:
    evidence = build_final_production_delta_evidence()
    assert evidence is not None
    assert evidence.base_sha == TENANT_X_FINAL_REJECTED_BASE_SHA
    assert evidence.production_delta_count == 0


def test_tx_final_r1_q20_ready_false_when_production_delta_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.qualification.tenant_x import _final_certification as cert

    monkeypatch.setattr(cert, "build_final_production_delta_evidence", lambda **_: None)
    report = build_final_qualification_report()
    assert report.final_ready_for_audit is False
    assert report.production_delta_evidence_available is False


# --- mutation anti-false-PASS ---


def test_tx_final_r1_mutation_remove_f4() -> None:
    execution = tuple(r for r in _golden_execution() if r.adversarial_class != FinalAdversarialClass.F4_EVIDENCE_ISOLATION)
    assert reconcile_final_adversarial_execution(execution).adversarial_matrix_complete is False


def test_tx_final_r1_mutation_change_f6_outcome() -> None:
    execution = list(_golden_execution())
    for i, item in enumerate(execution):
        if item.adversarial_class == FinalAdversarialClass.F6_RECOVERY_ISOLATION:
            execution[i] = replace(item, observed_outcome=FinalAdversarialOutcome.DENIED)
            break
    assert reconcile_final_adversarial_execution(tuple(execution)).adversarial_matrix_complete is False


def test_tx_final_r1_mutation_f8_not_executed() -> None:
    test_r1_f_not_executed_blocks()


def test_tx_final_r1_mutation_duplicate_f2() -> None:
    execution = list(_golden_execution())
    f2 = next(r for r in execution if r.adversarial_class == FinalAdversarialClass.F2_STATE_ISOLATION)
    execution.append(f2)
    assert reconcile_final_adversarial_execution(tuple(execution)).adversarial_matrix_complete is False
