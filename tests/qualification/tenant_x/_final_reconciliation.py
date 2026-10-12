# © Artur Czarnecki. All rights reserved.

"""FRZ-TEN-01..12 FINAL reconciliation rows (typed, identity-checked)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._final_execution import registry_class_ids
from tests.qualification.tenant_x._final_matrices import FINAL_ADVERSARIAL_MATRIX
from tests.qualification.tenant_x._final_types import (
    FinalAdversarialClass,
    FinalAdversarialExecutionResult,
    FinalAdversarialOutcome,
    FinalAdversarialReconciliationEvidence,
    FrzTenId,
    FrzTenReconciliationRow,
    FrzTenReconciliationVerdict,
    REQUIRED_FRZ_TEN_IDS,
    TenantXWave,
)


@dataclass(frozen=True, slots=True)
class WaveReadinessOnHead:
    r1_ready: bool
    r2_ready: bool
    r3_ready: bool
    r4_ready: bool
    accepted_ancestry_ok: bool
    adversarial_reconciliation: FinalAdversarialReconciliationEvidence

    def wave_ready(self, wave: TenantXWave) -> bool:
        if wave == TenantXWave.R1:
            return self.r1_ready
        if wave == TenantXWave.R2:
            return self.r2_ready
        if wave == TenantXWave.R3:
            return self.r3_ready
        if wave == TenantXWave.R4:
            return self.r4_ready
        if wave == TenantXWave.FINAL:
            return self.adversarial_reconciliation.adversarial_matrix_complete
        if wave == TenantXWave.P0:
            return self.accepted_ancestry_ok
        return False

    def waves_ready(self, waves: tuple[TenantXWave, ...]) -> bool:
        return all(self.wave_ready(w) for w in waves)


_FRZ_TEN_STATIC: tuple[tuple[FrzTenId, str, TenantXWave, tuple[TenantXWave, ...], str, str], ...] = (
    (
        FrzTenId.FRZ_TEN_01,
        "canonical tenant identity where tenant scope applies",
        TenantXWave.R1,
        (TenantXWave.R1, TenantXWave.R2, TenantXWave.R3, TenantXWave.R4),
        "build_r1_qualification_report + test_tenant_x_r1_current_head",
        "tests.qualification.tenant_x._r1_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_02,
        "end-to-end tenant propagation across relevant semantic and runtime boundaries",
        TenantXWave.R4,
        (TenantXWave.R1, TenantXWave.R4),
        "build_r4_qualification_report E2E chain + R1 propagation edges",
        "tests.qualification.tenant_x._r4_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_03,
        "child/downstream operations cannot silently switch or widen tenant scope",
        TenantXWave.R4,
        (TenantXWave.R1, TenantXWave.R3, TenantXWave.R4),
        "R4 async/child matrices + R1 authority scope resolutions",
        "tests.qualification.tenant_x._r4_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_04,
        "cross-tenant state/cache/persistence/memory/index access blocked",
        TenantXWave.R3,
        (TenantXWave.R3,),
        "build_r3_qualification_report checkpoint→recovery PROVEN_CURRENT_HEAD",
        "tests.qualification.tenant_x._r3_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_05,
        "provider selection, configuration, profiles preserve tenant scope",
        TenantXWave.R2,
        (TenantXWave.R2,),
        "build_r2_qualification_report provider continuity matrices",
        "tests.qualification.tenant_x._r2_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_06,
        "tenant credential references cannot resolve under another tenant context",
        TenantXWave.R2,
        (TenantXWave.R2,),
        "build_r2_qualification_report credential resolution + consumption",
        "tests.qualification.tenant_x._r2_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_07,
        "evidence, events, traces preserve tenant identity",
        TenantXWave.R4,
        (TenantXWave.R4, TenantXWave.R2),
        "build_r4_qualification_report evidence/event/trace matrices",
        "tests.qualification.tenant_x._r4_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_08,
        "checkpoint, retry, recovery, replay preserve original tenant identity",
        TenantXWave.R3,
        (TenantXWave.R3,),
        "build_r3_qualification_report recovery continuity",
        "tests.qualification.tenant_x._r3_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_09,
        "legal cross-tenant requires explicit governed contract; no implicit bypass",
        TenantXWave.R1,
        (TenantXWave.R1,),
        "build_r1_qualification_report legal_cross_tenant_mechanisms inventory",
        "tests.qualification.tenant_x._r1_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_10,
        "missing/invalid tenant must not become global/shared/default scope",
        TenantXWave.FINAL,
        (TenantXWave.R1, TenantXWave.R2, TenantXWave.R3, TenantXWave.R4, TenantXWave.FINAL),
        "composed fail-closed negatives on current HEAD",
        "tests.qualification.tenant_x._final_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_11,
        "extensions/providers cannot mint, rewrite or widen tenant scope",
        TenantXWave.R2,
        (TenantXWave.R2, TenantXWave.R4),
        "R2 provider rewrite matrices + R4 extension scope matrix",
        "tests.qualification.tenant_x._r2_adversarial",
    ),
    (
        FrzTenId.FRZ_TEN_12,
        "executable adversarial tests prove tenant isolation on critical paths",
        TenantXWave.FINAL,
        (TenantXWave.R1, TenantXWave.R2, TenantXWave.R3, TenantXWave.R4, TenantXWave.FINAL),
        "FINAL_ADVERSARIAL_MATRIX + test_tenant_x_final_fast",
        "tests.qualification.tenant_x.test_tenant_x_final_fast",
    ),
)


def _row_verdict(
    readiness: WaveReadinessOnHead,
    accepted_waves: tuple[TenantXWave, ...],
    criterion_id: FrzTenId,
) -> FrzTenReconciliationVerdict:
    if not readiness.accepted_ancestry_ok:
        return FrzTenReconciliationVerdict.BLOCKED
    if not readiness.waves_ready(accepted_waves):
        return FrzTenReconciliationVerdict.BLOCKED
    if criterion_id == FrzTenId.FRZ_TEN_10 and not readiness.adversarial_reconciliation.f8_passed():
        return FrzTenReconciliationVerdict.BLOCKED
    if criterion_id == FrzTenId.FRZ_TEN_12 and not readiness.adversarial_reconciliation.adversarial_matrix_complete:
        return FrzTenReconciliationVerdict.BLOCKED
    return FrzTenReconciliationVerdict.RECONCILED_READY_FOR_AUDIT


def build_frz_ten_reconciliation_rows(
    readiness: WaveReadinessOnHead,
) -> tuple[FrzTenReconciliationRow, ...]:
    rows: list[FrzTenReconciliationRow] = []
    for (
        criterion_id,
        meaning,
        primary_owner,
        accepted_waves,
        head_validation,
        adversarial_evidence,
    ) in _FRZ_TEN_STATIC:
        verdict = _row_verdict(readiness, accepted_waves, criterion_id)
        blocker: str | None = None
        if verdict != FrzTenReconciliationVerdict.RECONCILED_READY_FOR_AUDIT:
            missing = [w.value for w in accepted_waves if not readiness.wave_ready(w)]
            blocker = f"wave_not_ready_on_head:{','.join(missing)}"
        rows.append(
            FrzTenReconciliationRow(
                criterion_id=criterion_id,
                canonical_meaning=meaning,
                primary_evidence_owner=primary_owner,
                accepted_source_waves=accepted_waves,
                current_head_validation=head_validation,
                adversarial_evidence=adversarial_evidence,
                verdict=verdict,
                unresolved_blocker=blocker,
            )
        )
    return tuple(rows)


def frz_ten_identity_reconciled(rows: tuple[FrzTenReconciliationRow, ...]) -> bool:
    row_ids = {row.criterion_id for row in rows}
    required = set(REQUIRED_FRZ_TEN_IDS)
    return row_ids == required and len(rows) == len(REQUIRED_FRZ_TEN_IDS)


def unclassified_frz_ten_count(rows: tuple[FrzTenReconciliationRow, ...]) -> int:
    if not frz_ten_identity_reconciled(rows):
        return len(REQUIRED_FRZ_TEN_IDS)
    unknown = sum(
        1
        for row in rows
        if row.verdict not in (
            FrzTenReconciliationVerdict.RECONCILED_READY_FOR_AUDIT,
            FrzTenReconciliationVerdict.BLOCKED,
            FrzTenReconciliationVerdict.OPEN,
        )
    )
    return unknown


def matrix_class_ids() -> frozenset[FinalAdversarialClass]:
    return frozenset(row.adversarial_class for row in FINAL_ADVERSARIAL_MATRIX)


def _expected_outcome_for(adversarial_class: FinalAdversarialClass) -> FinalAdversarialOutcome:
    for row in FINAL_ADVERSARIAL_MATRIX:
        if row.adversarial_class == adversarial_class:
            return row.expected_outcome
    raise KeyError(adversarial_class)


def reconcile_final_adversarial_execution(
    execution: tuple[FinalAdversarialExecutionResult, ...],
) -> FinalAdversarialReconciliationEvidence:
    expected_set = set(FinalAdversarialClass)
    matrix_ids = matrix_class_ids()
    registry_ids = registry_class_ids()
    if matrix_ids != expected_set or registry_ids != expected_set or matrix_ids != registry_ids:
        return FinalAdversarialReconciliationEvidence(
            adversarial_matrix_complete=False,
            adversarial_classes_expected=len(expected_set),
            adversarial_classes_executed=0,
            adversarial_missing=tuple(sorted(expected_set, key=lambda c: c.value)),
            adversarial_duplicates=(),
            adversarial_unknown=0,
            adversarial_not_executed=len(expected_set),
            adversarial_outcome_mismatches=0,
            adversarial_failed=0,
            f8_missing_tenant_reconciled=False,
        )

    seen: list[FinalAdversarialClass] = []
    duplicates: list[FinalAdversarialClass] = []
    unknown = 0
    not_executed = 0
    outcome_mismatches = 0
    failed = 0
    f8_ok = False

    for item in execution:
        if item.adversarial_class in seen:
            duplicates.append(item.adversarial_class)
        seen.append(item.adversarial_class)
        if item.observed_outcome == FinalAdversarialOutcome.UNKNOWN:
            unknown += 1
        if not item.executed:
            not_executed += 1
        if item.observed_outcome == FinalAdversarialOutcome.FAILED:
            failed += 1
        expected = _expected_outcome_for(item.adversarial_class)
        if item.executed and item.observed_outcome != expected:
            outcome_mismatches += 1
        if item.adversarial_class == FinalAdversarialClass.F8_MISSING_TENANT_FAIL_CLOSED:
            f8_ok = (
                item.executed
                and item.observed_outcome == expected
                and item.observed_outcome != FinalAdversarialOutcome.FAILED
            )

    executed_classes = frozenset(seen)
    missing = tuple(sorted(expected_set - executed_classes, key=lambda c: c.value))
    dup_tuple = tuple(sorted(set(duplicates), key=lambda c: c.value))

    complete = (
        len(expected_set) == len(execution)
        and len(missing) == 0
        and len(dup_tuple) == 0
        and unknown == 0
        and not_executed == 0
        and outcome_mismatches == 0
        and failed == 0
    )

    return FinalAdversarialReconciliationEvidence(
        adversarial_matrix_complete=complete,
        adversarial_classes_expected=len(expected_set),
        adversarial_classes_executed=len(executed_classes),
        adversarial_missing=missing,
        adversarial_duplicates=dup_tuple,
        adversarial_unknown=unknown,
        adversarial_not_executed=not_executed,
        adversarial_outcome_mismatches=outcome_mismatches,
        adversarial_failed=failed,
        f8_missing_tenant_reconciled=f8_ok,
    )


def adversarial_matrix_complete_from_execution(
    execution: tuple[FinalAdversarialExecutionResult, ...],
) -> bool:
    return reconcile_final_adversarial_execution(execution).adversarial_matrix_complete


# Backward import surface: structural matrix identity only (not execution proof).
def adversarial_matrix_structural_identity_complete() -> bool:
    classes = {row.adversarial_class for row in FINAL_ADVERSARIAL_MATRIX}
    return (
        len(FINAL_ADVERSARIAL_MATRIX) == len(FinalAdversarialClass)
        and len(classes) == len(FinalAdversarialClass)
        and matrix_class_ids() == registry_class_ids()
    )
