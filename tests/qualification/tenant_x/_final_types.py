# © Artur Czarnecki. All rights reserved.

"""TENANT-X-FINAL typed reconciliation and adversarial matrix models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class TenantXWave(StrEnum):
    P0 = "TENANT-X-P0"
    R1 = "TENANT-X-R1"
    R2 = "TENANT-X-R2"
    R3 = "TENANT-X-R3"
    R4 = "TENANT-X-R4"
    FINAL = "TENANT-X-FINAL"


class FinalAdversarialClass(StrEnum):
    F1_AUTHORITY_ISOLATION = "F1"
    F2_STATE_ISOLATION = "F2"
    F3_PROVIDER_CONFIG_ISOLATION = "F3"
    F4_EVIDENCE_ISOLATION = "F4"
    F5_CREDENTIAL_ISOLATION = "F5"
    F6_RECOVERY_ISOLATION = "F6"
    F7_CHILD_ASYNC_ISOLATION = "F7"
    F8_MISSING_TENANT_FAIL_CLOSED = "F8"
    F9_EXTENSION_PROVIDER_REWRITE = "F9"
    F10_EVIDENCE_TRACE_PRESERVATION = "F10"


class FrzTenId(StrEnum):
    FRZ_TEN_01 = "FRZ-TEN-01"
    FRZ_TEN_02 = "FRZ-TEN-02"
    FRZ_TEN_03 = "FRZ-TEN-03"
    FRZ_TEN_04 = "FRZ-TEN-04"
    FRZ_TEN_05 = "FRZ-TEN-05"
    FRZ_TEN_06 = "FRZ-TEN-06"
    FRZ_TEN_07 = "FRZ-TEN-07"
    FRZ_TEN_08 = "FRZ-TEN-08"
    FRZ_TEN_09 = "FRZ-TEN-09"
    FRZ_TEN_10 = "FRZ-TEN-10"
    FRZ_TEN_11 = "FRZ-TEN-11"
    FRZ_TEN_12 = "FRZ-TEN-12"


REQUIRED_FRZ_TEN_IDS: Final[tuple[FrzTenId, ...]] = tuple(FrzTenId)


class FrzTenReconciliationVerdict(StrEnum):
    """FINAL reconciliation only — does not promote global FRZ-TEN PASS."""

    RECONCILED_READY_FOR_AUDIT = "RECONCILED_READY_FOR_AUDIT"
    BLOCKED = "BLOCKED"
    OPEN = "OPEN"


class FinalAdversarialOutcome(StrEnum):
    DENIED = "DENIED"
    REJECTED = "REJECTED"
    PRESERVED = "PRESERVED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class FinalF9WidenProof(StrEnum):
    REJECTED = "REJECTED"
    STRUCTURALLY_IMPOSSIBLE = "STRUCTURALLY_IMPOSSIBLE"


@dataclass(frozen=True, slots=True)
class FinalF9ExecutionEvidence:
    replace_rejected: bool
    widen_proof: FinalF9WidenProof | None
    drop_rejected: bool

    def complete(self) -> bool:
        return (
            self.replace_rejected
            and self.drop_rejected
            and self.widen_proof is not None
        )


@dataclass(frozen=True, slots=True)
class FinalAdversarialExecutionResult:
    adversarial_class: FinalAdversarialClass
    executed: bool
    observed_outcome: FinalAdversarialOutcome
    evidence_source: str
    failure_detail: str | None
    f9_evidence: FinalF9ExecutionEvidence | None = None


@dataclass(frozen=True, slots=True)
class FinalAdversarialReconciliationEvidence:
    adversarial_matrix_complete: bool
    adversarial_classes_expected: int
    adversarial_classes_executed: int
    adversarial_missing: tuple[FinalAdversarialClass, ...]
    adversarial_duplicates: tuple[FinalAdversarialClass, ...]
    adversarial_unknown: int
    adversarial_not_executed: int
    adversarial_outcome_mismatches: int
    adversarial_failed: int
    f8_missing_tenant_reconciled: bool
    f9_sub_evidence_complete: bool
    adversarial_specialized_evidence_failures: tuple[str, ...]

    def f8_passed(self) -> bool:
        return self.f8_missing_tenant_reconciled

    def f9_passed(self) -> bool:
        return self.f9_sub_evidence_complete


@dataclass(frozen=True, slots=True)
class FinalProductionDeltaEvidence:
    base_sha: str
    head_sha: str
    changed_production_paths: tuple[str, ...]
    production_delta_count: int


@dataclass(frozen=True, slots=True)
class AcceptedWaveEvidence:
    wave: TenantXWave
    accepted_evidence_sha: str
    rejected_lineage_shas: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FinalAdversarialMatrixRow:
    adversarial_class: FinalAdversarialClass
    scenario: str
    expected_outcome: FinalAdversarialOutcome
    primary_evidence_owner: TenantXWave
    accepted_source_waves: tuple[TenantXWave, ...]
    current_head_validation: str
    executable_evidence: str
    frz_ten_criteria: tuple[FrzTenId, ...]


@dataclass(frozen=True, slots=True)
class FrzTenReconciliationRow:
    criterion_id: FrzTenId
    canonical_meaning: str
    primary_evidence_owner: TenantXWave
    accepted_source_waves: tuple[TenantXWave, ...]
    current_head_validation: str
    adversarial_evidence: str
    verdict: FrzTenReconciliationVerdict
    unresolved_blocker: str | None


@dataclass(frozen=True, slots=True)
class TenantXFinalQualificationReport:
    current_head_sha: str
    accepted_r1_sha: str
    accepted_r2_sha: str
    accepted_r3_sha: str
    accepted_r4_sha: str
    r1_ready_on_head: bool
    r2_ready_on_head: bool
    r3_ready_on_head: bool
    r4_ready_on_head: bool
    adversarial_matrix_rows: int
    adversarial_matrix_executable: int
    adversarial_classes_expected: int
    adversarial_classes_executed: int
    adversarial_missing: int
    adversarial_duplicates: int
    adversarial_unknown: int
    adversarial_not_executed: int
    adversarial_outcome_mismatches: int
    adversarial_failed: int
    production_delta_evidence_available: bool
    frz_ten_rows: int
    frz_ten_reconciled: int
    frz_ten_blocked: int
    frz_ten_open: int
    cross_tenant_allowed: int
    cross_tenant_unknown: int
    unclassified_frz_ten: int
    final_blockers: tuple[str, ...]
    production_delta: int
    final_ready_for_audit: bool
    remediation_children: tuple[str, ...]
