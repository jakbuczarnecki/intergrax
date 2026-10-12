# © Artur Czarnecki. All rights reserved.

"""TENANT-X-FINAL qualification report and READY FOR AUDIT gates."""

from __future__ import annotations

from tests.qualification.tenant_x._final_execution import run_final_adversarial_matrix
from tests.qualification.tenant_x._final_matrices import FINAL_ADVERSARIAL_MATRIX
from tests.qualification.tenant_x._final_production_delta import build_final_production_delta_evidence
from tests.qualification.tenant_x._final_reconciliation import (
    WaveReadinessOnHead,
    build_frz_ten_reconciliation_rows,
    frz_ten_identity_reconciled,
    reconcile_final_adversarial_execution,
    unclassified_frz_ten_count,
)
from tests.qualification.tenant_x._final_types import (
    FrzTenReconciliationVerdict,
    TenantXFinalQualificationReport,
    TenantXWave,
)
from tests.qualification.tenant_x._final_wave_evidence import (
    accepted_wave_ancestry_violations,
    accepted_wave_by_id,
    current_git_head_sha,
)
from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_certification import build_r1_qualification_report
from tests.qualification.tenant_x._r2_certification import build_r2_qualification_report
from tests.qualification.tenant_x._r3_certification import build_r3_qualification_report
from tests.qualification.tenant_x._r4_certification import build_r4_qualification_report


def build_final_qualification_report() -> TenantXFinalQualificationReport:
    head = current_git_head_sha()
    ancestry_violations = accepted_wave_ancestry_violations(head)
    ancestry_ok = len(ancestry_violations) == 0

    inventory = get_tenant_x_inventory()
    r1 = build_r1_qualification_report(inventory)
    r2 = build_r2_qualification_report()
    r3 = build_r3_qualification_report()
    r4 = build_r4_qualification_report()

    execution = run_final_adversarial_matrix()
    adversarial = reconcile_final_adversarial_execution(execution)
    production_evidence = build_final_production_delta_evidence(head_sha=head)
    production_ok = production_evidence is not None and production_evidence.production_delta_count == 0

    readiness = WaveReadinessOnHead(
        r1_ready=r1.r1_ready_for_audit,
        r2_ready=r2.r2_ready_for_audit,
        r3_ready=r3.r3_ready_for_audit,
        r4_ready=r4.r4_ready_for_audit,
        accepted_ancestry_ok=ancestry_ok,
        adversarial_reconciliation=adversarial,
    )
    frz_rows = build_frz_ten_reconciliation_rows(readiness)

    reconciled = sum(
        1 for row in frz_rows if row.verdict == FrzTenReconciliationVerdict.RECONCILED_READY_FOR_AUDIT
    )
    blocked = sum(1 for row in frz_rows if row.verdict == FrzTenReconciliationVerdict.BLOCKED)
    open_count = sum(1 for row in frz_rows if row.verdict == FrzTenReconciliationVerdict.OPEN)

    cross_allowed = r2.cross_tenant_allowed + r3.cross_tenant_allowed + r4.cross_tenant_allowed
    cross_unknown = r2.cross_tenant_unknown + r3.cross_tenant_unknown + r4.cross_tenant_unknown

    remediation: list[str] = []
    if ancestry_violations:
        remediation.append("TENANT-X-FINAL-R1 - accepted wave SHAs not ancestor of current HEAD")
    if not r1.r1_ready_for_audit:
        remediation.append("TENANT-X-FINAL-R2 - R1 current-HEAD replay failed")
    if not r2.r2_ready_for_audit:
        remediation.append("TENANT-X-FINAL-R3 - R2 current-HEAD replay failed")
    if not r3.r3_ready_for_audit:
        remediation.append("TENANT-X-FINAL-R4 - R3 current-HEAD replay failed")
    if not r4.r4_ready_for_audit:
        remediation.append("TENANT-X-FINAL-R5 - R4 current-HEAD replay failed")
    if not adversarial.adversarial_matrix_complete:
        remediation.append("TENANT-X-FINAL-R6 - adversarial execution reconciliation incomplete")
    if production_evidence is None:
        remediation.append("TENANT-X-FINAL-R7 - production delta evidence unavailable")
    elif production_evidence.production_delta_count != 0:
        remediation.append("TENANT-X-FINAL-R8 - production delta non-zero in corrective range")
    if cross_allowed or cross_unknown:
        remediation.append("TENANT-X-FINAL-R9 - cross-tenant ALLOWED/UNKNOWN on composed matrices")
    if not frz_ten_identity_reconciled(frz_rows):
        remediation.append("TENANT-X-FINAL-R10 - FRZ-TEN identity reconciliation mismatch")
    for row in frz_rows:
        if row.unresolved_blocker:
            remediation.append(f"TENANT-X-FINAL-FRZ-{row.criterion_id.value} - {row.unresolved_blocker}")

    ready = (
        ancestry_ok
        and r1.r1_ready_for_audit
        and r2.r2_ready_for_audit
        and r3.r3_ready_for_audit
        and r4.r4_ready_for_audit
        and adversarial.adversarial_matrix_complete
        and production_ok
        and frz_ten_identity_reconciled(frz_rows)
        and reconciled == len(frz_rows)
        and blocked == 0
        and open_count == 0
        and cross_allowed == 0
        and cross_unknown == 0
        and unclassified_frz_ten_count(frz_rows) == 0
        and not remediation
    )

    prod_delta = production_evidence.production_delta_count if production_evidence else -1

    return TenantXFinalQualificationReport(
        current_head_sha=head,
        accepted_r1_sha=accepted_wave_by_id(TenantXWave.R1).accepted_evidence_sha,
        accepted_r2_sha=accepted_wave_by_id(TenantXWave.R2).accepted_evidence_sha,
        accepted_r3_sha=accepted_wave_by_id(TenantXWave.R3).accepted_evidence_sha,
        accepted_r4_sha=accepted_wave_by_id(TenantXWave.R4).accepted_evidence_sha,
        r1_ready_on_head=r1.r1_ready_for_audit,
        r2_ready_on_head=r2.r2_ready_for_audit,
        r3_ready_on_head=r3.r3_ready_for_audit,
        r4_ready_on_head=r4.r4_ready_for_audit,
        adversarial_matrix_rows=len(FINAL_ADVERSARIAL_MATRIX),
        adversarial_matrix_executable=adversarial.adversarial_classes_executed,
        adversarial_classes_expected=adversarial.adversarial_classes_expected,
        adversarial_classes_executed=adversarial.adversarial_classes_executed,
        adversarial_missing=len(adversarial.adversarial_missing),
        adversarial_duplicates=len(adversarial.adversarial_duplicates),
        adversarial_unknown=adversarial.adversarial_unknown,
        adversarial_not_executed=adversarial.adversarial_not_executed,
        adversarial_outcome_mismatches=adversarial.adversarial_outcome_mismatches,
        adversarial_failed=adversarial.adversarial_failed,
        production_delta_evidence_available=production_evidence is not None,
        frz_ten_rows=len(frz_rows),
        frz_ten_reconciled=reconciled,
        frz_ten_blocked=blocked,
        frz_ten_open=open_count,
        cross_tenant_allowed=cross_allowed,
        cross_tenant_unknown=cross_unknown,
        unclassified_frz_ten=unclassified_frz_ten_count(frz_rows),
        final_blockers=tuple(remediation),
        production_delta=prod_delta,
        final_ready_for_audit=ready,
        remediation_children=tuple(remediation),
    )


def summarize_final_metrics(report: TenantXFinalQualificationReport) -> dict[str, int | str | bool]:
    return {
        "current_head_sha": report.current_head_sha,
        "adversarial_matrix_rows": report.adversarial_matrix_rows,
        "frz_ten_rows": report.frz_ten_rows,
        "frz_ten_reconciled": report.frz_ten_reconciled,
        "cross_tenant_allowed": report.cross_tenant_allowed,
        "cross_tenant_unknown": report.cross_tenant_unknown,
        "unclassified_frz_ten": report.unclassified_frz_ten,
        "final_ready_for_audit": report.final_ready_for_audit,
    }
