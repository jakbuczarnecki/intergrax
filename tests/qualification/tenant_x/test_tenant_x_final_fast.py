# © Artur Czarnecki. All rights reserved.

"""TENANT-X-FINAL fast gates — cross-wave adversarial reconciliation."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._final_adversarial import run_final_composed_adversarial_negatives
from tests.qualification.tenant_x._final_certification import (
    build_final_qualification_report,
    summarize_final_metrics,
)
from tests.qualification.tenant_x._final_matrices import FINAL_ADVERSARIAL_MATRIX
from tests.qualification.tenant_x._final_reconciliation import (
    adversarial_matrix_complete,
    build_frz_ten_reconciliation_rows,
    frz_ten_identity_reconciled,
)
from tests.qualification.tenant_x._final_types import (
    FinalAdversarialClass,
    FrzTenId,
    FrzTenReconciliationVerdict,
    REQUIRED_FRZ_TEN_IDS,
)
from tests.qualification.tenant_x._final_wave_evidence import (
    ACCEPTED_TENANT_X_WAVE_EVIDENCE,
    accepted_wave_ancestry_violations,
    current_git_head_sha,
)
from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_certification import build_r1_qualification_report
from tests.qualification.tenant_x._r2_certification import build_r2_qualification_report
from tests.qualification.tenant_x._r3_certification import build_r3_qualification_report
from tests.qualification.tenant_x._r4_certification import build_r4_qualification_report
from tests.qualification.tenant_x._final_reconciliation import WaveReadinessOnHead

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_final_q01_head_is_git_object() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40
    assert head == current_git_head_sha()


def test_tx_final_q02_accepted_r1_r4_shas_ancestor_of_head() -> None:
    violations = accepted_wave_ancestry_violations()
    assert violations == ()
    assert len(ACCEPTED_TENANT_X_WAVE_EVIDENCE) >= 5


def test_tx_final_q03_adversarial_matrix_f1_f10_closed_world() -> None:
    assert adversarial_matrix_complete()
    assert len(FINAL_ADVERSARIAL_MATRIX) == len(FinalAdversarialClass)
    classes = {row.adversarial_class for row in FINAL_ADVERSARIAL_MATRIX}
    assert classes == set(FinalAdversarialClass)


def test_tx_final_q04_frz_ten_identity_reconciliation_01_12() -> None:
    inventory = get_tenant_x_inventory()
    r1 = build_r1_qualification_report(inventory)
    r2 = build_r2_qualification_report()
    r3 = build_r3_qualification_report()
    r4 = build_r4_qualification_report()
    readiness = WaveReadinessOnHead(
        r1_ready=r1.r1_ready_for_audit,
        r2_ready=r2.r2_ready_for_audit,
        r3_ready=r3.r3_ready_for_audit,
        r4_ready=r4.r4_ready_for_audit,
        accepted_ancestry_ok=accepted_wave_ancestry_violations() == (),
        adversarial_matrix_complete=adversarial_matrix_complete(),
        final_negatives_runnable=True,
    )
    rows = build_frz_ten_reconciliation_rows(readiness)
    assert frz_ten_identity_reconciled(rows)
    assert {row.criterion_id for row in rows} == set(REQUIRED_FRZ_TEN_IDS)
    assert len(rows) == len(REQUIRED_FRZ_TEN_IDS)
    assert all(row.criterion_id == FrzTenId(f"FRZ-TEN-{i:02d}") for i, row in enumerate(rows, start=1))


def test_tx_final_q05_composed_adversarial_negatives_executable() -> None:
    run_final_composed_adversarial_negatives()


def test_tx_final_q06_report_ready_for_audit() -> None:
    report = build_final_qualification_report(final_negatives_runnable=True)
    metrics = summarize_final_metrics(report)
    assert metrics["adversarial_matrix_rows"] == len(FinalAdversarialClass)
    assert report.frz_ten_rows == len(REQUIRED_FRZ_TEN_IDS)
    assert report.frz_ten_reconciled == len(REQUIRED_FRZ_TEN_IDS)
    assert report.frz_ten_blocked == 0
    assert report.frz_ten_open == 0
    assert report.unclassified_frz_ten == 0
    assert report.cross_tenant_allowed == 0
    assert report.cross_tenant_unknown == 0
    assert report.production_delta == 0
    assert report.r1_ready_on_head
    assert report.r2_ready_on_head
    assert report.r3_ready_on_head
    assert report.r4_ready_on_head
    assert report.final_ready_for_audit
    assert report.remediation_children == ()
    assert all(
        row.verdict == FrzTenReconciliationVerdict.RECONCILED_READY_FOR_AUDIT
        for row in build_frz_ten_reconciliation_rows(
            WaveReadinessOnHead(
                r1_ready=report.r1_ready_on_head,
                r2_ready=report.r2_ready_on_head,
                r3_ready=report.r3_ready_on_head,
                r4_ready=report.r4_ready_on_head,
                accepted_ancestry_ok=True,
                adversarial_matrix_complete=True,
                final_negatives_runnable=True,
            )
        )
    )
