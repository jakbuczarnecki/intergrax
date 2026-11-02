from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import AuditMode, CheckRunOutcome
from intergrax.platform.enterprise_audit.coverage import compute_coverage_label


def test_coverage_label_full_requires_all_executed() -> None:
    outcomes = tuple(CheckRunOutcome(check_id=f"C{i}", executed=True, skipped=False) for i in range(85))
    label = compute_coverage_label(
        AuditMode.FULL,
        total_checks=85,
        planned=85,
        outcomes=outcomes,
        findings=(),
    )
    assert label == "FULL"


def test_coverage_label_partial_when_skipped() -> None:
    outcomes = (
        CheckRunOutcome(check_id="C1", executed=True, skipped=False),
        CheckRunOutcome(check_id="C2", executed=False, skipped=True),
    )
    label = compute_coverage_label(
        AuditMode.FULL,
        total_checks=85,
        planned=2,
        outcomes=outcomes,
        findings=(),
    )
    assert label == "PARTIAL"
