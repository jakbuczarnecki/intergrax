# © Artur Czarnecki. All rights reserved.

"""EBH-7 audit remediation program — materialization gates (agent ≠ B6 PASS)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

_REPO = Path(__file__).resolve().parents[3]

REMEDIATION_ARTIFACTS: Final[tuple[str, ...]] = (
    "docs/project/maintainers/qualification/EBH_7_FINAL_INDEPENDENT_AUDIT_REMEDIATION_MASTER_PLAN.md",
    "docs/project/maintainers/qualification/EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md",
    "docs/project/maintainers/qualification/EBH_7_REMEDIATION_EXECUTION_LEDGER.md",
    "docs/project/maintainers/qualification/EBH_7_FINAL_EXECUTION_CONTINUATION_REPORT.md",
    "docs/project/maintainers/qualification/EBH_7_FINAL_CLOSURE_TRACEABILITY.md",
    "docs/project/maintainers/qualification/EBH_7_INT2E_BUNDLE_QUALIFICATION_MATRIX.md",
    "docs/project/maintainers/qualification/EBH_7_B6_CORRECTIVE_LOOP.md",
    "docs/project/maintainers/qualification/EBH_7_OD_EBH7_08_CLOSURE_PROCEDURE.md",
    "docs/project/maintainers/qualification/EBH_7_OPERATOR_DECISION_PACKAGES_REMEDIATION.md",
    "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json",
    "docs/project/maintainers/qualification/EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md",
    "docs/project/maintainers/qualification/EBH_7_ECAP_EXECUTABLE_CHECKLIST.md",
    "docs/project/maintainers/qualification/EBH_7_ENTERPRISE_FAMILY_COVERAGE_MATRIX.md",
    "scripts/qualification/render_ebh7_final_audit_matrices.py",
)


def assert_ebh7_audit_remediation_artifacts_materialized() -> None:
    missing = [rel for rel in REMEDIATION_ARTIFACTS if not (_REPO / rel).is_file()]
    assert missing == [], f"Remediation artifacts missing: {missing}"


def assert_ebh7_audit_remediation_program_materialized() -> None:
    assert_ebh7_audit_remediation_artifacts_materialized()
    from tests.qualification.ebh_7._rpl_h_normative_evidence import assert_ebh7_rpl_h_normative_evidence_integrity

    assert_ebh7_rpl_h_normative_evidence_integrity()
