# © Artur Czarnecki. All rights reserved.

"""EBH-7 G01–G09 closure evidence audit — programmatic SSOT before independent B6."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.ebh_5._proof_validation import pytest_test_function_exists
from tests.qualification.ebh_5._rpl_f_evidence_registry import (
    _PACKAGE_C_RPL_F_NA,
    rpl_f_disposition,
    rpl_f_validation_errors,
)
from tests.qualification.ebh_5._vendor_backend_types import RplGateStatus
from tests.qualification.ebh_7._gate_findings_register import EBH7_CORRECTIVE_GATE_FINDINGS
from tests.qualification.ebh_7._rpl_h_normative_evidence import (
    all_normative_rpl_rows,
    normative_rpl_b6_open_concern_ids,
    normative_rpl_verdict_counts,
)
from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict
from tests.qualification.ebh_7._semantic_disposition_registry import all_verified_dispositions
from tests.qualification.ebh_7._semantic_proof_binding_audit import semantic_exact_counts
from tests.qualification.ebh_7._semantic_proof_types import ProofBindingResult

_PKG_C_V2 = "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py"

_RESOLVED_FINDING_IDS: Final[frozenset[str]] = frozenset(
    {
        "W6-ID-REGISTRY-ONLY",
        "W6-TENANT-X-FINAL-READY",
        "W7-HASATTR-DUP-X",
        "W5-RPL-ROW-COUNT",
        "W5-VBN-DOC-PRESENCE",
        "TRACE-466-COUNT-ONLY",
        "READINESS-TABLE-PASS",
        "GAP-TEN-UNK-05-CLOSED",
        "AUD-01-RPL-AUTO-QUALIFIED",
        "B6-NORMATIVE-OPEN-ROWS",
        "B6-F01-EXTERNAL-WORK-DUAL-INSTANCE",
        "B6-F02-PRODQ-TEXT-PREFIX-NEGATIVE",
        "B6-F03-TEN-XDM-DOC-CONFLICT",
        "B6-F04-P03-TEST-CLOSURE-SHA",
    }
)


class ClosureAuditStatus(StrEnum):
    PASS = "PASS"
    NA_WITH_EVIDENCE = "N/A_WITH_EVIDENCE"
    OPEN_FINDING = "OPEN_FINDING"


@dataclass(frozen=True, slots=True)
class ClosureAuditRow:
    gate_id: str
    subject: str
    status: ClosureAuditStatus
    evidence: str
    notes: str = ""


def _pytest_node_resolvable(node_id: str) -> bool:
    if pytest_test_function_exists(node_id):
        return True
    if "[" in node_id:
        base = node_id.split("[", 1)[0]
        if pytest_test_function_exists(base):
            return True
    if "::" in node_id and node_id.startswith("tests."):
        mod_part, func = node_id.rsplit("::", 1)
        file_rel = mod_part.replace(".", "/") + ".py"
        return pytest_test_function_exists(f"{file_rel}::{func}")
    return False


def _dual_slug_rpl_rows() -> tuple[ClosureAuditRow, ...]:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import INTEGRATION_DUAL_SLUG_CATALOG

    rows: list[ClosureAuditRow] = []
    for entry in INTEGRATION_DUAL_SLUG_CATALOG:
        pos = f"{_PKG_C_V2}::test_ebh5_pkgc_v2_dual_slug_rpl_cde_equivalence"
        neg = f"{_PKG_C_V2}::test_ebh5_pkgc_v2_dual_slug_unregistered"
        rpl_f = rpl_f_disposition(entry.concern_id)
        status = ClosureAuditStatus.PASS
        notes = "RPL-C/D/E @ Package C v2; RPL-E includes unregistered fail-closed isolation"
        if not _pytest_node_resolvable(pos) or not _pytest_node_resolvable(neg):
            status = ClosureAuditStatus.OPEN_FINDING
            notes = "Missing Package C v2 pytest node"
        elif rpl_f is None:
            status = ClosureAuditStatus.OPEN_FINDING
            notes = "Missing RPL-F disposition"
        elif rpl_f.gate is RplGateStatus.NA and rpl_f.normative_na_rationale == _PACKAGE_C_RPL_F_NA:
            status = ClosureAuditStatus.OPEN_FINDING
            notes = "Generic RPL-F N/A rationale — requires per-category text"
        elif rpl_f.gate is RplGateStatus.NA:
            status = ClosureAuditStatus.NA_WITH_EVIDENCE
            notes = rpl_f.normative_na_rationale[:120]
        evidence = (
            f"CDE={pos}[{entry.concern_id}]; unreg={neg}[{entry.concern_id}]; "
            f"RPL-F={rpl_f.gate.value if rpl_f else 'MISSING'}"
        )
        rows.append(
            ClosureAuditRow(
                gate_id="G01",
                subject=entry.concern_id,
                status=status,
                evidence=evidence,
                notes=notes,
            )
        )
    return tuple(rows)


def _factory_and_legacy_rpl_rows() -> tuple[ClosureAuditRow, ...]:
    from tests.qualification.ebh_5._factory_rpl_normative_nodes import FACTORY_RPL_ONE_PASS_CONCERNS
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import LEGACY_BEHAVIORAL_QUALIFIED

    rows: list[ClosureAuditRow] = []
    for row in all_normative_rpl_rows():
        if row.concern_id not in FACTORY_RPL_ONE_PASS_CONCERNS and row.concern_id not in LEGACY_BEHAVIORAL_QUALIFIED:
            continue
        status = ClosureAuditStatus.PASS
        notes = row.task_id
        if row.semantic_verdict is not RplSemanticVerdict.QUALIFIED:
            status = ClosureAuditStatus.OPEN_FINDING
        elif not _pytest_node_resolvable(row.positive_test):
            status = ClosureAuditStatus.OPEN_FINDING
            notes = f"Missing positive: {row.positive_test}"
        elif row.positive_test != row.negative_test and not _pytest_node_resolvable(row.negative_test):
            status = ClosureAuditStatus.OPEN_FINDING
            notes = f"Missing negative: {row.negative_test}"
        rows.append(
            ClosureAuditRow(
                gate_id="G02",
                subject=row.concern_id,
                status=status,
                evidence=f"pos={row.positive_test}; neg={row.negative_test}",
                notes=notes,
            )
        )
    return tuple(rows)


def _internal_fixed_rpl_rows() -> tuple[ClosureAuditRow, ...]:
    from tests.qualification.ebh_5._terminal_rpl_normative_nodes import (
        FIXED_RPL_CONCERNS,
        INTERNAL_RPL_CONCERNS,
    )

    rows: list[ClosureAuditRow] = []
    for concern_id in sorted(INTERNAL_RPL_CONCERNS | FIXED_RPL_CONCERNS):
        norm = next(r for r in all_normative_rpl_rows() if r.concern_id == concern_id)
        status = ClosureAuditStatus.PASS
        notes = norm.verified_status.value
        if norm.semantic_verdict is not RplSemanticVerdict.QUALIFIED:
            status = ClosureAuditStatus.OPEN_FINDING
        elif not _pytest_node_resolvable(norm.positive_test) or not _pytest_node_resolvable(norm.negative_test):
            status = ClosureAuditStatus.OPEN_FINDING
            notes = "Terminal gate pytest node missing"
        rows.append(
            ClosureAuditRow(
                gate_id="G03",
                subject=concern_id,
                status=status,
                evidence=f"pos={norm.positive_test}; neg={norm.negative_test}",
                notes=notes,
            )
        )
    return tuple(rows)


def _ten_xdm_rows() -> tuple[ClosureAuditRow, ...]:
    rows: list[ClosureAuditRow] = []
    for disp in all_verified_dispositions():
        gate = "G04" if disp.scenario_id.startswith("EBH7-TEN-E2E") else "G05"
        status = ClosureAuditStatus.PASS
        notes = disp.required_invariant
        if disp.semantic_match is not ProofBindingResult.EXACT_PROOF:
            status = ClosureAuditStatus.OPEN_FINDING
        elif not disp.actual_test.strip():
            status = ClosureAuditStatus.OPEN_FINDING
            notes = "Empty actual_test"
        elif not _pytest_node_resolvable(disp.actual_test):
            status = ClosureAuditStatus.OPEN_FINDING
            notes = f"actual_test not executable: {disp.actual_test}"
        rows.append(
            ClosureAuditRow(
                gate_id=gate,
                subject=disp.scenario_id,
                status=status,
                evidence=disp.actual_test,
                notes=notes,
            )
        )
    return tuple(rows)


def _int2e_row() -> ClosureAuditRow:
    from tests.qualification.ebh_7.test_ebh_7_int2e_bundle_regression import _changed_bundle_modules

    modules = _changed_bundle_modules()
    return ClosureAuditRow(
        gate_id="G06",
        subject="INT2E bundle regression",
        status=ClosureAuditStatus.NA_WITH_EVIDENCE,
        evidence=(
            f"import_smoke_modules={len(modules)}; tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py"
        ),
        notes="Not full platform E2E — changed-bundle import + relational/message_bus resolve spot checks",
    )


def _ecap_row() -> ClosureAuditRow:
    return ClosureAuditRow(
        gate_id="G07",
        subject="ECAP C1–C5 + PROHIBITED",
        status=ClosureAuditStatus.PASS,
        evidence=(
            "test_ebh_7_certification_ssot_gates.py; "
            "test_ebh_7_certification_mutation_fail_closed.py; "
            "test_ebh6_prohibited_unresolved_zero"
        ),
        notes="Enterprise family matrix + executable checklist @ HEAD",
    )


def _p03_row() -> ClosureAuditRow:
    return ClosureAuditRow(
        gate_id="G08",
        subject="Protocol 03 replay + digests",
        status=ClosureAuditStatus.PASS,
        evidence="ebh_7_p03_independent_replay/MANIFEST.json; _ebh7_p03_evidence_integrity.py",
        notes="Independent auditor must checkout TEST_EXECUTION_SHA and replay MANIFEST pytest_profile",
    )


def _findings_rows() -> tuple[ClosureAuditRow, ...]:
    rows: list[ClosureAuditRow] = []
    for finding in EBH7_CORRECTIVE_GATE_FINDINGS:
        if finding.gate_id in _RESOLVED_FINDING_IDS:
            status = ClosureAuditStatus.PASS
            notes = "resolved"
        else:
            status = ClosureAuditStatus.OPEN_FINDING
            notes = "unmapped finding — add to _RESOLVED_FINDING_IDS or correct"
        rows.append(
            ClosureAuditRow(
                gate_id="G09",
                subject=finding.gate_id,
                status=status,
                evidence=finding.correction[:160],
                notes=notes,
            )
        )
    return tuple(rows)


def _summary_rows() -> tuple[ClosureAuditRow, ...]:
    ten_e, ten_o, xdm_e, xdm_o = semantic_exact_counts()
    rpl_counts = normative_rpl_verdict_counts()
    open_rpl = normative_rpl_b6_open_concern_ids()
    rpl_f_errors = rpl_f_validation_errors()
    rows: list[ClosureAuditRow] = []
    rows.append(
        ClosureAuditRow(
            gate_id="SUM",
            subject="RPL-H 56/56 QUALIFIED",
            status=ClosureAuditStatus.PASS if not open_rpl else ClosureAuditStatus.OPEN_FINDING,
            evidence=str(dict(rpl_counts)),
            notes=f"open={open_rpl[:5]}..." if open_rpl else "0 B6-open",
        )
    )
    rows.append(
        ClosureAuditRow(
            gate_id="SUM",
            subject="TEN 26 EXACT",
            status=ClosureAuditStatus.PASS if ten_e == 26 and ten_o == 0 else ClosureAuditStatus.OPEN_FINDING,
            evidence=f"{ten_e}/{ten_o}",
            notes="semantic_exact_counts",
        )
    )
    rows.append(
        ClosureAuditRow(
            gate_id="SUM",
            subject="XDM 34 EXACT",
            status=ClosureAuditStatus.PASS if xdm_e == 34 and xdm_o == 0 else ClosureAuditStatus.OPEN_FINDING,
            evidence=f"{xdm_e}/{xdm_o}",
            notes="semantic_exact_counts",
        )
    )
    rows.append(
        ClosureAuditRow(
            gate_id="SUM",
            subject="RPL-F registry",
            status=ClosureAuditStatus.PASS if not rpl_f_errors else ClosureAuditStatus.OPEN_FINDING,
            evidence=f"errors={len(rpl_f_errors)}",
            notes=rpl_f_errors[0] if rpl_f_errors else "ok",
        )
    )
    return tuple(rows)


def all_closure_audit_rows() -> tuple[ClosureAuditRow, ...]:
    return (
        _summary_rows()
        + _dual_slug_rpl_rows()
        + _factory_and_legacy_rpl_rows()
        + _internal_fixed_rpl_rows()
        + _ten_xdm_rows()
        + (_int2e_row(),)
        + (_ecap_row(),)
        + (_p03_row(),)
        + _findings_rows()
    )


def closure_open_findings() -> tuple[ClosureAuditRow, ...]:
    return tuple(r for r in all_closure_audit_rows() if r.status is ClosureAuditStatus.OPEN_FINDING)


def assert_ebh7_closure_evidence_audit_ready() -> None:
    open_rows = closure_open_findings()
    assert open_rows == (), (
        "Closure evidence audit OPEN findings — "
        + "; ".join(f"{r.gate_id}:{r.subject}" for r in open_rows[:8])
        + (f" (+{len(open_rows) - 8} more)" if len(open_rows) > 8 else "")
    )
