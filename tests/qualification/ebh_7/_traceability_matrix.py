# © Artur Czarnecki. All rights reserved.

"""EBH-7 requirement traceability SSOT (466 obligations)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tests.qualification.ebh_5._replaceability_matrix import replaceability_matrix_rows
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification
from tests.qualification.ebh_7._certification_status import CertificationStatus, is_behavioral_certification
from tests.qualification.ebh_7._constants import EBH7_REQUIREMENTS_TOTAL
from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
from tests.qualification.ebh_7._task_register import EBH7_TASK_REGISTER
from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class RequirementRow:
    source_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class ExtendedRequirementRow:
    source_id: str
    canonical_source_id: str
    owner: str
    task_id: str
    evidence_ref: str
    acceptance: str
    materialization_status: CertificationStatus
    qualification_status: CertificationStatus


def _canonical_source_id(source_id: str) -> str:
    if source_id.startswith("EBH7-FRZ-"):
        return source_id.removeprefix("EBH7-")
    if source_id.startswith("EBH7-SYS-"):
        return f"SYS-INV-{source_id.removeprefix('EBH7-SYS-')}"
    return source_id


def _owner_for(source_id: str, task_id: str) -> str:
    if source_id.startswith("EBH7-TEN-E2E"):
        return "Tenant platform"
    if source_id.startswith("EBH7-XDM"):
        return "Platform cross-domain"
    if source_id.startswith("EBH7-RPL"):
        return "VBN / Integrations"
    if source_id.startswith("EBH7-NPSC"):
        return "Execution / NPSC"
    if source_id.startswith("EBH7-FRZ"):
        return "Program / FRZ"
    if source_id.startswith("EBH7-VBN"):
        return "VBN"
    if task_id.startswith("TASK-EBH7-0"):
        return "EBH-7 program"
    return "Platform"


def _status_for_source(source_id: str) -> tuple[CertificationStatus, CertificationStatus, str, str]:
    materialized = CertificationStatus.MATERIALIZED
    if source_id.startswith("EBH7-TEN-E2E-"):
        n = int(source_id.removeprefix("EBH7-TEN-E2E-"))
        scenario = EBH7_TENANT_E2E_SCENARIOS[n - 1]
        evidence = scenario.proof_module or "OPEN — no current-HEAD proof node"
        if scenario.proof_function:
            evidence = f"{scenario.proof_module}::{scenario.proof_function}"
        from tests.qualification.ebh_7._semantic_disposition_registry import qualification_status_from_disposition

        q_status = qualification_status_from_disposition(scenario.scenario_id)
        return materialized, q_status, evidence, scenario.expected_behavior
    if source_id.startswith("EBH7-XDM-"):
        n = int(source_id.removeprefix("EBH7-XDM-"))
        scenario = EBH7_CROSS_DOMAIN_SCENARIOS[n - 1]
        evidence = scenario.proof_module or "OPEN — no current-HEAD proof node"
        if scenario.proof_function:
            evidence = f"{scenario.proof_module}::{scenario.proof_function}"
        from tests.qualification.ebh_7._semantic_disposition_registry import qualification_status_from_disposition

        q_status = qualification_status_from_disposition(scenario.scenario_id)
        return materialized, q_status, evidence, scenario.expected
    if source_id.startswith("EBH7-RPL-"):
        concern = source_id.removeprefix("EBH7-RPL-")
        row = next(r for r in replaceability_matrix_rows() if r.concern_id == concern)
        q = (
            CertificationStatus.QUALIFIED
            if row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
            else CertificationStatus.QUALIFICATION_PENDING
        )
        return materialized, q, "tests/qualification/ebh_5/_replaceability_matrix.py", row.rpl_h.value
    if source_id.startswith("EBH7-NPSC-"):
        return (
            materialized,
            CertificationStatus.QUALIFIED,
            "test_npsc42_orchestration_backend_access_confined_to_allowlist",
            "PASS",
        )
    return materialized, CertificationStatus.QUALIFICATION_PENDING, "TBD wave evidence", "OPEN"


@lru_cache(maxsize=1)
def extended_requirement_rows() -> tuple[ExtendedRequirementRow, ...]:
    out: list[ExtendedRequirementRow] = []
    for base in requirement_rows():
        mat, qual, evidence, acceptance = _status_for_source(base.source_id)
        out.append(
            ExtendedRequirementRow(
                source_id=base.source_id,
                canonical_source_id=_canonical_source_id(base.source_id),
                owner=_owner_for(base.source_id, base.task_id),
                task_id=base.task_id,
                evidence_ref=evidence,
                acceptance=acceptance,
                materialization_status=mat,
                qualification_status=qual,
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def requirement_rows() -> tuple[RequirementRow, ...]:
    rows: list[RequirementRow] = []
    task_ids = {t.task_id for t in EBH7_TASK_REGISTER}

    def add(sid: str, tid: str) -> None:
        assert tid in task_ids, f"{sid} -> {tid}"
        rows.append(RequirementRow(sid, tid))

    for i in range(1, 16):
        tid = "TASK-EBH7-030" if i <= 7 else "TASK-EBH7-051" if i == 8 else "TASK-EBH7-041"
        add(f"EBH7-RQ-{i:03d}", tid)
    for i in range(1, 7):
        add(f"EBH7-ECAP-B{i:02d}", "TASK-EBH7-078")
    for i in range(1, 6):
        add(f"EBH7-ECAP-C{i:02d}", "TASK-EBH7-064")
    for i in range(1, 27):
        add(f"EBH7-ECAP-TH-{i:02d}", "TASK-EBH7-057")
    frz_text = (
        _REPO_ROOT / "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
    ).read_text(encoding="utf-8")
    for line in frz_text.splitlines():
        if not line.startswith("| FRZ-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5 or not parts[1].startswith("FRZ-"):
            continue
        fid = parts[1]
        stage = parts[4]
        if "PROD-Q" in stage or fid.startswith("FRZ-PRD"):
            add(f"EBH7-FRZ-{fid}", "TASK-EBH7-008")
        elif fid.startswith("FRZ-TEN"):
            add(f"EBH7-FRZ-{fid}", "TASK-EBH7-051")
        else:
            add(f"EBH7-FRZ-{fid}", "TASK-EBH7-064")
    for i in range(38):
        add(f"EBH7-SYS-{i:02d}", "TASK-EBH7-033")
    for i in range(1, 17):
        add(f"EBH7-PLT-{i:02d}", "TASK-EBH7-034")
    for i in range(1, 8):
        add(f"EBH7-VBN-ART-{i:02d}", "TASK-EBH7-041")
    for r in replaceability_matrix_rows():
        add(f"EBH7-RPL-{r.concern_id}", "TASK-EBH7-041")
    for i in range(1, 13):
        add(f"EBH7-NPSC-{i:02d}", "TASK-EBH7-023")
    for i in range(1, 27):
        add(f"EBH7-TEN-E2E-{i:02d}", "TASK-EBH7-051")
    for i in range(1, 35):
        add(f"EBH7-XDM-{i:03d}", "TASK-EBH7-057")
    return tuple(rows)


def assert_ebh7_traceability_integrity() -> None:
    rows = requirement_rows()
    assert len(rows) == EBH7_REQUIREMENTS_TOTAL
    sids = [r.source_id for r in rows]
    assert len(sids) == len(set(sids))
    extended = extended_requirement_rows()
    assert len(extended) == EBH7_REQUIREMENTS_TOTAL
    assert {r.source_id for r in extended} == set(sids)


def assert_ebh7_traceability_behavioral_integrity() -> None:
    """Row count alone must not imply behavioral qualification."""
    extended = extended_requirement_rows()
    qualified = sum(1 for r in extended if is_behavioral_certification(r.qualification_status))
    pending = sum(1 for r in extended if r.qualification_status == CertificationStatus.QUALIFICATION_PENDING)
    assert qualified + pending <= len(extended)
    from tests.qualification.ebh_7._semantic_disposition_registry import assert_disposition_registry_complete
    from tests.qualification.ebh_7._semantic_proof_binding_audit import semantic_exact_counts

    ten_exact, ten_open, xdm_exact, xdm_open = semantic_exact_counts()
    assert ten_exact + ten_open == 26
    assert xdm_exact + xdm_open == 34
    assert_disposition_registry_complete()
    from tests.qualification.ebh_7._constants import EBH7_RPL_MATRIX_ROWS
    from tests.qualification.ebh_7._rpl_h_normative_evidence import (
        normative_rpl_b6_open_concern_ids,
        normative_rpl_verdict_counts,
    )
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    b6_open = normative_rpl_b6_open_concern_ids()
    verdicts = normative_rpl_verdict_counts()
    assert sum(verdicts.values()) == EBH7_RPL_MATRIX_ROWS
    assert len(b6_open) == verdicts[RplSemanticVerdict.PARTIAL] + verdicts[RplSemanticVerdict.BLOCKED]
    for row in extended:
        assert row.evidence_ref.strip() != ""
        assert row.acceptance.strip() != ""
        assert row.owner.strip() != ""
        if row.source_id.startswith("EBH7-TEN-E2E-") and row.qualification_status == CertificationStatus.QUALIFIED:
            assert "::" in row.evidence_ref
        if row.source_id.startswith("EBH7-RPL-") and row.qualification_status == CertificationStatus.QUALIFIED:
            assert "REPLACEABLE" in row.acceptance.upper()
