# © Artur Czarnecki. All rights reserved.

"""F07 — 28-row dual-slug RPL-E/F reconciliation matrix (SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.ebh_5._dual_slug_rpl_ef_contract_evidence import (
    canonical_rpl_e_guarantees,
    dual_slug_rpl_ef_positive_negative_nodes,
)
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import (
    INTEGRATION_DUAL_SLUG_CATALOG,
    IntegrationDualSlugCatalogEntry,
)
from tests.qualification.ebh_5._rpl_f_evidence_registry import rpl_f_disposition
from tests.qualification.ebh_5._vendor_backend_types import RplGateStatus


class F07Verdict(StrEnum):
    QUALIFIED = "QUALIFIED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class DualSlugRplEfMatrixRow:
    concern_id: str
    contract: str
    claimed_guarantee: str
    rpl_e_proof: str
    rpl_f_obligation: str
    rpl_f_proof_or_na: str
    executed_test: str
    verdict: F07Verdict


def _row_for_entry(entry: IntegrationDualSlugCatalogEntry) -> DualSlugRplEfMatrixRow:
    from intergrax.runtime.integrations.contract_metadata import contract_for_category

    contract_cls = contract_for_category(entry.category.value)
    pos, neg = dual_slug_rpl_ef_positive_negative_nodes(entry)
    guarantees = canonical_rpl_e_guarantees(entry)
    rpl_f = rpl_f_disposition(entry.concern_id)
    if rpl_f is None:
        rpl_f_text = "MISSING"
        verdict = F07Verdict.BLOCKED
    elif rpl_f.gate is RplGateStatus.PASS:
        rpl_f_text = "; ".join(rpl_f.proof_pytest_nodes) or "PASS without nodes"
        verdict = F07Verdict.QUALIFIED
    elif rpl_f.gate is RplGateStatus.NA:
        rpl_f_text = rpl_f.normative_na_rationale[:160]
        verdict = (
            F07Verdict.QUALIFIED
            if rpl_f.normative_na_rationale.startswith("N/A — WITH EVIDENCE")
            else F07Verdict.BLOCKED
        )
    else:
        rpl_f_text = rpl_f.gate.value
        verdict = F07Verdict.BLOCKED
    return DualSlugRplEfMatrixRow(
        concern_id=entry.concern_id,
        contract=contract_cls.__name__,
        claimed_guarantee=guarantees[-1],
        rpl_e_proof=f"{pos}; {neg}; Package C v2 CDE + operational surface @ _dual_slug_rpl_e_operational.py",
        rpl_f_obligation="Active failure/recovery when replaceability claims operational fault semantics",
        rpl_f_proof_or_na=rpl_f_text,
        executed_test=pos.split("[", 1)[0],
        verdict=verdict,
    )


def all_dual_slug_rpl_ef_matrix_rows() -> tuple[DualSlugRplEfMatrixRow, ...]:
    return tuple(_row_for_entry(e) for e in INTEGRATION_DUAL_SLUG_CATALOG)


def assert_dual_slug_rpl_ef_matrix_complete() -> None:
    rows = all_dual_slug_rpl_ef_matrix_rows()
    assert len(rows) == 28
    blocked = [r.concern_id for r in rows if r.verdict is not F07Verdict.QUALIFIED]
    assert not blocked, f"F07 matrix BLOCKED concerns: {blocked[:6]}"


def render_dual_slug_rpl_ef_matrix_markdown(*, test_execution_sha: str) -> str:
    lines = [
        "# EBH-7 — Dual-slug RPL-E/F matrix (F07)",
        "",
        f"**TEST_EXECUTION_SHA:** `{test_execution_sha}`",
        "",
        "| CONCERN | CONTRACT | CLAIMED GUARANTEE | RPL-E PROOF | RPL-F OBLIGATION | "
        "RPL-F PROOF/NA | EXECUTED TEST | VERDICT |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in all_dual_slug_rpl_ef_matrix_rows():

        def esc(s: str) -> str:
            t = s.replace("|", "\\|")
            return t[:90] + "..." if len(t) > 90 else t

        lines.append(
            f"| {row.concern_id} | {row.contract} | {esc(row.claimed_guarantee)} | "
            f"{esc(row.rpl_e_proof)} | {esc(row.rpl_f_obligation)} | {esc(row.rpl_f_proof_or_na)} | "
            f"{esc(row.executed_test)} | {row.verdict.value} |"
        )
    lines.append("")
    return "\n".join(lines)
