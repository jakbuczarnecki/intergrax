# © Artur Czarnecki. All rights reserved.

"""Normative RPL-H evidence classification — independent of agent counter greenwashing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id
from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
from tests.qualification.ebh_7._rpl_h_semantic_evidence import (
    BLOCKED_CONCERN_ROOT_CAUSE,
    EBH7_RPL_BATCH_C_EVIDENCE_TEST_NODE,
    LEGACY_BEHAVIORAL_QUALIFIED,
    RplSemanticVerdict,
)


class RplVerifiedStatus(StrEnum):
    DEMONSTRATED = "DEMONSTRATED"
    FACTORY_ONLY = "FACTORY_ONLY"
    DUAL_SLUG_REGISTRY_ONLY = "DUAL_SLUG_REGISTRY_ONLY"
    COMPOSED_PROXY = "COMPOSED_PROXY"
    FIXED_SUBSTRATE_PENDING_OD = "FIXED_SUBSTRATE_PENDING_OD"
    FIXED_SUBSTRATE_RATIFIED = "FIXED_SUBSTRATE_RATIFIED"
    INTERNAL_RATIFIED = "INTERNAL_RATIFIED"
    PRODUCTION_GAP = "PRODUCTION_GAP"
    BLOCKED_ARCHITECTURE = "BLOCKED_ARCHITECTURE"


@dataclass(frozen=True, slots=True)
class NormativeRplRow:
    concern_id: str
    semantic_verdict: RplSemanticVerdict
    verified_status: RplVerifiedStatus
    positive_test: str
    negative_test: str
    defect: str
    task_id: str
    acceptance: str


def _classify(concern_id: str) -> NormativeRplRow:
    from tests.qualification.ebh_5._terminal_rpl_normative_nodes import (
        FIXED_RPL_CONCERNS,
        INTERNAL_RPL_CONCERNS,
        fixed_rpl_negative_node,
        fixed_rpl_positive_node,
        internal_rpl_negative_node,
        internal_rpl_positive_node,
    )

    if concern_id in LEGACY_BEHAVIORAL_QUALIFIED:
        node = QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE[concern_id]
        return NormativeRplRow(
            concern_id=concern_id,
            semantic_verdict=RplSemanticVerdict.QUALIFIED,
            verified_status=RplVerifiedStatus.DEMONSTRATED,
            positive_test=node,
            negative_test=node,
            defect="Legacy EBH-5 behavioral — maintain mutation gates",
            task_id="EBH7-W2-RPL-LEGACY",
            acceptance="Distinct negative + RPL-F where applicable",
        )
    if concern_id in INTERNAL_RPL_CONCERNS:
        pos = internal_rpl_positive_node(concern_id)
        neg = internal_rpl_negative_node(concern_id)
        return NormativeRplRow(
            concern_id=concern_id,
            semantic_verdict=RplSemanticVerdict.QUALIFIED,
            verified_status=RplVerifiedStatus.INTERNAL_RATIFIED,
            positive_test=pos,
            negative_test=neg,
            defect="EBH-7 INTERNAL ratified @ operator ADR — extensibility not claimed",
            task_id="EBH7-TERMINAL-INTERNAL",
            acceptance="ADR + tier ownership gate; no alternate backend",
        )
    if concern_id in FIXED_RPL_CONCERNS:
        pos = fixed_rpl_positive_node(concern_id)
        neg = fixed_rpl_negative_node(concern_id)
        return NormativeRplRow(
            concern_id=concern_id,
            semantic_verdict=RplSemanticVerdict.QUALIFIED,
            verified_status=RplVerifiedStatus.FIXED_SUBSTRATE_RATIFIED,
            positive_test=pos,
            negative_test=neg,
            defect="EBH-7 FIXED substrate ratified @ OD-RPL-G2 — not vendor-equivalent",
            task_id="EBH7-TERMINAL-FIXED",
            acceptance="Fixed substrate decision record + fail-closed contract gate",
        )
    if concern_id in BLOCKED_CONCERN_ROOT_CAUSE:
        return NormativeRplRow(
            concern_id=concern_id,
            semantic_verdict=RplSemanticVerdict.BLOCKED,
            verified_status=RplVerifiedStatus.FIXED_SUBSTRATE_PENDING_OD,
            positive_test="NONE",
            negative_test="NONE",
            defect=BLOCKED_CONCERN_ROOT_CAUSE[concern_id],
            task_id="EBH7-W2-RPL-BLOCKED",
            acceptance="OD-RPL-G2 fixed substrate OR second provider sandbox",
        )
    from tests.qualification.ebh_5._factory_rpl_normative_nodes import (
        FACTORY_RPL_ONE_PASS_CONCERNS,
        factory_rpl_negative_node,
        factory_rpl_positive_node,
    )

    if concern_id in FACTORY_RPL_ONE_PASS_CONCERNS:
        return NormativeRplRow(
            concern_id=concern_id,
            semantic_verdict=RplSemanticVerdict.QUALIFIED,
            verified_status=RplVerifiedStatus.DEMONSTRATED,
            positive_test=factory_rpl_positive_node(concern_id),
            negative_test=factory_rpl_negative_node(concern_id),
            defect="EBH-7 one-pass factory RPL — distinct pos/neg @ sanctioned contracts",
            task_id="EBH7-ONE-PASS-FACTORY",
            acceptance="Adapter/registry/env/distributed/tool behavioral equivalence",
        )
    if concern_id in catalog_by_concern_id():
        entry = catalog_by_concern_id()[concern_id]
        positive = (
            f"{EBH7_RPL_BATCH_C_EVIDENCE_TEST_NODE}"
            f"::test_ebh5_pkgc_v2_dual_slug_rpl_cde_equivalence[{entry.concern_id}]"
        )
        negative = (
            f"{EBH7_RPL_BATCH_C_EVIDENCE_TEST_NODE}::test_ebh5_pkgc_v2_dual_slug_unregistered[{entry.concern_id}]"
        )
        return NormativeRplRow(
            concern_id=concern_id,
            semantic_verdict=RplSemanticVerdict.QUALIFIED,
            verified_status=RplVerifiedStatus.DEMONSTRATED,
            positive_test=positive,
            negative_test=negative,
            defect="RPL-C/D/E equivalence @ Package C v2 dual-slug catalog",
            task_id="EBH7-FC-C-RPL-CDE",
            acceptance="Profile selection, lifecycle parity, fail-closed isolation",
        )
    if concern_id in QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE:
        node = QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE[concern_id]
        same_node = True
        return NormativeRplRow(
            concern_id=concern_id,
            semantic_verdict=RplSemanticVerdict.PARTIAL,
            verified_status=RplVerifiedStatus.FACTORY_ONLY,
            positive_test=node,
            negative_test=node,
            defect="Factory reachability — no distinct alternate backend proof"
            if same_node
            else "Behavioral node without normative alternate",
            task_id="EBH7-W2-RPL-FACT",
            acceptance="Distinct positive/negative or fixed-substrate authorization",
        )
    return NormativeRplRow(
        concern_id=concern_id,
        semantic_verdict=RplSemanticVerdict.PARTIAL,
        verified_status=RplVerifiedStatus.PRODUCTION_GAP,
        positive_test="tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py",
        negative_test="tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py",
        defect="No Package C behavioral node",
        task_id="EBH7-W2-RPL-GAP",
        acceptance="Package C behavioral proof @ sanctioned path",
    )


def normative_rpl_row(concern_id: str) -> NormativeRplRow:
    return _classify(concern_id)


def all_normative_rpl_rows() -> tuple[NormativeRplRow, ...]:
    return tuple(_classify(row.concern_id) for row in CROSS_LAYER_REPLACEABILITY_MATRIX)


def normative_rpl_verdict_counts() -> dict[RplSemanticVerdict, int]:
    counts = {v: 0 for v in RplSemanticVerdict}
    for row in all_normative_rpl_rows():
        counts[row.semantic_verdict] += 1
    return counts


def normative_rpl_b6_open_concern_ids() -> tuple[str, ...]:
    """Concerns blocking B6 until QUALIFIED or formally scoped OD residual."""
    from tests.qualification.ebh_7._ebh7_scope_residual_register import allowed_non_demonstrated_concern_ids

    allowed = allowed_non_demonstrated_concern_ids()
    return tuple(
        sorted(
            row.concern_id
            for row in all_normative_rpl_rows()
            if row.semantic_verdict is not RplSemanticVerdict.QUALIFIED and row.concern_id not in allowed
        )
    )


def assert_ebh7_rpl_h_normative_evidence_integrity() -> None:
    """Fail-closed: normative layer must not regress to auto-QUALIFIED factory/blocked paths."""
    for row in all_normative_rpl_rows():
        if row.concern_id in BLOCKED_CONCERN_ROOT_CAUSE:
            assert row.semantic_verdict is RplSemanticVerdict.BLOCKED
        if row.verified_status is RplVerifiedStatus.FACTORY_ONLY:
            assert row.semantic_verdict is RplSemanticVerdict.PARTIAL
        if row.verified_status in (
            RplVerifiedStatus.INTERNAL_RATIFIED,
            RplVerifiedStatus.FIXED_SUBSTRATE_RATIFIED,
        ):
            assert row.semantic_verdict is RplSemanticVerdict.QUALIFIED
        if row.verified_status is RplVerifiedStatus.DEMONSTRATED:
            assert row.semantic_verdict is RplSemanticVerdict.QUALIFIED, (
                f"{row.concern_id}: DEMONSTRATED verified_status requires QUALIFIED semantic verdict"
            )
        if row.semantic_verdict is RplSemanticVerdict.PARTIAL and row.verified_status is RplVerifiedStatus.DEMONSTRATED:
            raise AssertionError(f"{row.concern_id}: PARTIAL semantic cannot use DEMONSTRATED verified_status")
        if row.positive_test == row.negative_test and row.semantic_verdict is RplSemanticVerdict.QUALIFIED:
            if row.concern_id not in LEGACY_BEHAVIORAL_QUALIFIED:
                raise AssertionError(f"{row.concern_id}: QUALIFIED with identical pos/neg nodes")


def assert_ebh7_rpl_h_normative_for_b6_readiness() -> None:
    """B6 readiness — final RPL-H semantic disposition QUALIFIED or authorized scope residual only."""
    open_rows = normative_rpl_b6_open_concern_ids()
    assert open_rows == (), f"RPL B6 blockers (semantic not QUALIFIED): {open_rows[:12]}... ({len(open_rows)})"
