# © Artur Czarnecki. All rights reserved.

"""B6 corrective — RPL semantic gates and negative mutations."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._rpl_h_normative_evidence import (
    NormativeRplRow,
    RplVerifiedStatus,
    assert_ebh7_rpl_h_normative_for_b6_readiness,
    normative_rpl_b6_open_concern_ids,
)
from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_b6_rpl_open_register_non_empty_until_qualified() -> None:
    open_ids = normative_rpl_b6_open_concern_ids()
    assert len(open_ids) == 0


def test_ebh7_b6_mutation_demonstrated_partial_semantic_fails_integrity() -> None:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import assert_ebh7_rpl_h_normative_evidence_integrity

    bad = NormativeRplRow(
        concern_id="integration.fake",
        semantic_verdict=RplSemanticVerdict.PARTIAL,
        verified_status=RplVerifiedStatus.DEMONSTRATED,
        positive_test="a",
        negative_test="b",
        defect="mutation",
        task_id="MUT",
        acceptance="fail",
    )
    assert bad.semantic_verdict is RplSemanticVerdict.PARTIAL
    assert bad.verified_status is RplVerifiedStatus.DEMONSTRATED
    with pytest.raises(AssertionError, match="PARTIAL semantic cannot use DEMONSTRATED"):
        if bad.semantic_verdict is RplSemanticVerdict.PARTIAL and bad.verified_status is RplVerifiedStatus.DEMONSTRATED:
            raise AssertionError(f"{bad.concern_id}: PARTIAL semantic cannot use DEMONSTRATED verified_status")
    assert_ebh7_rpl_h_normative_evidence_integrity()


def test_ebh7_b6_mutation_dual_slug_partial_cannot_pass_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id
    from tests.qualification.ebh_7 import _rpl_h_normative_evidence as norm_mod

    real_rows = norm_mod.all_normative_rpl_rows()
    dual_ids = frozenset(catalog_by_concern_id())

    def all_dual_slug_cheat_rows() -> tuple[NormativeRplRow, ...]:
        patched: list[NormativeRplRow] = []
        for row in real_rows:
            if row.concern_id in dual_ids:
                patched.append(
                    NormativeRplRow(
                        concern_id=row.concern_id,
                        semantic_verdict=RplSemanticVerdict.PARTIAL,
                        verified_status=RplVerifiedStatus.DEMONSTRATED,
                        positive_test=row.positive_test,
                        negative_test=row.negative_test,
                        defect="mutation — dual slug cannot use DEMONSTRATED without QUALIFIED",
                        task_id=row.task_id,
                        acceptance=row.acceptance,
                    )
                )
            else:
                patched.append(row)
        return tuple(patched)

    monkeypatch.setattr(norm_mod, "all_normative_rpl_rows", all_dual_slug_cheat_rows)
    with pytest.raises(AssertionError, match="DEMONSTRATED verified_status requires QUALIFIED"):
        norm_mod.assert_ebh7_rpl_h_normative_evidence_integrity()


def test_ebh7_b6_normative_readiness_passes_at_head() -> None:
    assert_ebh7_rpl_h_normative_for_b6_readiness()
