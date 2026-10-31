# © Artur Czarnecki. All rights reserved.

"""Fail-closed mutation tests for mandatory certification gates."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_mutation_scope_residual_empty_when_not_approved() -> None:
    from tests.qualification.ebh_7._ebh7_scope_residual_register import (
        OD_EBH7_SCOPE_RESIDUAL_APPROVED,
        allowed_non_demonstrated_concern_ids,
    )

    assert OD_EBH7_SCOPE_RESIDUAL_APPROVED is False
    assert allowed_non_demonstrated_concern_ids() == frozenset()


def test_ebh7_mutation_scope_residual_poison_fails_b6(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.qualification.ebh_7 import _ebh7_scope_residual_register as reg
    from tests.qualification.ebh_7 import _rpl_h_normative_evidence as norm_mod
    from tests.qualification.ebh_7._rpl_h_normative_evidence import assert_ebh7_rpl_h_normative_for_b6_readiness
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    real_rows = norm_mod.all_normative_rpl_rows()
    poisoned = tuple(
        norm_mod.NormativeRplRow(
            concern_id=real_rows[0].concern_id,
            semantic_verdict=RplSemanticVerdict.PARTIAL,
            verified_status=norm_mod.RplVerifiedStatus.FACTORY_ONLY,
            positive_test=real_rows[0].positive_test,
            negative_test=real_rows[0].negative_test,
            defect="mutation partial",
            task_id="MUT",
            acceptance="fail",
        )
        if i == 0
        else row
        for i, row in enumerate(real_rows)
    )
    monkeypatch.setattr(norm_mod, "all_normative_rpl_rows", lambda: poisoned)
    monkeypatch.setattr(reg, "OD_EBH7_SCOPE_RESIDUAL_APPROVED", True)
    monkeypatch.setattr(reg, "_SCOPE_RESIDUAL_CONCERN_IDS", frozenset({"integration.fake_concern"}))
    with pytest.raises(AssertionError, match="RPL B6 blockers"):
        assert_ebh7_rpl_h_normative_for_b6_readiness()


def test_ebh7_mutation_disposition_sha_poison(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.qualification.ebh_7 import _semantic_disposition_registry as reg
    from tests.qualification.ebh_7._semantic_disposition_registry import assert_disposition_sha_current

    monkeypatch.setattr(reg, "EBH7_SEMANTIC_DISPOSITION_SHA", "deadbeef" * 5)
    with pytest.raises(AssertionError):
        assert_disposition_sha_current()


def test_ebh7_mutation_composed_proxy_fails_ssot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.qualification.ebh_7 import _ebh7_scenario_evidence_validation as ev
    from tests.qualification.ebh_7._ebh7_certification_ssot import assert_ebh7_ten_xdm_evidence_ssot
    from tests.qualification.ebh_7._ebh7_scenario_normative_audit import ScenarioVerifiedStatus

    real = ev.ten_xdm_evidence_summary

    def poison() -> dict[str, dict[str, int]]:
        s = real()
        s["XDM"][ScenarioVerifiedStatus.COMPOSED_PROXY.value] = 1
        return s

    monkeypatch.setattr(ev, "ten_xdm_evidence_summary", poison)
    with pytest.raises(AssertionError, match="COMPOSED_PROXY"):
        assert_ebh7_ten_xdm_evidence_ssot()
