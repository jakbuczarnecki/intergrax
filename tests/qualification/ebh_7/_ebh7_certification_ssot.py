# © Artur Czarnecki. All rights reserved.

"""Single certification SSOT — normative vs mechanical vs evidence layers (fail-closed)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def assert_ebh7_rpl_certification_ssot() -> None:
    from tests.qualification.ebh_7._certification_status import CertificationStatus, is_behavioral_certification
    from tests.qualification.ebh_7._ebh7_scope_residual_register import allowed_non_demonstrated_concern_ids
    from tests.qualification.ebh_7._rpl_h_disposition import rpl_h_disposition_rows
    from tests.qualification.ebh_7._rpl_h_normative_evidence import (
        all_normative_rpl_rows,
        assert_ebh7_rpl_h_normative_evidence_integrity,
    )
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict, rpl_semantic_verdict_counts

    assert_ebh7_rpl_h_normative_evidence_integrity()
    agent = rpl_semantic_verdict_counts()
    norm = _normative_semantic_counts()
    assert agent == norm, f"Agent vs normative RPL mismatch: agent={agent} norm={norm}"

    allowed = allowed_non_demonstrated_concern_ids()
    for disp in rpl_h_disposition_rows():
        norm = next(r for r in all_normative_rpl_rows() if r.concern_id == disp.concern_id)
        if norm.semantic_verdict is RplSemanticVerdict.QUALIFIED:
            assert is_behavioral_certification(disp.qualification_status), norm.concern_id
        elif norm.concern_id in allowed:
            assert disp.qualification_status is CertificationStatus.QUALIFICATION_PENDING
        else:
            assert not is_behavioral_certification(disp.qualification_status), norm.concern_id


def _normative_semantic_counts():
    from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_verdict_counts

    return normative_rpl_verdict_counts()


def assert_ebh7_ten_xdm_evidence_ssot() -> None:
    from tests.qualification.ebh_7._ebh7_scenario_evidence_validation import (
        assert_ebh7_ten_xdm_evidence_validation_complete,
        ten_xdm_evidence_summary,
    )
    from tests.qualification.ebh_7._ebh7_scenario_normative_audit import ScenarioVerifiedStatus

    assert_ebh7_ten_xdm_evidence_validation_complete()
    summary = ten_xdm_evidence_summary()
    proxy = summary["XDM"].get(ScenarioVerifiedStatus.COMPOSED_PROXY.value, 0)
    assert proxy == 0, f"XDM COMPOSED_PROXY must be 0 for closure SSOT, got {proxy}"


def assert_ebh7_p03_evidence_ssot() -> None:
    from tests.qualification.ebh_7._ebh7_p03_evidence_integrity import assert_ebh7_p03_evidence_integrity_bundle

    assert_ebh7_p03_evidence_integrity_bundle()


def assert_ebh7_certification_ssot_bundle() -> None:
    assert_ebh7_rpl_certification_ssot()
    assert_ebh7_ten_xdm_evidence_ssot()
    assert_ebh7_p03_evidence_ssot()
