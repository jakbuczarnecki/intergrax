# © Artur Czarnecki. All rights reserved.

"""Single certification SSOT — normative vs mechanical vs evidence layers (fail-closed)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

_REPO = Path(__file__).resolve().parents[3]
_EXPECTED_CLOSURE_BASELINE_SHA: Final[str] = "cb3fcb3c93ff361a138d36136cba0c8d6a5b59bd"


def current_head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def assert_ebh7_rpl_certification_ssot() -> None:
    from tests.qualification.ebh_7._certification_status import CertificationStatus, is_behavioral_certification
    from tests.qualification.ebh_7._ebh7_scope_residual_register import allowed_non_demonstrated_concern_ids
    from tests.qualification.ebh_7._rpl_h_disposition import rpl_h_disposition_rows
    from tests.qualification.ebh_7._rpl_h_normative_evidence import (
        RplVerifiedStatus,
        all_normative_rpl_rows,
        assert_ebh7_rpl_h_normative_evidence_integrity,
    )
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import rpl_semantic_verdict_counts

    assert_ebh7_rpl_h_normative_evidence_integrity()
    agent = rpl_semantic_verdict_counts()
    norm = _normative_semantic_counts()
    assert agent == norm, f"Agent vs normative RPL mismatch: agent={agent} norm={norm}"

    allowed = allowed_non_demonstrated_concern_ids()
    for disp in rpl_h_disposition_rows():
        norm = next(r for r in all_normative_rpl_rows() if r.concern_id == disp.concern_id)
        if norm.verified_status is RplVerifiedStatus.DEMONSTRATED:
            assert is_behavioral_certification(disp.qualification_status), (
                f"{disp.concern_id}: normative DEMONSTRATED but disposition not QUALIFIED"
            )
        elif norm.concern_id in allowed:
            assert disp.qualification_status is CertificationStatus.QUALIFICATION_PENDING


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


def assert_ebh7_closure_documentation_pins_head() -> None:
    head = current_head_sha()
    manifest = (_REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json").read_text(
        encoding="utf-8"
    )
    assert head in manifest or _EXPECTED_CLOSURE_BASELINE_SHA in manifest, (
        "MANIFEST must pin TEST_EXECUTION_SHA to current HEAD after Package G"
    )
    ladder = (_REPO / "docs/project/maintainers/qualification/EBH_7_SHA_LADDER.md").read_text(encoding="utf-8")
    assert "6758b6e1b" not in ladder.split("TEST_EXECUTION_SHA")[1][:200], (
        "SHA ladder must not retain stale 6758b6e1b TEST_EXECUTION pin"
    )


def assert_ebh7_certification_ssot_bundle() -> None:
    assert_ebh7_rpl_certification_ssot()
    assert_ebh7_ten_xdm_evidence_ssot()
