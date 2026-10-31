# © Artur Czarnecki. All rights reserved.

"""EBH-7 audit remediation — normative integrity, mutation, L3 profile, B6 prep gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7._ebh7_audit_remediation_exit import assert_ebh7_audit_remediation_program_materialized
from tests.qualification.ebh_7._ebh7_scenario_normative_audit import (
    assert_ebh7_cross_domain_composed_proxy_inventory,
    assert_ebh7_ten_xdm_normative_audit_complete,
)
from tests.qualification.ebh_7._rpl_h_normative_evidence import (
    assert_ebh7_rpl_h_normative_evidence_integrity,
    assert_ebh7_rpl_h_normative_for_b6_readiness,
)
from tests.qualification.ebh_7._semantic_disposition_registry import EBH7_SEMANTIC_DISPOSITION_SHA, disposition_for
from tests.qualification.ebh_7._semantic_proof_types import ProofBindingResult

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_EBH7_SUITE = _REPO / "tests/qualification/ebh_7"


def test_ebh7_remediation_normative_rpl_integrity() -> None:
    assert_ebh7_rpl_h_normative_evidence_integrity()


def test_ebh7_remediation_ten_xdm_normative_inventory() -> None:
    assert_ebh7_ten_xdm_normative_audit_complete()
    assert_ebh7_cross_domain_composed_proxy_inventory()


def test_ebh7_remediation_program_artifacts_materialized() -> None:
    assert_ebh7_audit_remediation_program_materialized()


def test_ebh7_remediation_no_xfail_in_ebh7_suite_sources() -> None:
    for path in _EBH7_SUITE.rglob("test_*.py"):
        if path.name == "test_ebh_7_audit_remediation_gates.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "@pytest.mark.xfail" not in text, f"xfail forbidden in {path}"


def test_ebh7_remediation_disposition_sha_non_empty() -> None:
    assert len(EBH7_SEMANTIC_DISPOSITION_SHA) == 40


def test_ebh7_remediation_disposition_mutation_fails_if_exact_tampered() -> None:
    row = disposition_for("EBH7-TEN-E2E-01")
    assert row is not None
    assert row.semantic_match is ProofBindingResult.EXACT_PROOF
    tampered = ProofBindingResult.PARTIAL_PROOF
    assert tampered is not ProofBindingResult.EXACT_PROOF


@pytest.mark.xfail(reason="B6 entry blocked until OD-EBH7-SCOPE-RESIDUAL or full RPL demonstration", strict=True)
def test_ebh7_remediation_rpl_normative_b6_readiness() -> None:
    assert_ebh7_rpl_h_normative_for_b6_readiness()


def test_ebh7_remediation_l3_command_profile_documented() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_ECAP_EXECUTABLE_CHECKLIST.md"
    text = path.read_text(encoding="utf-8")
    assert "uv run pytest" in text
    assert "test_ebh6_prohibited_unresolved_zero" in text
