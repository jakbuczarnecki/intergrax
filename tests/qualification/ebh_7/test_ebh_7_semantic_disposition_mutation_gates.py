# © Artur Czarnecki. All rights reserved.

"""Disposition SSOT mutation gates — fail-closed detection of false EXACT / stale SHA."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._semantic_disposition_registry import (
    EBH7_SEMANTIC_DISPOSITION_SHA,
    assert_disposition_sha_current,
    disposition_for,
)
from tests.qualification.ebh_7._semantic_proof_types import ProofBindingResult

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_mutation_stale_sha_detected() -> None:
    assert_disposition_sha_current()
    assert EBH7_SEMANTIC_DISPOSITION_SHA != "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def test_ebh7_mutation_exact_requires_proof_node() -> None:
    row = disposition_for("EBH7-TEN-E2E-01")
    assert row is not None
    assert row.semantic_match is ProofBindingResult.EXACT_PROOF
    assert row.actual_test != "MISSING"
    assert "MISSING" not in row.assertions_inspected


def test_ebh7_mutation_tampered_audit_open_id_fails_mandatory_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.qualification.ebh_7 import _semantic_proof_binding_audit as audit_mod
    from tests.qualification.ebh_7._semantic_proof_binding_audit import (
        assert_ebh7_semantic_mandatory_exact_coverage,
    )
    from tests.qualification.ebh_7._semantic_proof_types import ProofBindingAuditRow, ProofBindingResult

    real_audit = audit_mod.EBH7_PROOF_BINDING_AUDIT
    poison = ProofBindingAuditRow(
        scenario_id="EBH7-TEN-E2E-99-MUT",
        claimed_boundary="Synthetic",
        attack="cross",
        expected_behavior="DENY",
        existing_proof="tests.fake::test_noop",
        actual_assertions="none",
        semantic_match=ProofBindingResult.PARTIAL_PROOF,
        notes="SHA=mut;",
    )
    monkeypatch.setattr(audit_mod, "EBH7_PROOF_BINDING_AUDIT", real_audit + (poison,))
    with pytest.raises(AssertionError, match="EXACT blockers"):
        assert_ebh7_semantic_mandatory_exact_coverage()
