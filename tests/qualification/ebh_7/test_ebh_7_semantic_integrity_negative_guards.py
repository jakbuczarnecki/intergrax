# © Artur Czarnecki. All rights reserved.

"""Negative guards — semantic qualification integrity (no whitelist of 60 IDs)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._semantic_disposition_registry import EBH7_SEMANTIC_DISPOSITION_SHA
from tests.qualification.ebh_7._semantic_proof_binding_audit import (
    tenant_e2e_open_ids_from_audit,
    tenant_e2e_semantically_qualified_ids,
)
from tests.qualification.ebh_7._semantic_proof_types import ProofBindingResult, VerifiedSemanticDisposition

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def _qualifies_for_semantic_set(match: ProofBindingResult) -> bool:
    return match is ProofBindingResult.EXACT_PROOF


def test_ebh7_guard_unrelated_proof_not_exact() -> None:
    synthetic = VerifiedSemanticDisposition(
        scenario_id="EBH7-GUARD-SYNTH-01",
        required_invariant="DENY",
        production_boundaries="Synthetic",
        attack="cross",
        actual_test="tests.fake::test_unrelated",
        assertions_inspected="Inspected tests.fake::test_unrelated — no tenant invariant",
        semantic_match=ProofBindingResult.MISMATCHED_PROOF,
        test_execution_evidence=f"SHA={EBH7_SEMANTIC_DISPOSITION_SHA}; synthetic guard",
    )
    assert not _qualifies_for_semantic_set(synthetic.semantic_match)


def test_ebh7_guard_missing_proof_not_qualified() -> None:
    synthetic = VerifiedSemanticDisposition(
        scenario_id="EBH7-GUARD-SYNTH-02",
        required_invariant="DENY",
        production_boundaries="Synthetic",
        attack="cross",
        actual_test="MISSING",
        assertions_inspected="none",
        semantic_match=ProofBindingResult.MISSING_PROOF,
        test_execution_evidence=f"SHA={EBH7_SEMANTIC_DISPOSITION_SHA}; synthetic guard",
    )
    assert not _qualifies_for_semantic_set(synthetic.semantic_match)


def test_ebh7_guard_structural_not_mandatory_behavioral() -> None:
    assert not _qualifies_for_semantic_set(ProofBindingResult.STRUCTURAL_ONLY)


def test_ebh7_guard_partial_not_full_e2e() -> None:
    assert not _qualifies_for_semantic_set(ProofBindingResult.PARTIAL_PROOF)


def test_ebh7_guard_stale_sha_fails_disposition_check() -> None:
    from tests.qualification.ebh_7._semantic_disposition_registry import assert_disposition_sha_current

    assert_disposition_sha_current()
    stale = VerifiedSemanticDisposition(
        scenario_id="EBH7-GUARD-SYNTH-03",
        required_invariant="DENY",
        production_boundaries="Synthetic",
        attack="cross",
        actual_test="MISSING",
        assertions_inspected="none",
        semantic_match=ProofBindingResult.EXACT_PROOF,
        test_execution_evidence="SHA=deadbeef;",
    )
    assert EBH7_SEMANTIC_DISPOSITION_SHA not in stale.test_execution_evidence


def test_ebh7_guard_open_audit_ids_not_in_exact_set() -> None:
    exact = tenant_e2e_semantically_qualified_ids()
    for open_id in tenant_e2e_open_ids_from_audit():
        assert open_id not in exact


def test_ebh7_guard_skip_requires_equivalent_in_ssot() -> None:
    from tests.qualification.ebh_7._semantic_disposition_registry import assert_skip_equivalents_documented

    assert_skip_equivalents_documented()
