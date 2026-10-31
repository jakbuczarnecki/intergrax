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


def test_ebh7_mutation_tampered_match_not_exact() -> None:
    assert ProofBindingResult.PARTIAL_PROOF not in (
        ProofBindingResult.EXACT_PROOF,
        ProofBindingResult.MISSING_PROOF,
    )
