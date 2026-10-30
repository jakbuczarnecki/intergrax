# © Artur Czarnecki. All rights reserved.

"""EBH-7 semantic proof-binding gates + B6 readiness (EXACT coverage)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._semantic_audit_markdown_sync import assert_semantic_audit_markdown_matches_ssot
from tests.qualification.ebh_7._semantic_proof_binding_audit import (
    assert_ebh7_semantic_mandatory_exact_coverage,
    assert_ebh7_semantic_proof_binding_audit,
    cross_domain_open_ids_from_audit,
    cross_domain_semantically_qualified_ids,
    semantic_exact_counts,
    tenant_e2e_open_ids_from_audit,
    tenant_e2e_semantically_qualified_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_semantic_proof_binding_audit_integrity() -> None:
    assert_ebh7_semantic_proof_binding_audit()


def test_ebh7_semantic_counts_from_registries() -> None:
    assert len(tenant_e2e_semantically_qualified_ids()) + len(tenant_e2e_open_ids_from_audit()) == 26
    assert len(cross_domain_semantically_qualified_ids()) + len(cross_domain_open_ids_from_audit()) == 34


def test_ebh7_semantic_audit_markdown_matches_ssot() -> None:
    assert_semantic_audit_markdown_matches_ssot()


@pytest.mark.xfail(reason="B6 blocked until 60/60 EXACT_PROOF", strict=False)
def test_ebh7_b6_readiness_mandatory_exact_coverage() -> None:
    assert_ebh7_semantic_mandatory_exact_coverage()


def test_ebh7_semantic_exact_blocker_counts_registered() -> None:
    ten_exact, ten_open, xdm_exact, xdm_open = semantic_exact_counts()
    assert ten_exact == 14 and ten_open == 12
    assert xdm_exact == 16 and xdm_open == 18
