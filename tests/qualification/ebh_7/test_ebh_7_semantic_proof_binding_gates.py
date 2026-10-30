# © Artur Czarnecki. All rights reserved.

"""Phase A semantic proof-binding audit gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._semantic_proof_binding_audit import (
    assert_ebh7_semantic_proof_binding_audit,
    cross_domain_open_ids_from_audit,
    cross_domain_semantically_qualified_ids,
    tenant_e2e_open_ids_from_audit,
    tenant_e2e_semantically_qualified_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_semantic_proof_binding_audit_integrity() -> None:
    assert_ebh7_semantic_proof_binding_audit()


def test_ebh7_semantic_counts_from_registries() -> None:
    assert len(tenant_e2e_semantically_qualified_ids()) + len(tenant_e2e_open_ids_from_audit()) == 26
    assert len(cross_domain_semantically_qualified_ids()) + len(cross_domain_open_ids_from_audit()) == 34
