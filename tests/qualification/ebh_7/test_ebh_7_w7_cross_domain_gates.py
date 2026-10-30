# © Artur Czarnecki. All rights reserved.

"""EBH-7 W7 cross-domain scenario gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._cross_domain_scenarios import assert_ebh7_cross_domain_registry_integrity

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_w7_cross_domain_registry_34() -> None:
    assert_ebh7_cross_domain_registry_integrity()


def test_ebh7_w7_dup_x_rf_cmp_reuse_gate() -> None:
    import tests.qualification.dup_x.test_dup_x_mechanical_exit_gates as dup_exit

    assert hasattr(dup_exit, "test_dup_x_exit_mandatory_docs_present")
