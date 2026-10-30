# © Artur Czarnecki. All rights reserved.

"""EBH-7 W7 cross-domain — structural registry gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._cross_domain_scenarios import assert_ebh7_cross_domain_registry_integrity

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_w7_cross_domain_registry_structural() -> None:
    assert_ebh7_cross_domain_registry_integrity()
