# © Artur Czarnecki. All rights reserved.

"""EBH-7 W1 mechanism registry gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._mechanism_registry import assert_ebh7_mechanism_registry_integrity

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_w1_mechanism_registry_685() -> None:
    assert_ebh7_mechanism_registry_integrity()
