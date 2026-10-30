# © Artur Czarnecki. All rights reserved.

"""EBH-7 W6 tenant E2E — structural gates only."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._tenant_e2e_scenarios import assert_ebh7_tenant_e2e_registry_integrity
from tests.qualification.ebh_7._tenant_unknown_role_dispositions import (
    assert_ebh7_gap_ten_unk_05_not_e2e_substitute,
    assert_ebh7_gap_ten_unk_05_structural,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_w6_tenant_e2e_registry_structural() -> None:
    assert_ebh7_tenant_e2e_registry_integrity()


def test_ebh7_w6_gap_ten_unk_05_structural_disposition() -> None:
    assert_ebh7_gap_ten_unk_05_structural()
    assert_ebh7_gap_ten_unk_05_not_e2e_substitute()
