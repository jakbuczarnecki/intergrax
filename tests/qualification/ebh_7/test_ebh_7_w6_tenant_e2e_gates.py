# © Artur Czarnecki. All rights reserved.

"""EBH-7 W6 tenant platform E2E + GAP-TEN-UNK-05 gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._tenant_e2e_scenarios import assert_ebh7_tenant_e2e_registry_integrity
from tests.qualification.ebh_7._tenant_unknown_role_dispositions import assert_ebh7_gap_ten_unk_05_closed

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_w6_tenant_e2e_registry_26() -> None:
    assert_ebh7_tenant_e2e_registry_integrity()


def test_ebh7_w6_gap_ten_unk_05_closed() -> None:
    assert_ebh7_gap_ten_unk_05_closed()


def test_ebh7_w6_tenant_x_final_head_ready() -> None:
    from tests.qualification.tenant_x.test_tenant_x_final_current_head import (
        test_tx_final_head_q01_final_ready_for_audit,
    )

    test_tx_final_head_q01_final_ready_for_audit()
