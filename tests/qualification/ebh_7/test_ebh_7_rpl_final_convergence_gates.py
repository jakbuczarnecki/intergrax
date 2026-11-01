# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._rpl_h_final_convergence_disposition import (
    assert_ebh7_rpl_convergence_disposition_complete,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_rpl_final_convergence_disposition_covers_open_concerns() -> None:
    assert_ebh7_rpl_convergence_disposition_complete()
