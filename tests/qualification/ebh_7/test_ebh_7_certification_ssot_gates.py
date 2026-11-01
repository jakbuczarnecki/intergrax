# © Artur Czarnecki. All rights reserved.

"""EBH-7 final closure — certification SSOT gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._ebh7_certification_ssot import (
    assert_ebh7_certification_ssot_bundle,
    assert_ebh7_rpl_certification_ssot,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_closure_rpl_certification_ssot() -> None:
    assert_ebh7_rpl_certification_ssot()


def test_ebh7_closure_ten_xdm_evidence_ssot() -> None:
    assert_ebh7_certification_ssot_bundle()
