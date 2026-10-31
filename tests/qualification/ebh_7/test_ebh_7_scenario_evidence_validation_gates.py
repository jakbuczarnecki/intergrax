# © Artur Czarnecki. All rights reserved.

"""TEN/XDM behavioral evidence validation gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._ebh7_scenario_evidence_validation import (
    assert_ebh7_ten_xdm_evidence_validation_complete,
    ten_xdm_evidence_summary,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_ten_xdm_proof_nodes_resolve_and_disposition_aligned() -> None:
    assert_ebh7_ten_xdm_evidence_validation_complete()


def test_ebh7_ten_xdm_evidence_summary_reported() -> None:
    summary = ten_xdm_evidence_summary()
    assert sum(summary["TEN"].values()) == 26
    assert sum(summary["XDM"].values()) == 34
