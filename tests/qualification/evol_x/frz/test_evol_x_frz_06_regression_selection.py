# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-06: regression selection from impact graph with mandatory suites; unknown -> conservative (S6)."""

import pytest
from testing_support.change_safety.selection.policy import POLICY_ID, selection_policy_forbid_empty
from tests.qualification.evol_x.experiments.arms import s6

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_frz_evol_06_s6_arms_pass(evol_inventory) -> None:  # noqa: ANN001
    positive, negative = s6.run(evol_inventory)
    assert positive.passed, positive.detail
    assert negative.passed, negative.detail


def test_frz_evol_06_policy_is_scoped_and_forbids_empty_selection() -> None:
    assert selection_policy_forbid_empty()
    assert POLICY_ID == "EVOL_CONSERVATIVE_FULL"  # EVOL-scoped policy; no PLATFORM_CONSERVATIVE_FULL claim
