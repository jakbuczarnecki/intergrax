# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-07: deployment/rollback qualified or unsafe refusal explicit (S7; scoped PASS, Part X)."""

import pytest
from testing_support.change_safety.assess.deployment_safety import load_na_register
from tests.qualification.evol_x._constants import FRZ_EVOL_07_SCOPE_FOOTNOTE
from tests.qualification.evol_x.experiments.arms import s7

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_frz_evol_07_s7_arms_pass() -> None:
    positive, negative = s7.run()
    assert positive.passed, positive.detail
    assert negative.passed, negative.detail


def test_frz_evol_07_scope_is_qualification_layer_only() -> None:
    assert "qualification-layer" in FRZ_EVOL_07_SCOPE_FOOTNOTE
    assert "NOT claimed" in FRZ_EVOL_07_SCOPE_FOOTNOTE
    assert {e["na_id"] for e in load_na_register()} == {"NA-01", "NA-02", "NA-03", "NA-04"}
