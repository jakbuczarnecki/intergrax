# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-04: contract/schema evolution detects consumers; COMPAT policy applied (S2, S5, O-D)."""

import pytest
from testing_support.change_safety.assess.consumers import critical_consumer_closure
from tests.qualification.evol_x.experiments.arms import s2, s5

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_frz_evol_04_s2_and_s5_arms_pass() -> None:
    for arm in (*s2.run(), *s5.run()):
        assert arm.passed, f"{arm.scenario_id}/{arm.arm}: {arm.detail}"


def test_frz_evol_04_no_orphan_consumers_on_real_inventory(evol_inventory) -> None:  # noqa: ANN001
    closure = critical_consumer_closure(evol_inventory)
    assert closure.orphan_count == 0, closure.omissions[:5]
