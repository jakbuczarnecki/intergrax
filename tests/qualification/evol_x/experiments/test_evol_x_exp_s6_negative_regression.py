# © Artur Czarnecki. All rights reserved.

"""EXP-S6: positive and negative arm (record: evol_experiment_verdict.v1, 16 verdicts collected in W6)."""

import pytest
from tests.qualification.evol_x.experiments.arms import s6

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


@pytest.fixture(scope="module")
def arms(evol_inventory):  # noqa: ANN201
    return s6.run(evol_inventory)


def test_evol_x_exp_s6_positive_arm(arms) -> None:  # noqa: ANN001
    positive, _ = arms
    assert (positive.scenario_id, positive.arm) == ("S6", "positive")
    assert positive.passed, positive.detail


def test_evol_x_exp_s6_negative_arm(arms) -> None:  # noqa: ANN001
    _, negative = arms
    assert (negative.scenario_id, negative.arm) == ("S6", "negative")
    assert negative.passed, negative.detail
