# © Artur Czarnecki. All rights reserved.

"""EXP-S3: positive and negative arm (record: evol_experiment_verdict.v1, 16 verdicts collected in W6)."""

import pytest
from tests.qualification.evol_x.experiments.arms import s3

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


@pytest.fixture(scope="module")
def arms():  # noqa: ANN201
    return s3.run()


def test_evol_x_exp_s3_positive_arm(arms) -> None:  # noqa: ANN001
    positive, _ = arms
    assert (positive.scenario_id, positive.arm) == ("S3", "positive")
    assert positive.passed, positive.detail


def test_evol_x_exp_s3_negative_arm(arms) -> None:  # noqa: ANN001
    _, negative = arms
    assert (negative.scenario_id, negative.arm) == ("S3", "negative")
    assert negative.passed, negative.detail
