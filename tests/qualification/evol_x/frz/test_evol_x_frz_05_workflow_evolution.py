# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-05: workflow definition evolution rules explicit for in-flight instances (WM + S3)."""

import pytest
from testing_support.change_safety.assess.workflow_evolution import matrix_row_coverage, unsafe_inflight_adoptions
from tests.qualification.evol_x.experiments.arms import s3

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_frz_evol_05_s3_dual_arm_passes() -> None:
    positive, negative = s3.run()
    assert positive.passed, positive.detail
    assert negative.passed, negative.detail


def test_frz_evol_05_matrix_total_and_no_unsafe_adoption() -> None:
    assert unsafe_inflight_adoptions() == ()
    assert sum(matrix_row_coverage().values()) == 128
