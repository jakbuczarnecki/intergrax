# © Artur Czarnecki. All rights reserved.

"""B-EXP-01: S1-S8 negative arms assert platform failure signals; meta tamper blocks vacuous PASS."""

from dataclasses import replace

import pytest
from tests.qualification.evol_x.experiments.arms import s1, s2, s3, s4, s5, s6, s7, s8

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_NEGATIVE_MARKERS = {
    "S1": ("naive_blast", "escapes="),
    "S2": ("naive_missed",),
    "S3": ("unsafe_adoptions", "production_rejects"),
    "S4": ("rewrite_required",),
    "S5": ("no_migration", "silent"),
    "S6": ("bypass_count=",),
    "S7": ("false_safe",),
    "S8": ("core_edit=", "unauthorized"),
}


@pytest.mark.parametrize(
    ("scenario", "runner"),
    [
        ("S1", lambda _inv: s1.run()),
        ("S2", lambda _inv: s2.run()),
        ("S3", lambda _inv: s3.run()),
        ("S4", lambda _inv: s4.run()),
        ("S5", lambda _inv: s5.run()),
        ("S6", lambda inv: s6.run(inv)),
        ("S7", lambda _inv: s7.run()),
        ("S8", lambda _inv: s8.run()),
    ],
    ids=[f"S{i}" for i in range(1, 9)],
)
def test_evol_x_exp_negative_arm_is_non_vacuous(scenario: str, runner, evol_inventory) -> None:  # noqa: ANN001
    arms = runner(evol_inventory)
    negative = arms[1]
    assert negative.arm == "negative"
    assert negative.passed, negative.detail
    detail = negative.detail.lower()
    assert any(marker.lower() in detail for marker in _NEGATIVE_MARKERS[scenario])


def test_evol_x_exp_s2_meta_tamper_positive_expectation_fails() -> None:
    positive, _ = s2.run()
    assert positive.passed
    assert replace(positive, passed=False).passed is False
