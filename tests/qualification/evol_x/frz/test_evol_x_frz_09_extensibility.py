# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-09: extensibility without unauthorized platform-core modification (S8, EVOL-M-10)."""

import pytest
from testing_support.change_safety.assess.extension_safety import ExtensionVerdict, assess_extension_safety
from tests.qualification.evol_x.experiments.arms import s4, s8

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_frz_evol_09_s8_and_s4_arms_pass() -> None:
    for arm in (*s8.run(), *s4.run()):
        assert arm.passed, f"{arm.scenario_id}/{arm.arm}: {arm.detail}"


def test_frz_evol_09_core_edit_without_authorization_is_never_allowed() -> None:
    report = assess_extension_safety(changed_paths=frozenset({"intergrax/runtime/governance/policy.py"}))
    assert report.verdict is ExtensionVerdict.CORE_TOUCH_UNAUTHORIZED and report.effective_class == "C"
