# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-03: change isolation / blast radius explainable per scenario and application (S1, O-F)."""

import pytest
from testing_support.change_safety.assess.isolation import application_of_path, assess_isolation
from testing_support.change_safety.graph.impact import assess_path_change_impact
from tests.qualification.evol_x.experiments.arms import s1

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_frz_evol_03_s1_arms_pass() -> None:
    positive, negative = s1.run()
    assert positive.passed, positive.detail
    assert negative.passed, negative.detail


def test_frz_evol_03_every_escape_on_real_inventory_has_a_witness(evol_inventory) -> None:  # noqa: ANN001
    app_modules = sorted(p for p in evol_inventory.universe_paths if application_of_path(p) and p.endswith(".py"))
    assert app_modules
    explained = 0
    for path in app_modules[:25]:
        changed = frozenset({path})
        impact = assess_path_change_impact(evol_inventory, changed_paths=changed)
        report = assess_isolation(impact, changed_paths=changed)
        by_id = {n.node_id: n for n in impact.affected}
        assert all(len(by_id[e].witness_path) >= 2 for e in report.escapes)
        assert set(report.origin_apps) == {application_of_path(path)}
        explained += 1
    assert explained == min(25, len(app_modules))
