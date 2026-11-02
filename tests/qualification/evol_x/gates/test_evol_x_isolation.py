# © Artur Czarnecki. All rights reserved.

from pathlib import Path

import pytest
from testing_support.change_safety.assess.isolation import application_of_node, application_of_path, assess_isolation
from testing_support.change_safety.graph.impact import assess_path_change_impact
from tests.qualification.evol_x.adapters.fixture_tree import inventory_for, write_tree

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_TREE = {
    "applications/app_a/y.py": "VALUE = 1\n",
    "applications/app_a/x.py": "from applications.app_a.y import VALUE\n",
    "applications/app_b/solo.py": "OTHER = 2\n",
    "applications/app_c/leak.py": "from applications.app_a.y import VALUE\n",
    "applications/shared/lib.py": "LIB = 3\n",
    "applications/app_b/uses_shared.py": "from applications.shared.lib import LIB\n",
}


def _report(tmp_path: Path, changed: str, **kwargs):
    inv = inventory_for(write_tree(tmp_path, _TREE))
    impact = assess_path_change_impact(inv, changed_paths=frozenset({changed}))
    return assess_isolation(impact, changed_paths=frozenset({changed}), **kwargs)


def test_node_and_path_application_parsers() -> None:
    assert application_of_node("module:applications.app_a.y") == "app_a"
    assert application_of_node("module:intergrax.x") is None
    assert application_of_path("applications/app_a/y.py") == "app_a"
    assert application_of_path("agents/x.py") is None


def test_local_change_blast_stays_in_app_or_flags_escape(tmp_path: Path) -> None:
    rep = _report(tmp_path, "applications/app_b/solo.py")
    assert rep.isolated
    assert rep.blast_apps == ("app_b",)


def test_cross_app_consumer_is_flagged_as_escape(tmp_path: Path) -> None:
    rep = _report(tmp_path, "applications/app_a/y.py")
    assert not rep.isolated
    assert "module:applications.app_c.leak" in rep.escapes
    assert set(rep.blast_apps) == {"app_a", "app_c"}


def test_declared_shared_surface_is_not_an_escape(tmp_path: Path) -> None:
    undeclared = _report(tmp_path, "applications/shared/lib.py")
    assert not undeclared.isolated
    declared = _report(tmp_path, "applications/shared/lib.py", declared_shared=frozenset({"applications.shared"}))
    assert declared.isolated
    assert "app_b" in declared.blast_apps
