# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-M / R4 (G-06): two PDC-touched product paths are classified in the CHANGED-SINCE impact map."""

from __future__ import annotations

import pytest
from intergrax.platform.enterprise_audit.changed_since import _check_touched
from intergrax.platform.enterprise_audit.impact_map import check_ids_for_changed_path, load_default_impact_map
from intergrax.platform.enterprise_audit.manifest import load_default_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EXPECTED = {
    "intergrax/experiments/workflow.py": frozenset({"ENT-A-REL-02", "ENT-A-REL-03"}),
    "intergrax/runtime/policy/pre_context_policy_audit.py": frozenset({"ENT-A-GOV-01", "ENT-A-GOV-02", "ENT-A-GOV-03"}),
}
_UNMAPPED_SIBLINGS = (
    "intergrax/experiments/_p8_unmapped_probe.py",
    "intergrax/runtime/policy/_p8_unmapped_probe.py",
)


def _unmapped(paths: tuple[str, ...]) -> list[str]:
    """The expression of the coverage tests: a path is unmapped when no prefix and no entrypoint matches."""
    prefixes = load_default_impact_map()
    checks = tuple(load_default_manifest())
    return [
        path
        for path in paths
        if not check_ids_for_changed_path(path, prefixes)
        and not any(_check_touched(check, frozenset({path})) for check in checks)
    ]


@pytest.mark.parametrize("path", sorted(_EXPECTED))
def test_the_two_paths_map_to_exactly_the_named_checks(path: str) -> None:
    ids = check_ids_for_changed_path(path, load_default_impact_map())
    assert ids == _EXPECTED[path]
    assert ids <= frozenset(check.check_id for check in load_default_manifest())


def test_the_entries_are_exact_files_so_unmapped_siblings_still_fail() -> None:
    assert _unmapped(tuple(_EXPECTED)) == []
    assert _unmapped(_UNMAPPED_SIBLINGS) == list(_UNMAPPED_SIBLINGS)


def test_the_manifest_still_has_exactly_85_checks() -> None:
    assert len(load_default_manifest()) == 85
