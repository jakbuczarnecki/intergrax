from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.changed_since import select_checks_for_changed_since
from intergrax.platform.enterprise_audit.impact_map import (
    check_ids_for_changed_path,
    load_default_impact_map,
    unmapped_fail_closed_paths,
)
from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_impact_map_selects_invariant_owner() -> None:
    prefixes = load_default_impact_map()
    ids = check_ids_for_changed_path("intergrax/platform/enterprise_audit/changed_since.py", prefixes)
    assert "C024" in ids
    assert "C029" in ids


def test_unknown_platform_path_fail_closed() -> None:
    prefixes = load_default_impact_map()
    fake_path = "intergrax/platform/enterprise_audit/__totally_unmapped_new_module.py"
    fake = frozenset({fake_path})
    unmapped = unmapped_fail_closed_paths(fake, prefixes=prefixes, entrypoint_covered=frozenset())
    assert fake_path in unmapped


def test_select_changed_since_requires_ancestor_sha() -> None:
    checks = load_default_manifest()
    root = Path(__file__).resolve().parents[4]
    bogus = "0" * 40
    try:
        select_checks_for_changed_since(checks, repo_root=root, since_sha=bogus)
        raise AssertionError("expected ValueError for non-ancestor since_sha")
    except ValueError as exc:
        assert "ancestor" in str(exc).lower()
