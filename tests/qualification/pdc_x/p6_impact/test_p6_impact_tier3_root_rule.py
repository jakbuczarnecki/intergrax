# © Artur Czarnecki. All rights reserved.

"""T-IGX-040-A (B-P6-J, F-41): the CHANGED-SINCE impact map covers the Tier-3 root ``applications/``.

The root ``applications/`` (Tier-3) is distinct from ``intergrax/applications`` (Tier-0 shared code). Its
rule lives in the second typed table ``PDC_TIER3_ROOT_IMPACT_RULES`` of ``impact_map.py`` so that the
``intergrax/``-only prefix pin of ``test_p6_impact_map_coverage.py`` and the xd_12 test stay unchanged.
The rule may only name ids that exist in the shipped manifest (still exactly 85 checks), and it must
select more checks for ``applications/`` paths without dropping or rerouting a check for any other path.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit import impact_map
from intergrax.platform.enterprise_audit.changed_since import _check_touched, git_changed_files
from intergrax.platform.enterprise_audit.impact_map import (
    build_impact_map_from_manifest,
    check_ids_for_changed_path,
    fail_closed_prefixes,
    load_default_impact_map,
)
from intergrax.platform.enterprise_audit.manifest import load_default_manifest

REPO_ROOT = Path(__file__).resolve().parents[4]
LADDER = REPO_ROOT / "docs/project/maintainers/qualification/platform_domain_completeness/PDC_SHA_LADDER.md"
SSOT = REPO_ROOT / "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json"
_PRODUCT_ROOTS = ("intergrax/", "agents/", "applications/")
_ROOT_PREFIX = "applications"
_MANIFEST_SIZE = 85
_EXPECTED_IDS = frozenset(
    {
        "ENT-A-BND-01",
        "ENT-A-CTR-01",
        "ENT-A-CTR-02",
        "ENT-A-CTR-03",
        "ENT-A-TYP-01",
        "ENT-A-SEC-01",
        "ENT-A-SEC-02",
        "VBN-CONTRACT-LEAKAGE",
    }
)
_MCP_APPLICATIONS = (
    "dispute_sim_application",
    "governed_contractor_application",
    "intergrax_assistant_application",
    "lab_application",
    "legal_application",
    "local_workspace_application",
    "poc_template_application",
    "research_application",
)
_MCP_SERVER_MODULES = tuple(f"applications/{app}/mcp/server.py" for app in _MCP_APPLICATIONS)


def _rules() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return getattr(impact_map, "PDC_TIER3_ROOT_IMPACT_RULES", ())


def _manifest_ids() -> frozenset[str]:
    return frozenset(check.check_id for check in load_default_manifest())


def _tracked(pathspec: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", pathspec], cwd=REPO_ROOT, capture_output=True, check=True, text=True
    )
    return sorted(p for p in out.stdout.split("\0") if p)


def _pdc_touched_product_modules() -> list[str]:
    match = re.search(r"\|\s*BASELINE_SHA\s*\|\s*`([0-9a-f]{40})`", LADDER.read_text(encoding="utf-8"))
    assert match is not None, "BASELINE_SHA row not found in PDC_SHA_LADDER.md"
    sha = match.group(1)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert ancestor.returncode == 0, f"PDC BASELINE_SHA {sha} is not an ancestor of HEAD (shallow clone?)"
    changed = git_changed_files(REPO_ROOT, sha)
    return sorted(p for p in changed if p.endswith(".py") and p.startswith(_PRODUCT_ROOTS))


def test_manifest_still_has_exactly_85_checks() -> None:
    assert len(load_default_manifest()) == _MANIFEST_SIZE


def test_tier3_root_rule_table_is_a_single_normalized_root_prefix_with_existing_ids() -> None:
    rules = _rules()
    assert rules, "PDC_TIER3_ROOT_IMPACT_RULES must not be empty"
    assert [prefix for prefix, _ in rules] == [_ROOT_PREFIX]
    manifest_ids = _manifest_ids()
    for prefix, ids in rules:
        assert not prefix.startswith("intergrax/"), prefix
        assert not prefix.endswith("/") and "\\" not in prefix, prefix
        assert not any(prefix.startswith(fc.rstrip("/")) for fc in fail_closed_prefixes()), prefix
        assert len(set(ids)) == len(ids), f"{prefix}: duplicate check id"
        unknown = sorted(set(ids) - manifest_ids)
        assert not unknown, f"{prefix}: ids missing from the manifest: {unknown}"


def test_tier3_root_rule_names_the_eight_ids_of_the_intergrax_applications_rule() -> None:
    rules = dict(_rules())
    assert frozenset(rules.get(_ROOT_PREFIX, ())) == _EXPECTED_IDS
    tier0 = dict(impact_map.PDC_DOMAIN_IMPACT_RULES)["intergrax/applications"]
    assert frozenset(tier0) == _EXPECTED_IDS


def test_generator_output_contains_the_tier3_root_rule() -> None:
    mapping = build_impact_map_from_manifest(load_default_manifest())
    assert _ROOT_PREFIX in mapping
    assert _EXPECTED_IDS <= mapping[_ROOT_PREFIX]


def test_shipped_ssot_contains_the_tier3_root_key() -> None:
    shipped = json.loads(SSOT.read_text(encoding="utf-8"))["path_prefix_to_check_ids"]
    assert _ROOT_PREFIX in shipped
    assert _EXPECTED_IDS <= set(shipped[_ROOT_PREFIX])


@pytest.mark.parametrize("module", _MCP_SERVER_MODULES)
def test_named_mcp_server_module_maps_to_a_manifest_check(module: str) -> None:
    ids = check_ids_for_changed_path(module, load_default_impact_map())
    assert ids, f"{module} maps to no check id"
    assert ids <= _manifest_ids()


def test_every_tracked_applications_python_file_maps_to_at_least_one_check() -> None:
    modules = _tracked("applications/**/*.py")
    assert modules, "no tracked applications/**/*.py file; the test would be vacuous"
    prefixes = load_default_impact_map()
    unmapped = [path for path in modules if not check_ids_for_changed_path(path, prefixes)]
    assert unmapped == [], f"{len(unmapped)} tracked applications/ modules map to no check, e.g. {unmapped[:5]}"


def test_root_rule_does_not_shadow_or_reroute_any_other_path() -> None:
    shipped = load_default_impact_map()
    without_root = {prefix: ids for prefix, ids in shipped.items() if prefix != _ROOT_PREFIX}
    tracked = _tracked("*.py") + _tracked("*.json") + _tracked("*.md") + _tracked("*.yaml")
    assert tracked
    for path in tracked:
        new_ids = check_ids_for_changed_path(path, shipped)
        old_ids = check_ids_for_changed_path(path, without_root)
        if path.startswith("applications/"):
            assert old_ids <= new_ids, f"{path}: a check was dropped: {sorted(old_ids - new_ids)}"
        else:
            assert new_ids == old_ids, f"{path}: selection changed outside applications/"


def test_synthetic_mcp_rc_range_leaves_no_pdc_touched_product_module_unmapped() -> None:
    """The two target tests over the PDC range plus the eight MCP-RC ``mcp/server.py`` paths."""
    modules = sorted({*_pdc_touched_product_modules(), *_MCP_SERVER_MODULES})
    prefixes = load_default_impact_map()
    checks = tuple(load_default_manifest())
    unmapped = [
        path
        for path in modules
        if not check_ids_for_changed_path(path, prefixes)
        and not any(_check_touched(check, frozenset({path})) for check in checks)
    ]
    assert unmapped == [], f"{len(unmapped)} PDC-touched product modules map to no check, e.g. {unmapped[:5]}"
