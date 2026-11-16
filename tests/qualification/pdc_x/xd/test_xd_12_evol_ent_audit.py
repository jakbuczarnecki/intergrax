# © Artur Czarnecki. All rights reserved.

"""XD-12 (All <-> EVOL/ENT-AUDIT): ENT-AUDIT CHANGED-SINCE maps the PDC-touched paths to checks.

The set of PDC-touched paths is real: ``git diff --name-only BASELINE_SHA..HEAD`` where BASELINE_SHA is
the PDC-P0 start recorded in ``PDC_SHA_LADDER.md``. The selection is the real
``select_checks_for_changed_since`` over the real default manifest and the shipped impact map
(``changed_since_impact_map.v1.json``); nothing is stubbed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.changed_since import (
    _check_touched,
    git_changed_files,
    invariant_floor,
    select_checks_for_changed_since,
    unmapped_evol_changed_paths,
)
from intergrax.platform.enterprise_audit.impact_map import (
    check_ids_for_changed_path,
    load_default_impact_map,
    unmapped_fail_closed_paths,
)
from intergrax.platform.enterprise_audit.manifest import load_default_manifest

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[4]
LADDER = REPO_ROOT / "docs/project/maintainers/qualification/platform_domain_completeness/PDC_SHA_LADDER.md"
_PRODUCT_ROOTS = ("intergrax/", "agents/", "applications/")


def _baseline_sha() -> str:
    match = re.search(r"\|\s*BASELINE_SHA\s*\|\s*`([0-9a-f]{40})`", LADDER.read_text(encoding="utf-8"))
    assert match is not None, "BASELINE_SHA row not found in PDC_SHA_LADDER.md"
    sha = match.group(1)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert ancestor.returncode == 0, f"PDC BASELINE_SHA {sha} is not an ancestor of HEAD (shallow clone?)"
    return sha


@pytest.fixture(scope="module")
def pdc_changed() -> frozenset[str]:
    return git_changed_files(REPO_ROOT, _baseline_sha())


@pytest.fixture(scope="module")
def checks():  # type: ignore[no-untyped-def]
    return tuple(load_default_manifest())


def _product_modules(changed: frozenset[str]) -> list[str]:
    return sorted(p for p in changed if p.endswith(".py") and p.startswith(_PRODUCT_ROOTS))


def test_xd12_the_pdc_campaign_touched_product_paths(pdc_changed: frozenset[str]) -> None:
    product = _product_modules(pdc_changed)
    assert product, "the PDC campaign changed no product module; the scenario would be vacuous"
    assert any(p.startswith("tests/qualification/pdc_x/") for p in pdc_changed)


def test_xd12_changed_since_over_the_pdc_range_never_fails_closed_and_keeps_the_invariant_floor(
    checks,
    pdc_changed: frozenset[str],  # type: ignore[no-untyped-def]
) -> None:
    selected = select_checks_for_changed_since(checks, repo_root=REPO_ROOT, since_sha=_baseline_sha())
    manifest_ids = {check.check_id for check in checks}
    floor = invariant_floor(checks)
    assert floor, "the invariant floor must not be empty"
    assert floor <= selected, "every CHANGED-SINCE run keeps the invariant floor"
    assert selected <= manifest_ids, "selection only names checks that exist in the manifest"
    assert not unmapped_evol_changed_paths(checks, pdc_changed), "no evol_x path is left unmapped"
    prefixes = load_default_impact_map()
    covered = frozenset(p for p in pdc_changed if any(_check_touched(c, frozenset({p})) for c in checks))
    assert unmapped_fail_closed_paths(pdc_changed, prefixes=prefixes, entrypoint_covered=covered) == frozenset()


def test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check(
    checks,
    pdc_changed: frozenset[str],  # type: ignore[no-untyped-def]
) -> None:
    prefixes = load_default_impact_map()
    unmapped = [
        path
        for path in _product_modules(pdc_changed)
        if not check_ids_for_changed_path(path, prefixes)
        and not any(_check_touched(c, frozenset({path})) for c in checks)
    ]
    assert unmapped == [], f"{len(unmapped)} PDC-touched product modules map to no check, e.g. {unmapped[:5]}"
