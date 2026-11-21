from __future__ import annotations

import re
import subprocess
from pathlib import Path

from intergrax.platform.enterprise_audit.changed_since import (
    _check_touched,
    git_changed_files,
    invariant_floor,
    select_checks_for_changed_since,
    unmapped_evol_changed_paths,
)
from intergrax.platform.enterprise_audit.impact_map import (
    WAVE_DIFF_FULL_IMPACT_PREFIXES,
    check_ids_for_changed_path,
    load_default_impact_map,
    unmapped_fail_closed_paths,
)
from intergrax.platform.enterprise_audit.manifest import load_default_manifest

REPO_ROOT = Path(__file__).resolve().parents[4]
_PDC_LADDER = REPO_ROOT / "docs/project/maintainers/qualification/platform_domain_completeness/PDC_SHA_LADDER.md"


def _pdc_baseline_sha() -> str:
    match = re.search(r"\|\s*BASELINE_SHA\s*\|\s*`([0-9a-f]{40})`", _PDC_LADDER.read_text(encoding="utf-8"))
    assert match is not None, "BASELINE_SHA row not found in PDC_SHA_LADDER.md"
    sha = match.group(1)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert ancestor.returncode == 0, f"PDC BASELINE_SHA {sha} is not an ancestor of HEAD"
    return sha


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


def test_development_wave_diff_residual_modules_map_to_manifest_ids() -> None:
    """NT-043M-S3: slice-2 wave-diff prefixes cover KNOWN_RED / train-parity residual groups."""
    prefixes = load_default_impact_map()
    manifest_ids = frozenset(check.check_id for check in load_default_manifest())
    samples = (
        "intergrax/background_tasks/worker_runtime.py",
        "intergrax/runtime/sandbox/runtime.py",
        "intergrax/runtime/wiring/llm_resolver.py",
        "agents/local_search/steps.py",
    )
    for path in samples:
        ids = check_ids_for_changed_path(path, prefixes)
        assert ids, path
        assert ids <= manifest_ids
        assert len(ids) == len(manifest_ids), path


def test_development_wave_diff_touched_modules_map_to_manifest_ids() -> None:
    """NT-043M-S2: development-tor wave-diff prefixes cover remediation-wave touched modules."""
    prefixes = load_default_impact_map()
    manifest_ids = frozenset(check.check_id for check in load_default_manifest())
    samples = (
        "intergrax/runtime/interactions/intake_service.py",
        "intergrax/runtime/notifications/factory.py",
        "intergrax/model_inference/contracts.py",
    )
    for path in samples:
        ids = check_ids_for_changed_path(path, prefixes)
        assert ids, path
        assert ids <= manifest_ids
    for prefix in WAVE_DIFF_FULL_IMPACT_PREFIXES:
        sample = f"{prefix}/__wave_diff_probe__.py"
        ids = check_ids_for_changed_path(sample, prefixes)
        assert len(ids) == len(manifest_ids), prefix


def test_select_changed_since_requires_ancestor_sha() -> None:
    checks = load_default_manifest()
    bogus = "0" * 40
    try:
        select_checks_for_changed_since(checks, repo_root=REPO_ROOT, since_sha=bogus)
        raise AssertionError("expected ValueError for non-ancestor since_sha")
    except ValueError as exc:
        assert "ancestor" in str(exc).lower()


def test_development_tor_pdc_changed_since_is_fail_closed_clean() -> None:
    """NT-043M-S4: slice-3 governance closure — CHANGED-SINCE over the PDC range at development head."""
    checks = load_default_manifest()
    baseline = _pdc_baseline_sha()
    changed = git_changed_files(REPO_ROOT, baseline)
    selected = select_checks_for_changed_since(checks, repo_root=REPO_ROOT, since_sha=baseline)
    manifest_ids = {check.check_id for check in checks}
    floor = invariant_floor(checks)
    assert floor <= selected <= manifest_ids
    assert not unmapped_evol_changed_paths(checks, changed)
    prefixes = load_default_impact_map()
    covered = frozenset(p for p in changed if any(_check_touched(c, frozenset({p})) for c in checks))
    assert unmapped_fail_closed_paths(changed, prefixes=prefixes, entrypoint_covered=covered) == frozenset()
