from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.changed_since import select_checks_for_changed_since
from intergrax.platform.enterprise_audit.impact_map import (
    WAVE_DIFF_FULL_IMPACT_PREFIXES,
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
    root = Path(__file__).resolve().parents[4]
    bogus = "0" * 40
    try:
        select_checks_for_changed_since(checks, repo_root=root, since_sha=bogus)
        raise AssertionError("expected ValueError for non-ancestor since_sha")
    except ValueError as exc:
        assert "ancestor" in str(exc).lower()
