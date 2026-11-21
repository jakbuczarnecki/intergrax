# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-I Part B: evol_x test-support tree mapped (EQOA-A6-003, X-A6-003). Tests B01..B04 of the plan."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit import changed_since as cs
from intergrax.platform.enterprise_audit.impact_map import build_impact_map_from_manifest, load_default_impact_map
from intergrax.platform.enterprise_audit.manifest import load_default_manifest

_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_default_manifest()
_EVOL_IDS = frozenset({"ENT-A-EVOL-01", "ENT-A-EVOL-02", "ENT-A-EVOL-03", "ENT-A-INV-01", "ENT-A-REG-02"})
_DOSSIER_PATHS = (
    "tests/qualification/evol_x/_constants.py",
    "tests/qualification/evol_x/conftest.py",
    *(f"tests/qualification/evol_x/adapters/{name}" for name in ("a.py", "b.py", "c.py", "d.py", "e.py", "f.py")),
)


def _tracked_evol_x_files() -> frozenset[str]:
    out = subprocess.run(
        ["git", "ls-files", "tests/qualification/evol_x"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return frozenset(line for line in out.splitlines() if line)


def test_wd_b01_every_tracked_evol_x_file_is_mapped() -> None:
    files = _tracked_evol_x_files()
    assert len(files) > 50
    unmapped = cs.unmapped_evol_changed_paths(_MANIFEST, files, prefixes=load_default_impact_map())
    assert unmapped == frozenset()


def test_wd_b02_the_eight_dossier_paths_select_the_evol_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "git_changed_files", lambda _root, _sha: frozenset(_DOSSIER_PATHS))
    monkeypatch.setattr(cs, "assert_git_ancestor", lambda *_a, **_k: None)
    selected = cs.select_checks_for_changed_since(_MANIFEST, repo_root=_ROOT, since_sha="0" * 40)
    assert _EVOL_IDS <= selected


def test_wd_b03_unknown_new_evol_x_directory_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cs, "git_changed_files", lambda _root, _sha: frozenset({"tests/qualification/evol_x/zz_new/x.py"})
    )
    monkeypatch.setattr(cs, "assert_git_ancestor", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="unmapped evol_x paths"):
        cs.select_checks_for_changed_since(_MANIFEST, repo_root=_ROOT, since_sha="0" * 40)


def test_wd_b04_shipped_map_still_equals_generator() -> None:
    generated = build_impact_map_from_manifest(_MANIFEST)
    assert load_default_impact_map() == generated
    payload = {
        "schema_version": 1,
        "path_prefix_to_check_ids": {k: sorted(v) for k, v in sorted(generated.items())},
    }
    expected = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    shipped = (_ROOT / "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json").read_bytes()
    assert shipped == expected
