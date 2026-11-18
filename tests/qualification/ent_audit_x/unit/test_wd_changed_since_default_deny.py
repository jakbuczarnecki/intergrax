# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-I Part A: CHANGED-SINCE default-deny (EQOA-A6-007, DP-66). Tests A01..A16 of the plan."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit import changed_since as cs
from intergrax.platform.enterprise_audit import impact_map as im
from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.impact_map import (
    check_ids_for_changed_path,
    load_default_impact_map,
    load_impact_map_from_path,
    selected_check_ids_from_impact,
)
from intergrax.platform.enterprise_audit.manifest import checks_for_mode, load_default_manifest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_default_manifest()
_MANIFEST_IDS = frozenset(c.check_id for c in _MANIFEST)
_FULL_SET = frozenset(c.check_id for c in checks_for_mode(_MANIFEST, AuditMode.FULL))
_FLOOR = cs.invariant_floor(_MANIFEST)


def _select(monkeypatch: pytest.MonkeyPatch, *paths: str) -> frozenset[str]:
    """Public selector (exists before and after the change): used by the GUARD tests."""
    monkeypatch.setattr(cs, "git_changed_files", lambda _root, _sha: frozenset(paths))
    monkeypatch.setattr(cs, "assert_git_ancestor", lambda *_a, **_k: None)
    return cs.select_checks_for_changed_since(_MANIFEST, repo_root=_ROOT, since_sha="0" * 40)


def _plan(monkeypatch: pytest.MonkeyPatch, *paths: str) -> cs.ChangedSinceSelection:
    monkeypatch.setattr(cs, "git_changed_files", lambda _root, _sha: frozenset(paths))
    monkeypatch.setattr(cs, "assert_git_ancestor", lambda *_a, **_k: None)
    return cs.plan_changed_since_selection(_MANIFEST, repo_root=_ROOT, since_sha="0" * 40)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=wd", "-c", "user.email=wd@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _scratch_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "scratch"
    repo.mkdir()
    _git(repo, "init", "-q")
    for rel, text in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("prefix", ("intergrax/", "agents/"))
def test_wd_a01_new_product_package_selects_full(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    plan = _plan(monkeypatch, f"{prefix}zz_wd_new_pkg/mod.py")
    assert plan.escalated_to_full
    assert plan.check_ids == _FULL_SET
    assert plan.unmapped_paths == (f"{prefix}zz_wd_new_pkg/mod.py",)


def test_wd_a01b_root_applications_tree_is_mapped_by_its_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """``applications/`` carries one tree-wide rule (B-P6-J), so a new package there is covered, not unmapped."""
    path = "applications/zz_wd_new_pkg/mod.py"
    selected = _select(monkeypatch, path)
    mapped = check_ids_for_changed_path(path, load_default_impact_map()) & _MANIFEST_IDS
    assert mapped
    assert selected == _FLOOR | mapped
    assert selected < _FULL_SET


def test_wd_a02_mapped_product_path_stays_narrow(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "intergrax/rag/retrieval/retrieval_service.py"
    selected = _select(monkeypatch, path)
    mapped = check_ids_for_changed_path(path, load_default_impact_map()) & _MANIFEST_IDS
    assert mapped
    assert selected == _FLOOR | mapped | {c.check_id for c in _MANIFEST if cs._check_touched(c, frozenset({path}))}
    assert selected < _FULL_SET
    assert selected <= _MANIFEST_IDS


def test_wd_a03_engine_file_with_only_legacy_ids_selects_full(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = (
        "intergrax/platform/enterprise_audit/runner.py",
        "scripts/enterprise_audit/build_impact_map.py",
    )
    for path in paths:
        plan = _plan(monkeypatch, path)
        assert plan.escalated_to_full, path
        assert plan.check_ids == _FULL_SET, path
        assert plan.check_ids <= _MANIFEST_IDS, path


def test_wd_a04_unknown_engine_tree_file_selects_full(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(monkeypatch, "intergrax/platform/enterprise_audit/zz_new.py")
    assert plan.escalated_to_full
    assert plan.check_ids == _FULL_SET


def test_wd_a05_backslash_path_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(monkeypatch, "intergrax\\zz_wd_new_pkg\\mod.py")
    assert plan.escalated_to_full
    assert plan.check_ids == _FULL_SET
    assert plan.unmapped_paths == ("intergrax/zz_wd_new_pkg/mod.py",)


def test_wd_a06_rename_reports_old_and_new_path(tmp_path: Path) -> None:
    repo, base = _scratch_repo(tmp_path, {"intergrax/zz_old/a.py": "x = 1\n"})
    (repo / "docs").mkdir()
    _git(repo, "mv", "intergrax/zz_old/a.py", "docs/moved.py")
    _git(repo, "commit", "-q", "-m", "move")
    changed = cs.git_changed_files(repo, base)
    assert {"intergrax/zz_old/a.py", "docs/moved.py"} <= changed


def test_wd_a07_deleted_unmapped_product_path_escalates(tmp_path: Path) -> None:
    repo, base = _scratch_repo(tmp_path, {"intergrax/zz_old/a.py": "x = 1\n", "docs/keep.md": "k\n"})
    _git(repo, "rm", "-q", "intergrax/zz_old/a.py")
    _git(repo, "commit", "-q", "-m", "delete")
    plan = cs.plan_changed_since_selection(_MANIFEST, repo_root=repo, since_sha=base)
    assert plan.escalated_to_full
    assert plan.check_ids == _FULL_SET
    assert plan.unmapped_paths == ("intergrax/zz_old/a.py",)


def test_wd_a08_non_product_paths_do_not_escalate(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = (
        "docs/x.md",
        "tests/unit/zz.py",
        "scripts/maintenance/zz.py",
        ".github/workflows/zz.yml",
        "governance/tickets/T.md",
    )
    prefixes = load_default_impact_map()
    selected = _select(monkeypatch, *paths)
    # Non-product paths select the floor plus whatever the shipped map attaches to them; never the FULL set.
    mapped = set().union(*(check_ids_for_changed_path(p, prefixes) & _MANIFEST_IDS for p in paths))
    assert selected == _FLOOR | mapped
    assert selected < _FULL_SET


def test_wd_a09_empty_diff_is_floor_not_full(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _select(monkeypatch) == _FLOOR


def test_wd_a10_prefix_with_only_unknown_ids_counts_unmapped() -> None:
    synthetic = {"intergrax/zz_pkg": frozenset({"C999"})}
    path = "intergrax/zz_pkg/mod.py"
    assert check_ids_for_changed_path(path, synthetic) == frozenset({"C999"})
    unmapped = im.unmapped_product_paths(
        frozenset({path}),
        prefixes=synthetic,
        manifest_check_ids=_MANIFEST_IDS,
        entrypoint_covered=frozenset(),
    )
    assert unmapped == frozenset({path})


def test_wd_a11_malformed_impact_map_fails_closed(tmp_path: Path) -> None:
    cases = {
        "missing_table": {"schema_version": 1},
        "wrong_schema": {"schema_version": 2, "path_prefix_to_check_ids": {}},
        "non_list_ids": {"schema_version": 1, "path_prefix_to_check_ids": {"intergrax/x": "ENT-A-BND-01"}},
        "non_string_ids": {"schema_version": 1, "path_prefix_to_check_ids": {"intergrax/x": [1]}},
        "table_not_object": {"schema_version": 1, "path_prefix_to_check_ids": []},
        "top_level_list": [],
    }
    for name, payload in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="impact map"):
            load_impact_map_from_path(path)
    ok = tmp_path / "ok.json"
    ok.write_text(
        json.dumps({"schema_version": 1, "path_prefix_to_check_ids": {"intergrax/x/": ["ENT-A-BND-01"]}}),
        encoding="utf-8",
    )
    assert load_impact_map_from_path(ok) == {"intergrax/x": frozenset({"ENT-A-BND-01"})}


def _package_probe_paths() -> list[str]:
    probes: list[str] = []
    for top in ("intergrax", "agents", "applications"):
        for child in sorted(p for p in (_ROOT / top).iterdir() if p.is_dir() and not p.name.startswith(("__", "."))):
            probes.append(f"{top}/{child.name}/zz_wd_probe.py")
            if top == "intergrax":
                for grand in sorted(p for p in child.iterdir() if p.is_dir() and not p.name.startswith(("__", "."))):
                    probes.append(f"{top}/{child.name}/{grand.name}/zz_wd_probe.py")
    return probes


def test_wd_a12_no_floor_only_outcome_for_any_product_package(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = _package_probe_paths()
    assert len(probes) > 100
    prefixes = load_default_impact_map()
    monkeypatch.setattr(cs, "assert_git_ancestor", lambda *_a, **_k: None)
    floor_only: list[str] = []
    for probe in probes:
        monkeypatch.setattr(cs, "git_changed_files", lambda _root, _sha, p=probe: frozenset({p}))
        plan = cs.plan_changed_since_selection(_MANIFEST, repo_root=_ROOT, since_sha="0" * 40)
        covered = bool(check_ids_for_changed_path(probe, prefixes) & _MANIFEST_IDS)
        if plan.escalated_to_full:
            assert plan.check_ids == _FULL_SET, probe
        elif not covered:
            floor_only.append(probe)
    assert floor_only == []


def test_wd_a13_escalation_is_disclosed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = _plan(monkeypatch, "intergrax/zz_wd_new_pkg/mod.py")
    assert plan.reasons
    report, _manifest, _code = EnterpriseAuditRunner(_ROOT, dry_run=True).run(
        AuditMode.CHANGED_SINCE,
        output_dir=tmp_path / "a13",
        changed_check_ids=plan.check_ids,
        selection_notes=plan.reasons,
    )
    joined = "\n".join(report.limitations)
    assert "escalated to FULL" in joined
    assert "intergrax/zz_wd_new_pkg/mod.py" in joined
    written = json.loads((tmp_path / "a13" / "enterprise_audit_report.json").read_text(encoding="utf-8"))
    assert any("escalated to FULL" in item for item in written["limitations"])


def test_wd_a14_full_set_is_the_full_mode_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    full = frozenset(c.check_id for c in checks_for_mode(_MANIFEST, AuditMode.FULL))
    assert full == _FULL_SET
    assert len(full) == len(_MANIFEST)


def test_wd_a15_product_prefix_parity_with_iiac_common() -> None:
    tree = ast.parse((_ROOT / "scripts/governance/iiac_common.py").read_text(encoding="utf-8"))
    found: tuple[str, ...] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PRODUCT_PATH_PREFIXES" for t in node.targets
        ):
            found = tuple(ast.literal_eval(node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PRODUCT_PATH_PREFIXES"
        ):
            assert node.value is not None
            found = tuple(ast.literal_eval(node.value))
    assert found is not None
    assert found == im.PRODUCT_ESCALATION_PREFIXES


def _tracked_paths() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def _a16_probe_paths() -> list[str]:
    """First tracked file of every directory (the impact map is prefix based) plus the whole engine tree."""
    seen: set[str] = set()
    probes: list[str] = []
    for path in _tracked_paths():
        parent = path.rpartition("/")[0]
        engine = path.startswith(("intergrax/platform/enterprise_audit/", "scripts/enterprise_audit/"))
        if engine or parent not in seen:
            seen.add(parent)
            probes.append(path)
    return probes


def test_wd_a16_selection_never_shrinks_versus_prefix_union(monkeypatch: pytest.MonkeyPatch) -> None:
    prefixes = load_default_impact_map()
    current: list[frozenset[str]] = [frozenset()]
    monkeypatch.setattr(cs, "load_default_impact_map", lambda: prefixes)
    monkeypatch.setattr(cs, "assert_git_ancestor", lambda *_a, **_k: None)
    monkeypatch.setattr(cs, "git_changed_files", lambda _root, _sha: current[0])
    checked = 0
    for path in _a16_probe_paths():
        if path.startswith("tests/qualification/evol_x/"):
            continue  # evol_x refusals are covered by test_wd_evol_x_tree_mapping
        current[0] = frozenset({path})
        try:
            selected = cs.select_checks_for_changed_since(_MANIFEST, repo_root=_ROOT, since_sha="0" * 40)
        except ValueError:
            continue  # a refusal is not a shrink; the engine-tree and platform cases are covered by A03 and A04
        old = _FLOOR | (selected_check_ids_from_impact(current[0], prefixes=prefixes) & _MANIFEST_IDS)
        old |= {c.check_id for c in _MANIFEST if cs._check_touched(c, current[0])}
        assert old <= selected, path
        checked += 1
    assert checked > 500
