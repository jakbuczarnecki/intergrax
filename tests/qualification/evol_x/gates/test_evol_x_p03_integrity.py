# © Artur Czarnecki. All rights reserved.

"""W6-03: Part V-I negative cases — the P03 verifier is not a HEAD-only check (temporary git repos)."""

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from testing_support.change_safety.contracts.artifacts import P03Manifest
from testing_support.change_safety.evidence import p03
from testing_support.change_safety.selection.ci_manifest import manifest_content_digest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO_ROOT / "scripts" / "qualification" / "evol_x_p03_evidence.py"
_NODES = ("tests/a.py::test_one", "tests/a.py::test_two")
Mutator = Callable[[dict[str, Any]], None]


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    import os

    proc = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, env={**os.environ, **env}
    )
    return proc.stdout.strip()


@dataclass(frozen=True)
class Repo:
    path: Path
    plan: str
    tes: str
    evidence: str


def _files(tes: str) -> dict[str, Any]:
    files: dict[str, Any] = {
        p03.JUNIT_NAME: {
            "schema_version": "evol_junit_outcomes.v1",
            "source_commit_sha": tes,
            "outcomes": dict.fromkeys(_NODES, "passed"),
        },
        p03.TEST_EXECUTION_MANIFEST_NAME: {
            "schema_version": "evol_test_execution_manifest.v1",
            "source_commit_sha": tes,
            "node_ids": list(_NODES),
            "manifest_content_digest": manifest_content_digest(_NODES),
            "excluded_populations": [],
            "exclusions": [],
        },
        p03.INVENTORY_BINDING_NAME: {"source_commit_sha": tes, "inventory_digest": "d"},
        p03.METRICS_NAME: {
            "schema_version": "evol_metrics.v1",
            "source_commit_sha": tes,
            "readings": [
                {"metric_id": "EVOL-M-01", "numerator": 40, "denominator": 40, "status": "MEASURED", "passed": True},
                {
                    "metric_id": "EVOL-M-09T",
                    "numerator": 3,
                    "denominator": 3,
                    "status": "MEASURED",
                    "threshold": 1.0,
                    "passed": True,
                    "population_digest": "d",
                },
                {
                    "metric_id": "EVOL-M-09H",
                    "numerator": 0,
                    "denominator": 0,
                    "status": "NOT_MEASURED",
                    "threshold": 1.0,
                    "passed": None,
                    "population_digest": None,
                },
            ],
        },
        p03.FRZ_NAME: {
            "rows": [
                {
                    "frz_id": frz,
                    "behavioral_test": f"t::{frz}",
                    "passed": True,
                    "source_commit_sha": tes,
                    "scope_footnote": "scope" if frz in p03.FRZ_SCOPE_FOOTNOTE_REQUIRED else "",
                }
                for frz in p03.EXPECTED_FRZ_IDS
            ]
        },
    }
    for index in range(p03.EXPECTED_EXPERIMENT_VERDICTS):
        files[f"{p03.EXPERIMENTS_DIR}/S{index // 2}_{'positive' if index % 2 == 0 else 'negative'}.json"] = {
            "scenario_id": f"S{index // 2}",
            "arm": "positive" if index % 2 == 0 else "negative",
            "passed": True,
            "source_commit_sha": tes,
        }
    return files


def _write_bundle(
    repo: Path, tes: str, plan: str, *, mutate: Mutator | None, manifest_extra: dict[str, Any] | None, tamper: bool
) -> None:
    bundle = repo / p03.BUNDLE_REL
    files = _files(tes)
    if mutate:
        mutate(files)
    for rel, payload in files.items():
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    artifact_paths = tuple(sorted(files))
    p03.write_digests(bundle, artifact_paths)
    if tamper:
        digests_file = bundle / p03.DIGESTS_NAME
        doc = json.loads(digests_file.read_text(encoding="utf-8"))
        doc["digests"][p03.METRICS_NAME] = "0" * 64
        digests_file.write_text(json.dumps(doc), encoding="utf-8", newline="\n")
    manifest = P03Manifest(
        plan_publication_sha=plan,
        test_execution_sha=tes,
        implementation_sha=tes,
        independent_replay_sha=tes,
        pytest_profile="evol_x_p3_l3",
        uv_extra="dev-ci",
        executed_command="uv run pytest tests/qualification/evol_x -q",
        artifact_paths=artifact_paths,
    ).model_dump()
    manifest.update(manifest_extra or {})
    (bundle / p03.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")


def _make_repo(
    tmp_path: Path,
    *,
    mutate: Mutator | None = None,
    manifest_extra: dict[str, Any] | None = None,
    tamper: bool = False,
    tes_override: str | None = None,
) -> Repo:
    _git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("plan", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "plan")
    plan = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "implementation")
    tes = _git(tmp_path, "rev-parse", "HEAD")
    if tes_override == "UNRELATED":
        tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
        tes_override = _git(tmp_path, "commit-tree", tree, "-m", "side-root")
    _write_bundle(tmp_path, tes_override or tes, plan, mutate=mutate, manifest_extra=manifest_extra, tamper=tamper)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "evidence")
    return Repo(tmp_path, plan, tes, _git(tmp_path, "rev-parse", "HEAD"))


def test_valid_bundle_passes_mode_a_and_mode_b(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert repo.evidence != repo.tes
    assert p03.verify_mode_a(repo.path, technical=True) == []
    _git(repo.path, "checkout", "-q", repo.tes)
    replay = dict.fromkeys(_NODES, "passed")
    assert p03.verify_mode_b(repo.path, evidence_ref=repo.evidence, replay_outcomes=replay, certification=True) == []


def test_mode_b_head_differs_from_manifest_test_execution_sha(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)  # HEAD == evidence commit, not TEST_EXECUTION_SHA
    problems = p03.verify_mode_b(repo.path, evidence_ref=repo.evidence)
    assert any("test_execution_sha" in p for p in problems)


def test_mode_a_test_execution_sha_not_ancestor_of_evidence_head(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, tes_override="UNRELATED")
    assert p03.classify_ancestry(repo.path, _git(repo.path, "rev-parse", "HEAD~1"), repo.evidence) == "ok"
    assert any("not an ancestor" in p for p in p03.verify_mode_a(repo.path))


def test_mode_a_reversed_ancestry_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
    later = _git(tmp_path, "commit-tree", tree, "-p", repo.evidence, "-m", "execution after evidence")
    assert p03.classify_ancestry(repo.path, later, repo.evidence) == "reversed"
    assert p03.classify_ancestry(repo.path, repo.tes, repo.evidence) == "ok"
    assert p03.classify_ancestry(repo.path, repo.evidence, repo.evidence) == "same"


def test_manifest_with_forbidden_self_field_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, manifest_extra={"evidence_publication_sha": "a" * 40})
    assert any("forbidden self-referential field" in p for p in p03.verify_mode_a(repo.path))


def test_swapped_canonical_junit_outcome_is_rejected(tmp_path: Path) -> None:
    def swap(files: dict[str, Any]) -> None:
        files[p03.JUNIT_NAME]["outcomes"][_NODES[0]] = "failed"

    repo = _make_repo(tmp_path, mutate=swap)  # digests regenerated: only the outcome check can fire
    assert any("non-passed outcome" in p for p in p03.verify_mode_a(repo.path))
    replayed = dict.fromkeys(_NODES, "passed")
    assert p03.compare_outcomes({_NODES[0]: "failed", _NODES[1]: "passed"}, replayed)


def test_missing_manifest_node_in_canonical_outcomes_is_rejected(tmp_path: Path) -> None:
    def drop(files: dict[str, Any]) -> None:
        del files[p03.JUNIT_NAME]["outcomes"][_NODES[1]]

    repo = _make_repo(tmp_path, mutate=drop)
    assert any("missing manifest node" in p for p in p03.verify_mode_a(repo.path))


def test_stale_source_commit_sha_in_inventory_is_rejected(tmp_path: Path) -> None:
    def stale(files: dict[str, Any]) -> None:
        files[p03.INVENTORY_BINDING_NAME]["source_commit_sha"] = "b" * 40

    repo = _make_repo(tmp_path, mutate=stale)
    assert any("stale source_commit_sha" in p for p in p03.verify_mode_a(repo.path))


def test_tampered_artifact_digest_entry_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, tamper=True)
    assert any("tampered ARTIFACT_DIGESTS entry" in p for p in p03.verify_mode_a(repo.path))


def test_failing_metric_or_experiment_or_frz_is_rejected(tmp_path: Path) -> None:
    def break_all(files: dict[str, Any]) -> None:
        files[p03.METRICS_NAME]["readings"][0]["passed"] = False
        files[p03.FRZ_NAME]["rows"][0]["passed"] = False
        files[f"{p03.EXPERIMENTS_DIR}/S0_positive.json"]["passed"] = False

    repo = _make_repo(tmp_path, mutate=break_all)
    joined = " | ".join(p03.verify_mode_a(repo.path))
    assert "metric not passing" in joined and "FRZ behavioral test not passed" in joined
    assert "experiment arm not passed" in joined


def test_mode_b_certification_without_replay_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _git(repo.path, "checkout", "-q", repo.tes)
    problems = p03.verify_mode_b(repo.path, evidence_ref=repo.evidence, certification=True)
    assert any("certification Mode B requires replay" in p for p in problems)


def test_mode_a_final_requires_m09h_and_human_drill(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert any("EVOL-M-09H" in p for p in p03.verify_mode_a(repo.path))


def test_composite_m09_pass_without_m09h_is_rejected(tmp_path: Path) -> None:
    def add_composite(files: dict[str, Any]) -> None:
        files[p03.METRICS_NAME]["readings"].append(
            {
                "metric_id": "EVOL-M-09",
                "numerator": 1,
                "denominator": 1,
                "status": "MEASURED",
                "passed": True,
            }
        )

    repo = _make_repo(tmp_path, mutate=add_composite)
    assert any("composite EVOL-M-09" in p for p in p03.verify_mode_a(repo.path, technical=True))


def test_cli_exits_2_on_integrity_failure_and_0_on_success(tmp_path: Path) -> None:
    good = _make_repo(tmp_path / "good") if (tmp_path / "good").mkdir() is None else None
    assert good is not None
    ok = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--repo",
            str(good.path),
            "verify-evidence-publish",
            "--technical",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    (tmp_path / "bad").mkdir()
    bad = _make_repo(tmp_path / "bad", tamper=True)
    failed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--repo", str(bad.path), "verify-evidence-publish"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 2 and "tampered" in failed.stderr
