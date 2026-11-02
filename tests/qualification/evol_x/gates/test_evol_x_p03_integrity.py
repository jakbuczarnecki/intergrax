# © Artur Czarnecki. All rights reserved.

"""W6-03: Part V-I negative cases — the P03 verifier is not a HEAD-only check (temporary git repos)."""

import json
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from testing_support.change_safety.contracts.artifacts import M09DrillCaseRegistry, P03Manifest
from testing_support.change_safety.evidence import p03
from testing_support.change_safety.selection.ci_manifest import manifest_content_digest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO_ROOT / "scripts" / "qualification" / "evol_x_p03_evidence.py"
_NODES = ("tests/a.py::test_one", "tests/a.py::test_two")
_M09_REGISTRY_TEMPLATE = json.loads(
    (_REPO_ROOT / p03.BUNDLE_REL / p03.M09_DRILL_CASE_REGISTRY_NAME).read_text(encoding="utf-8")
)
_M09_SEED_PATHS = tuple(case["changed_path"] for case in _M09_REGISTRY_TEMPLATE["cases"])
_M09_TES_SEED = _M09_REGISTRY_TEMPLATE["bound_test_execution_sha"]
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
                {
                    "metric_id": "EVOL-M-09A",
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


def _seed_m09_paths(repo_root: Path) -> None:
    for path in _M09_SEED_PATHS:
        raw = subprocess.run(
            ["git", "show", f"{_M09_TES_SEED}:{path}"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        target = repo_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def _m09_registry_doc(tes: str) -> dict[str, Any]:
    doc = json.loads(json.dumps(_M09_REGISTRY_TEMPLATE))
    doc["bound_test_execution_sha"] = tes
    return doc


def _m09_drill_doc(tes: str, registry: M09DrillCaseRegistry, registry_digest: str) -> dict[str, Any]:
    tool_seconds = (45.0, 60.0, 55.0)
    attribution_seconds = (600.0, 700.0, 650.0)
    cases = []
    for entry, tool, attribution in zip(registry.cases, tool_seconds, attribution_seconds, strict=True):
        cases.append(
            {
                "case_id": entry.case_id,
                "correct": True,
                "tool_seconds": tool,
                "attribution_seconds": attribution,
                "attributed_cause_identities": list(entry.expected_cause_identities),
            }
        )
    return {
        "schema_version": "evol_m09_human_drill.v2",
        "measurement_policy_id": "OD-EVOL-08-M1",
        "source_commit_sha": tes,
        "test_execution_sha": tes,
        "registry_digest": registry_digest,
        "cases": cases,
        "median_tool_seconds": statistics.median(tool_seconds),
        "median_attribution_seconds": statistics.median(attribution_seconds),
        "adversarial_control_passed": True,
        "drill_started_at_utc": "2026-11-01T10:00:00+00:00",
        "drill_completed_at_utc": "2026-11-01T10:15:00+00:00",
        "evaluator_id": "evol-w7-drill-evaluator",
        "evaluator_display_name": "EVOL-X W7 Human Drill Evaluator",
    }


def _apply_final_m09a(files: dict[str, Any], repo: Path, tes: str) -> None:
    registry_raw = _m09_registry_doc(tes)
    registry = M09DrillCaseRegistry.model_validate(registry_raw)
    registry_text = json.dumps(registry_raw, indent=2)
    registry_digest = p03.sha256_hex(registry_text.encode("utf-8"))
    files[p03.M09_DRILL_CASE_REGISTRY_NAME] = registry_raw
    population = p03.m09h_population_digest(registry)
    case_rows = []
    for entry in registry.cases:
        rel = f"{p03.M09_AUTOMATED_DIAGNOSTIC_DIR}/{entry.case_id}.json"
        diag = {"case_id": entry.case_id, "seed": entry.changed_path}
        digest = p03.sha256_hex(json.dumps(diag, indent=2).encode("utf-8"))
        files[rel] = diag
        case_rows.append(
            {
                "case_id": entry.case_id,
                "seed_changed_path": entry.changed_path,
                "input_digest": entry.input_digest,
                "tool_seconds": 1.0,
                "tool_sla_met": True,
                "direct_node_id": "module:x",
                "direct_node_kind": "DIRECT",
                "derived_cause_identities": list(entry.expected_cause_identities),
                "cause_identity_digest": entry.expected_cause_identities[0],
                "registry_agreement": True,
                "diagnostic_output_digest": digest,
                "diagnostic_output_relpath": rel,
            }
        )
    qual = {
        "schema_version": p03.M09_AUTOMATED_QUALIFICATION_SCHEMA,
        "measurement_mode": "AUTOMATED_DIAGNOSTIC",
        "qualification_policy_id": p03.M09_AUTOMATION_POLICY_ID,
        "operator_decision_id": p03.M09_AUTOMATION_POLICY_ID,
        "source_commit_sha": tes,
        "test_execution_sha": tes,
        "registry_path": p03.M09_DRILL_CASE_REGISTRY_NAME,
        "registry_digest": registry_digest,
        "run_id": "a" * 32,
        "execution_started_at_utc": "2026-11-01T10:00:00+00:00",
        "execution_finished_at_utc": "2026-11-01T10:00:05+00:00",
        "cases": case_rows,
        "summary": {
            "cases_passed": True,
            "median_tool_seconds": 1.0,
            "automated_adversarial_controls_passed": True,
        },
        "adversarial_controls": [{"control_id": "AC-AUTO-01", "passed": True, "detail": "ok", "automated_only": True}],
        "certification_disposition": "AUTOMATED_SCOPED_PASS",
        "human_m09h_measured": False,
    }
    files[p03.M09_AUTOMATED_QUALIFICATION_NAME] = qual
    for reading in files[p03.METRICS_NAME]["readings"]:
        if reading["metric_id"] == p03.METRIC_M09A:
            reading.update(
                {
                    "numerator": 3,
                    "denominator": 3,
                    "status": "MEASURED",
                    "passed": True,
                    "population_digest": population,
                }
            )
        if reading["metric_id"] == p03.METRIC_M09H:
            reading.update(
                {"numerator": 0, "denominator": 0, "status": "NOT_MEASURED", "passed": None, "population_digest": None}
            )


def _apply_final_m09(files: dict[str, Any], _repo: Path, tes: str) -> None:
    registry_raw = _m09_registry_doc(tes)
    registry = M09DrillCaseRegistry.model_validate(registry_raw)
    registry_text = json.dumps(registry_raw, indent=2)
    registry_digest = p03.sha256_hex(registry_text.encode("utf-8"))
    files[p03.M09_DRILL_CASE_REGISTRY_NAME] = registry_raw
    files[p03.M09_HUMAN_DRILL_NAME] = _m09_drill_doc(tes, registry, registry_digest)
    population = p03.m09h_population_digest(registry)
    for reading in files[p03.METRICS_NAME]["readings"]:
        if reading["metric_id"] == "EVOL-M-09H":
            reading.update(
                {
                    "numerator": 3,
                    "denominator": 3,
                    "status": "MEASURED",
                    "passed": True,
                    "population_digest": population,
                    "note": "human operator attribution drill (W7 supplement)",
                }
            )


def _write_bundle(
    repo: Path,
    tes: str,
    plan: str,
    *,
    mutate: Mutator | None,
    manifest_extra: dict[str, Any] | None,
    tamper: bool,
    final: bool,
    final_automation: bool = False,
) -> None:
    bundle = repo / p03.BUNDLE_REL
    files = _files(tes)
    if final:
        _apply_final_m09(files, repo, tes)
    if final_automation:
        _apply_final_m09a(files, repo, tes)
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
        certification_profile=p03.CERT_PROFILE_AUTOMATION_FIRST_SCOPED if final_automation else None,
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
    final: bool = False,
    final_automation: bool = False,
) -> Repo:
    _git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("plan", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "plan")
    plan = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    if final or final_automation:
        _seed_m09_paths(tmp_path)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "implementation")
    tes = _git(tmp_path, "rev-parse", "HEAD")
    if tes_override == "UNRELATED":
        tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
        tes_override = _git(tmp_path, "commit-tree", tree, "-m", "side-root")
    _write_bundle(
        tmp_path,
        tes_override or tes,
        plan,
        mutate=mutate,
        manifest_extra=manifest_extra,
        tamper=tamper,
        final=final,
        final_automation=final_automation,
    )
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


def test_mode_a_final_valid_m09_bundle_passes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, final=True)
    assert p03.verify_mode_a(repo.path) == []


def test_mode_a_automation_first_scoped_final_passes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, final_automation=True)
    assert p03.verify_mode_a(repo.path) == []


def _timings(tool: tuple[float, float, float], attribution: tuple[float, float, float]) -> Mutator:
    """Set per-case tool/attribution seconds and keep both declared medians consistent with the cases."""

    def mutate(files: dict[str, Any]) -> None:
        drill = files[p03.M09_HUMAN_DRILL_NAME]
        for case, tool_s, attribution_s in zip(drill["cases"], tool, attribution, strict=True):
            case["tool_seconds"] = tool_s
            case["attribution_seconds"] = attribution_s
        drill["median_tool_seconds"] = statistics.median(tool)
        drill["median_attribution_seconds"] = statistics.median(attribution)

    return mutate


def _swap_tool_and_attribution(files: dict[str, Any]) -> None:
    for case in files[p03.M09_HUMAN_DRILL_NAME]["cases"]:
        case["tool_seconds"], case["attribution_seconds"] = case["attribution_seconds"], case["tool_seconds"]


def _forged_m09h_pass_with_failing_attribution(files: dict[str, Any]) -> None:
    _timings((45.0, 60.0, 55.0), (1000.0, 1000.0, 1000.0))(files)
    reading = next(r for r in files[p03.METRICS_NAME]["readings"] if r["metric_id"] == "EVOL-M-09H")
    assert reading["status"] == "MEASURED" and reading["passed"] is True  # forged PASS row stays in place


def _worksheet_with_wrong_digest(files: dict[str, Any]) -> None:
    files[p03.M09_HUMAN_DRILL_WORKSHEET_NAME] = {"schema_version": "evol_m09_human_drill_worksheet.v1"}
    files[p03.M09_HUMAN_DRILL_NAME]["worksheet_digest"] = "0" * 64


_M09_NEGATIVES: tuple[tuple[str, Mutator, str], ...] = (
    (
        "N-M09-01",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"unexpected_field": True}),
        "invalid",
    ),
    (
        "N-M09-02",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"source_commit_sha": "b" * 40}),
        "stale source_commit_sha",
    ),
    (
        "N-M09-03",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"test_execution_sha": "c" * 40}),
        "test_execution_sha",
    ),
    (
        "N-M09-04",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"registry_digest": "0" * 64}),
        "registry_digest",
    ),
    (
        "N-M09-05",
        lambda files: files[p03.M09_DRILL_CASE_REGISTRY_NAME].update({"bound_test_execution_sha": "d" * 40}),
        "bound_test_execution_sha",
    ),
    (
        "N-M09-06",
        lambda files: files[p03.M09_DRILL_CASE_REGISTRY_NAME]["cases"][0].update({"input_digest": "0" * 64}),
        "input_digest mismatch",
    ),
    (
        "N-M09-07",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME]["cases"][0].update({"correct": False}),
        "not marked correct",
    ),
    (
        "N-M09-08",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME]["cases"][0].update({"tool_seconds": 121.0}),
        "tool_seconds>120",
    ),
    (
        "N-M09-09",
        lambda files: files[p03.M09_DRILL_CASE_REGISTRY_NAME]["cases"].pop(),
        "exactly 3 cases",
    ),
    (
        "N-M09-10",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"median_tool_seconds": 999.0}),
        "median_tool_seconds disagrees",
    ),
    (
        "N-M09-11",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"adversarial_control_passed": False}),
        "adversarial_control_passed",
    ),
    (
        "N-M09-12",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update(
            {
                "drill_started_at_utc": "2026-11-01T12:00:00+00:00",
                "drill_completed_at_utc": "2026-11-01T11:00:00+00:00",
            }
        ),
        "drill_completed_at_utc precedes",
    ),
    (
        "N-M09-13",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"evaluator_id": ""}),
        "String should have at least 1 character",
    ),
    (
        "N-M09-14",
        lambda files: next(r for r in files[p03.METRICS_NAME]["readings"] if r["metric_id"] == "EVOL-M-09H").update(
            {"numerator": 2}
        ),
        "3/3 MEASURED PASS",
    ),
    (
        "N-M09-15",
        lambda files: next(r for r in files[p03.METRICS_NAME]["readings"] if r["metric_id"] == "EVOL-M-09H").update(
            {"population_digest": "0" * 64}
        ),
        "population_digest inconsistent",
    ),
    (
        "N-M09-16",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME]["cases"][0].update({"attributed_cause_identities": ["0" * 64]}),
        "attributed causes mismatch",
    ),
    (
        "N-M09-17",
        _timings((4.0, 5.0, 6.0), (1200.0, 1500.0, 1800.0)),
        "median attribution_seconds>900",
    ),
    ("N-M09-18", _timings((121.0, 5.0, 6.0), (600.0, 700.0, 650.0)), "tool_seconds>120"),
    (
        "N-M09-19",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME]["cases"][1].pop("attribution_seconds"),
        "invalid",
    ),
    (
        "N-M09-20-negative",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME]["cases"][0].update({"attribution_seconds": -1.0}),
        "finite and >= 0",
    ),
    (
        "N-M09-20-nan",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME]["cases"][0].update({"attribution_seconds": float("nan")}),
        "finite and >= 0",
    ),
    (
        "N-M09-20-inf",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME]["cases"][0].update({"attribution_seconds": float("inf")}),
        "finite and >= 0",
    ),
    ("N-M09-21-swap", _swap_tool_and_attribution, "tool_seconds>120"),
    (
        "N-M09-22",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"median_attribution_seconds": 100.0}),
        "median_attribution_seconds disagrees",
    ),
    ("N-M09-23-FAIL", _timings((45.0, 60.0, 55.0), (900.0, 901.0, 902.0)), "median attribution_seconds>900"),
    (
        "N-M09-24",
        _forged_m09h_pass_with_failing_attribution,
        "median attribution_seconds>900",
    ),
    (
        "N-M09-25",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"schema_version": "evol_m09_human_drill.v1"}),
        "schema_version must be exactly evol_m09_human_drill.v2",
    ),
    (
        "N-M09-25b-policy-missing",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].pop("measurement_policy_id"),
        "measurement_policy_id must be exactly OD-EVOL-08-M1",
    ),
    (
        "N-M09-25c-policy-wrong",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"measurement_policy_id": "OD-EVOL-08"}),
        "measurement_policy_id must be exactly OD-EVOL-08-M1",
    ),
    (
        "N-M09-27",
        _worksheet_with_wrong_digest,
        "worksheet_digest mismatch",
    ),
    (
        "N-M09-28",
        lambda files: files[p03.M09_HUMAN_DRILL_NAME].update({"schema_version": "evol_m09_human_drill.v3"}),
        "schema_version must be exactly evol_m09_human_drill.v2",
    ),
    (
        "N-M09-29",
        lambda files: files[p03.M09_DRILL_CASE_REGISTRY_NAME]["cases"][0]["expected_cause_identities"].__setitem__(
            0, "0" * 64
        ),
        "expected_cause_identities mismatch",
    ),
)

# PASS arms: the verifier must accept these (median math, equality boundary, outlier robustness).
_M09_POSITIVES: tuple[tuple[str, Mutator], ...] = (
    ("N-M09-23-PASS-spike", _timings((45.0, 60.0, 55.0), (100.0, 100.0, 1200.0))),
    ("N-M09-26-PASS-boundary", _timings((45.0, 60.0, 55.0), (900.0, 900.0, 900.0))),
    ("N-M09-26b-PASS-mixed", _timings((45.0, 60.0, 55.0), (600.0, 900.0, 800.0))),
    ("N-M09-18b-PASS-tool-boundary", _timings((120.0, 120.0, 120.0), (600.0, 700.0, 650.0))),
)


@pytest.mark.parametrize(("case_id", "mutate", "needle"), _M09_NEGATIVES, ids=[case for case, _, _ in _M09_NEGATIVES])
def test_m09_human_drill_negative_cases(tmp_path: Path, case_id: str, mutate: Mutator, needle: str) -> None:
    _ = case_id
    repo = _make_repo(tmp_path, final=True, mutate=mutate)
    joined = " | ".join(p03.verify_mode_a(repo.path))
    assert needle in joined


@pytest.mark.parametrize(("case_id", "mutate"), _M09_POSITIVES, ids=[case for case, _ in _M09_POSITIVES])
def test_m09_human_drill_positive_cases(tmp_path: Path, case_id: str, mutate: Mutator) -> None:
    _ = case_id
    repo = _make_repo(tmp_path, final=True, mutate=mutate)
    assert p03.verify_mode_a(repo.path) == []


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
