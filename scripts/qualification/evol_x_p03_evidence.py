#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""EVOL-X Protocol 03 evidence tool (stage-local; does not modify EBH-7 tooling).

Subcommands (all failures exit 2; tool errors exit 1):

* ``build-bundle``              W6-06 producer: write ``evol_x_p03_independent_replay/**`` from a VL3 JUnit XML.
* ``verify-evidence-publish``   Mode A: ``HEAD`` is the evidence commit (ancestry, digests, bindings, FRZ x10, exp x16).
* ``verify-test-execution``     Mode B: ``HEAD`` is TEST_EXECUTION_SHA; optional canonical outcome replay compare.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from testing_support.change_safety.contracts.artifacts import (
    EvolJunitOutcomes,
    EvolMetricsArtifact,
    FrzEvidence,
    MetricReadingArtifact,
    P03Manifest,
    TestExecutionManifest,
)
from testing_support.change_safety.evidence import p03

_REPO = Path(__file__).resolve().parents[2]
_PLAN_PUBLICATION_REF = "b4fca4bc4"
_EXECUTED_COMMAND = "uv run pytest tests/qualification/evol_x -q"
_IMPLEMENTATION_PATHS = ("testing_support", "tests/qualification/evol_x", "scripts/qualification")


def _out(cmd: list[str], repo: Path) -> str:
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def outcomes_from_junit(xml_path: Path, repo: Path) -> dict[str, str]:
    """Canonical ``node id -> outcome`` map from a pytest JUnit XML (classname resolved to the longest real path)."""
    outcomes: dict[str, str] = {}
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        parts = case.get("classname", "").split(".")
        path = ""
        for cut in range(len(parts), 0, -1):
            candidate = "/".join(parts[:cut]) + ".py"
            if (repo / candidate).is_file():
                path = candidate
                break
        node_id = f"{path}::{case.get('name', '')}"
        status = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                status = child.tag
        outcomes[node_id] = status
    return outcomes


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def build_bundle(repo: Path, junit_xml: Path, out_dir: Path, plan_sha: str) -> list[str]:
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
    from testing_support.change_safety.discovery.closed_world_gates import run_cw08, run_cw10
    from testing_support.change_safety.discovery.extractors import reset_fact_cache
    from testing_support.change_safety.selection.ci_manifest import load_pinned_manifest, manifest_content_digest
    from tests.qualification.evol_x._constants import (
        FRZ_EVOL_07_SCOPE_FOOTNOTE,
        FRZ_EVOL_08_SCOPE_FOOTNOTE,
        FRZ_EVOL_10_SCOPE_FOOTNOTE,
    )
    from tests.qualification.evol_x.experiments.arms import run_all
    from tests.qualification.evol_x.metrics.measure import collect_all

    problems: list[str] = []
    if _out(["git", "status", "--porcelain", "--untracked-files=no"], repo):
        return ["working tree is dirty: TEST_EXECUTION_SHA needs a clean tracked tree"]
    tes = _out(["git", "rev-parse", "HEAD"], repo)
    outcomes = outcomes_from_junit(junit_xml, repo)
    pinned = tuple(load_pinned_manifest()["pytest_node_ids"])
    missing = [n for n in pinned if n not in outcomes]
    if missing:
        return [f"JUnit lacks {len(missing)} manifest node(s), first: {missing[0]}"]
    bad = [n for n, o in outcomes.items() if o != "passed"]
    if bad:
        return [f"{len(bad)} non-passed outcome(s), first: {bad[0]}={outcomes[bad[0]]}"]

    cold_a = build_closed_world_inventory(source_commit_sha=tes, use_cache=False)
    cold_b = build_closed_world_inventory(source_commit_sha=tes, use_cache=False)
    reset_fact_cache()
    incremental = build_closed_world_inventory(source_commit_sha=tes, use_cache=True)
    cw08, cw10 = run_cw08(cold_a, cold_b), run_cw10(cold_a, incremental)
    if not (cw08.passed and cw10.passed):
        return [f"dual cold build gates failed: CW-08={cw08.passed} CW-10={cw10.passed}"]

    _dump(out_dir / p03.JUNIT_NAME, EvolJunitOutcomes(source_commit_sha=tes, outcomes=outcomes).model_dump())
    tem = TestExecutionManifest(
        source_commit_sha=tes,
        node_ids=pinned,
        manifest_content_digest=manifest_content_digest(pinned),
        excluded_populations=("DIM-B-PROD-Q-LIVE",),
        exclusions=("tests/unit/agents/test_reference_harness_strict.py::RC-7_BASELINE_DEFECT",),
    )
    _dump(out_dir / p03.TEST_EXECUTION_MANIFEST_NAME, tem.model_dump())
    _dump(
        out_dir / p03.INVENTORY_BINDING_NAME,
        {
            "schema_version": "evol_inventory_binding.v1",
            "source_commit_sha": tes,
            "inventory_digest": cold_a.inventory_digest,
            "node_count": len(cold_a.nodes),
            "edge_count": len(cold_a.edges),
            "CW-08": {"passed": cw08.passed, **cw08.metrics},
            "CW-10": {"passed": cw10.passed, **cw10.metrics},
        },
    )
    readings = collect_all(cold_a, tes)
    metrics = EvolMetricsArtifact(
        source_commit_sha=tes,
        readings=tuple(
            MetricReadingArtifact(
                metric_id=r.metric_id,
                numerator=r.numerator,
                denominator=r.denominator,
                status=r.status.value,
                threshold=r.threshold,
                passed=r.passed,
                population_digest=r.population_digest,
                note=r.note,
            )
            for r in readings
        ),
    )
    _dump(out_dir / p03.METRICS_NAME, metrics.model_dump())
    experiment_paths = []
    for arm in run_all(cold_a):
        rel = f"{p03.EXPERIMENTS_DIR}/{arm.scenario_id}_{arm.arm}.json"
        _dump(out_dir / rel, arm.to_verdict(tes).model_dump())
        experiment_paths.append(rel)
    footnotes = {
        "FRZ-EVOL-07": FRZ_EVOL_07_SCOPE_FOOTNOTE,
        "FRZ-EVOL-08": FRZ_EVOL_08_SCOPE_FOOTNOTE,
        "FRZ-EVOL-10": FRZ_EVOL_10_SCOPE_FOOTNOTE,
    }
    rows = []
    for index, frz_id in enumerate(p03.EXPECTED_FRZ_IDS, start=1):
        prefix = f"tests/qualification/evol_x/frz/test_evol_x_frz_{index:02d}_"
        nodes = sorted(n for n in outcomes if n.startswith(prefix))
        rows.append(
            FrzEvidence(
                frz_id=frz_id,
                behavioral_test=nodes[0] if nodes else "",
                passed=bool(nodes) and all(outcomes[n] == "passed" for n in nodes),
                source_commit_sha=tes,
                scope_footnote=footnotes.get(frz_id, ""),
            ).model_dump()
        )
    _dump(out_dir / p03.FRZ_NAME, {"schema_version": "evol_frz_evidence_set.v1", "rows": rows})

    artifact_paths = tuple(
        sorted(
            [
                p03.JUNIT_NAME,
                p03.TEST_EXECUTION_MANIFEST_NAME,
                p03.INVENTORY_BINDING_NAME,
                p03.METRICS_NAME,
                p03.FRZ_NAME,
            ]
            + experiment_paths
        )
    )
    p03.write_digests(out_dir, artifact_paths)
    manifest = P03Manifest(
        plan_publication_sha=plan_sha,
        test_execution_sha=tes,
        implementation_sha=_out(["git", "log", "-1", "--format=%H", "--", *_IMPLEMENTATION_PATHS], repo),
        independent_replay_sha=tes,
        pytest_profile="evol_x_p3_l3",
        uv_extra="dev-ci",
        executed_command=_EXECUTED_COMMAND,
        parent_test_execution_tree=_out(["git", "rev-parse", f"{tes}^{{tree}}"], repo),
        artifact_paths=artifact_paths,
    )
    _dump(out_dir / p03.MANIFEST_NAME, manifest.model_dump())
    return problems


def _report(problems: list[str], ok_label: str, extra: dict[str, str]) -> int:
    if problems:
        sys.stderr.write("\n".join(f"EVIDENCE_INTEGRITY_FAIL: {p}" for p in problems) + "\n")
        return 2
    sys.stdout.write(json.dumps({"status": ok_label, **extra}) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument("--bundle-rel", default=p03.BUNDLE_REL)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-bundle")
    build.add_argument("--junit-xml", type=Path, required=True)
    build.add_argument("--plan-sha", default=None)
    sub.add_parser("verify-evidence-publish")
    mode_b = sub.add_parser("verify-test-execution")
    mode_b.add_argument("--evidence-ref", required=True)
    mode_b.add_argument("--replay-junit", type=Path, default=None)
    args = parser.parse_args(argv)
    repo: Path = args.repo
    try:
        if args.command == "build-bundle":
            plan_sha = args.plan_sha or _out(["git", "rev-parse", _PLAN_PUBLICATION_REF], repo)
            problems = build_bundle(repo, args.junit_xml, repo / args.bundle_rel, plan_sha)
            return _report(problems, "BUNDLE_WRITTEN", {"bundle": args.bundle_rel})
        if args.command == "verify-evidence-publish":
            problems = p03.verify_mode_a(repo, bundle_rel=args.bundle_rel)
            return _report(problems, "EVIDENCE_INTEGRITY_OK", {"mode": "A", "evidence_head": p03.head_sha(repo)})
        replay = outcomes_from_junit(args.replay_junit, repo) if args.replay_junit else None
        problems = p03.verify_mode_b(
            repo, evidence_ref=args.evidence_ref, bundle_rel=args.bundle_rel, replay_outcomes=replay
        )
        return _report(problems, "EVIDENCE_INTEGRITY_OK", {"mode": "B", "test_execution_sha": p03.head_sha(repo)})
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        sys.stderr.write(f"tool error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
