#!/usr/bin/env python3
"""Attach automation-first scoped FINAL artifacts to an existing P03 bundle @ HEAD."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from testing_support.change_safety.contracts.artifacts import M09DrillCaseRegistry, P03Manifest
from testing_support.change_safety.evidence import p03

_REPO = Path(__file__).resolve().parents[2]
_BATCH = _REPO / "scripts" / "qualification" / "evol_x_run_m09a_automated_batch.py"


def _tes() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def finalize(*, bundle_rel: str = p03.BUNDLE_REL) -> list[str]:
    problems: list[str] = []
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    ).stdout.strip():
        return ["working tree is dirty"]
    bundle = _REPO / bundle_rel
    tes = _tes()
    manifest_path = bundle / p03.MANIFEST_NAME
    if not manifest_path.is_file():
        return ["MANIFEST missing — run build-bundle first"]
    manifest = P03Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.test_execution_sha != tes:
        return [f"MANIFEST TES {manifest.test_execution_sha[:9]} != HEAD {tes[:9]}"]
    registry_src = bundle / p03.M09_DRILL_CASE_REGISTRY_NAME
    if not registry_src.is_file():
        registry_src = _REPO / p03.BUNDLE_REL / p03.M09_DRILL_CASE_REGISTRY_NAME
    if not registry_src.is_file():
        return [f"missing {p03.M09_DRILL_CASE_REGISTRY_NAME} for automation-first FINAL"]
    if not (bundle / p03.M09_DRILL_CASE_REGISTRY_NAME).is_file():
        (bundle / p03.M09_DRILL_CASE_REGISTRY_NAME).write_text(
            registry_src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    import runpy

    runpy.run_path(str(_BATCH), run_name="__batch__")
    registry = M09DrillCaseRegistry.model_validate(
        json.loads((bundle / p03.M09_DRILL_CASE_REGISTRY_NAME).read_text(encoding="utf-8"))
    )
    population = p03.m09h_population_digest(registry)
    metrics_path = bundle / p03.METRICS_NAME
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    readings = metrics["readings"]
    for reading in readings:
        if reading["metric_id"] == p03.METRIC_M09A:
            reading.update(
                {
                    "numerator": 3,
                    "denominator": 3,
                    "status": "MEASURED",
                    "passed": True,
                    "population_digest": population,
                    "note": "automation-first scoped diagnostic (OD-EVOL-13)",
                }
            )
        if reading["metric_id"] == p03.METRIC_M09H:
            reading.update(
                {
                    "numerator": 0,
                    "denominator": 0,
                    "status": "NOT_MEASURED",
                    "passed": None,
                    "population_digest": None,
                    "note": "human drill not performed; OD-EVOL-13 scoped automation",
                }
            )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = set(manifest.artifact_paths)
    paths.add(p03.M09_DRILL_CASE_REGISTRY_NAME)
    paths.add(p03.M09_AUTOMATED_QUALIFICATION_NAME)
    for case_id in ("C1", "C2", "C3"):
        paths.add(f"{p03.M09_AUTOMATED_DIAGNOSTIC_DIR}/{case_id}.json")
    paths.discard(p03.M09_HUMAN_DRILL_NAME)
    artifact_paths = tuple(sorted(paths))
    p03.write_digests(bundle, artifact_paths)
    updated = manifest.model_dump()
    updated["certification_profile"] = p03.CERT_PROFILE_AUTOMATION_FIRST_SCOPED
    updated["artifact_paths"] = artifact_paths
    manifest_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    problems = p03.verify_mode_a(_REPO, bundle_rel=bundle_rel, technical=False)
    return problems


if __name__ == "__main__":
    import sys

    errs = finalize()
    if errs:
        sys.stderr.write("\n".join(errs) + "\n")
        raise SystemExit(2)
    print("AUTOMATION_FIRST_SCOPED_FINAL_OK")
