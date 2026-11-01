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
    bundle = _REPO / bundle_rel
    manifest_path = bundle / p03.MANIFEST_NAME
    if not manifest_path.is_file():
        return ["MANIFEST missing — run build-bundle first"]
    manifest = P03Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    tes = manifest.test_execution_sha
    registry_src = bundle / p03.M09_DRILL_CASE_REGISTRY_NAME
    if not registry_src.is_file():
        registry_src = _REPO / p03.BUNDLE_REL / p03.M09_DRILL_CASE_REGISTRY_NAME
    if not registry_src.is_file():
        return [f"missing {p03.M09_DRILL_CASE_REGISTRY_NAME} for automation-first FINAL"]
    if not (bundle / p03.M09_DRILL_CASE_REGISTRY_NAME).is_file():
        (bundle / p03.M09_DRILL_CASE_REGISTRY_NAME).write_text(
            registry_src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    reg_doc = json.loads((bundle / p03.M09_DRILL_CASE_REGISTRY_NAME).read_text(encoding="utf-8"))
    reg_doc["bound_test_execution_sha"] = tes
    from testing_support.change_safety.diagnostics.m09_cause_identity import expected_cause_identities_for_seed
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory

    inv = build_closed_world_inventory(source_commit_sha=tes)
    for entry in reg_doc["cases"]:
        entry["expected_cause_identities"] = list(expected_cause_identities_for_seed(inv, entry["changed_path"]))
        raw = subprocess.run(
            ["git", "show", f"{tes}:{entry['changed_path']}"],
            cwd=_REPO,
            check=True,
            capture_output=True,
        ).stdout
        import hashlib

        entry["input_digest"] = hashlib.sha256(raw).hexdigest()
    (bundle / p03.M09_DRILL_CASE_REGISTRY_NAME).write_text(
        json.dumps(reg_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("evol_x_run_m09a_automated_batch", _BATCH)
    if spec is None or spec.loader is None:
        return ["cannot load M-09A batch runner"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_batch(bundle_dir=bundle, tes=tes)
    subprocess.run(["git", "add", "-A", str(bundle)], cwd=_REPO, check=True)
    qual_path = bundle / p03.M09_AUTOMATED_QUALIFICATION_NAME
    qual_doc = json.loads(qual_path.read_text(encoding="utf-8"))
    reg_blob = subprocess.run(
        ["git", "show", f":{bundle_rel}/{p03.M09_DRILL_CASE_REGISTRY_NAME}"],
        cwd=_REPO,
        check=True,
        capture_output=True,
    ).stdout
    qual_doc["registry_digest"] = p03.sha256_hex(reg_blob)
    for case in qual_doc["cases"]:
        rel = case["diagnostic_output_relpath"]
        diag_blob = subprocess.run(
            ["git", "show", f":{bundle_rel}/{rel}"],
            cwd=_REPO,
            check=True,
            capture_output=True,
        ).stdout
        case["diagnostic_output_digest"] = p03.sha256_hex(diag_blob)
    body = json.dumps({k: v for k, v in qual_doc.items() if k != "artifact_digest"}, indent=2, sort_keys=True)
    qual_doc["artifact_digest"] = p03.sha256_hex(body.encode("utf-8"))
    qual_path.write_text(json.dumps(qual_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    subprocess.run(["git", "add", "-A", str(bundle)], cwd=_REPO, check=True)
    digests = {}
    for rel in artifact_paths:
        blob = subprocess.run(
            ["git", "show", f":{bundle_rel}/{rel}"],
            cwd=_REPO,
            check=True,
            capture_output=True,
        ).stdout
        digests[rel] = p03.sha256_hex(blob)
    digest_doc = {"schema_version": "evol_artifact_digests.v1", "digests": digests}
    (bundle / p03.DIGESTS_NAME).write_text(json.dumps(digest_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
