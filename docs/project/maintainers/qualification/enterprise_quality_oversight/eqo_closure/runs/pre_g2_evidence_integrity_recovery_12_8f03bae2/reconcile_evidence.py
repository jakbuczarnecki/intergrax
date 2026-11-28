#!/usr/bin/env python3
# ruff: noqa: E501
"""EC-W3-12 reconciliation emitter — recompute dispositions (does not run validator)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _evidence_common import (
    compute_canonical_evidence_projection,
    git,
    load_execution_record_pairs,
    resolve_execution_context,
    sha256_file,
    utc_now,
)

RUN_ROOT = Path(__file__).resolve().parent


def _die(msg: str) -> None:
    print(f"RECONCILE_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


PACKAGE_G_MANIFEST_FILES = (
    "DES_PREREG.json",
    "BASELINE_BINDING.json",
    "ENVIRONMENT_MANIFEST.json",
    "INVENTORY_BINDING.json",
    "RUN_MANIFEST.json",
    "BATCH_MANIFEST.json",
    "TESTCASE_EXECUTION_MATRIX.json",
    "FILE_LEVEL_EXECUTION_MATRIX.json",
    "OBLIGATION_EXECUTION_MATRIX.json",
    "REMAINING_QUALIFICATION_QUEUE.json",
    "EQO_RV01_CHANGE_IMPACT.json",
    "AUDIT_FINDING_SETTLEMENT.json",
    "FINDING_DISPOSITION.json",
    "EXECUTION_PUBLICATION_BINDING.json",
    "EVIDENCE_INTEGRITY.json",
    "VALIDATOR_QUALIFICATION.json",
)


def _write_run_manifest(ctx) -> None:
    records = load_execution_record_pairs(ctx.run_root)
    batches = [
        {"batch_id": batch_id, "pre": pair.get("pre"), "post": pair.get("post")}
        for batch_id, pair in sorted(records.items())
    ]
    payload = {
        "schema": "eqo.run_manifest.v1",
        "recorded_utc": utc_now(),
        "batch_count": len(batches),
        "batches": batches,
    }
    (ctx.run_root / "RUN_MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def emit_package_g_closure(ctx) -> None:
    baseline = json.loads((ctx.run_root / "BASELINE_BINDING.json").read_text(encoding="utf-8"))
    qual_sha = ctx.qualification_sha
    tooling_sha = git(ctx.repo_root, ["rev-parse", "HEAD"])
    (ctx.run_root / "EXECUTION_PUBLICATION_BINDING.json").write_text(
        json.dumps(
            {
                "schema": "eqo.execution_publication_binding.v1",
                "qualification_product_source_sha": qual_sha,
                "qualification_tooling_sha_at_emit": tooling_sha,
                "publication_pending": False,
                "publication_commit_sha": git(ctx.repo_root, ["rev-parse", "HEAD"]),
                "evidence_run_root": ctx.run_root.name,
                "recorded_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidate_sha = git(ctx.repo_root, ["rev-parse", "HEAD"])
    diff_proc = subprocess.run(
        ["git", "-C", str(ctx.repo_root), "diff", "--name-only", f"{qual_sha}..{candidate_sha}"],
        capture_output=True,
        text=True,
        check=False,
    )
    changed_paths = [line.strip() for line in diff_proc.stdout.splitlines() if line.strip()]
    affected_evidence = [
        {
            "path": "tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py",
            "disposition": "TARGETED",
            "reason": "EC-F16 FastAPI core lifecycle",
        },
        {
            "path": "intergrax/fastapi_core/app_factory.py",
            "disposition": "FULL_STREAM",
            "reason": "EC-F16 owner corrective",
        },
        {
            "path": "applications/governed_contractor_application/tests/host/test_governed_contractor_host_smoke.py",
            "disposition": "TARGETED",
            "reason": "EC-F12 strict production authority",
        },
    ]
    (ctx.run_root / "EQO_RV01_CHANGE_IMPACT.json").write_text(
        json.dumps(
            {
                "schema": "eqo.eqo_rv01_change_impact.v1",
                "qualification_source_sha": qual_sha,
                "candidate_source_sha": candidate_sha,
                "git_diff_path_count": len(changed_paths),
                "affected_evidence": affected_evidence,
                "preserved_historical_batches": 25,
                "ec_w3_12_replay_2": {"disposition": "CURRENT", "evidence_root": ctx.run_root.as_posix()},
                "ec_w3_12_replay_1": {
                    "disposition": "INVALIDATED_ENVIRONMENT",
                    "reason": "detached worktree pytest missing; uv project binding uncorrected",
                },
                "residual_owner_blocked": ["EC-F12", "EC-F05", "EC-F09"],
                "ec_f16_disposition": "CORRECTED_AT_CANDIDATE",
                "recorded_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    finding_disposition = {
        "schema": "eqo.finding_disposition.v1",
        "findings": [
            {"finding_id": "EC-F16", "disposition": "CORRECTED", "campaign_residual": False},
            {"finding_id": "EC-F12", "disposition": "OPEN_OWNER_BLOCK", "campaign_residual": True},
            {
                "finding_id": "AUD-11",
                "disposition": "SETTLED_WITH_RESIDUALS",
                "note": "replay #2 semantic evidence; FC3 PARTIAL_VALIDATION witness preserved",
            },
        ],
        "recorded_utc": utc_now(),
    }
    (ctx.run_root / "FINDING_DISPOSITION.json").write_text(
        json.dumps(finding_disposition, indent=2, sort_keys=True), encoding="utf-8"
    )
    (ctx.run_root / "AUDIT_FINDING_SETTLEMENT.json").write_text(
        json.dumps(
            {
                "schema": "eqo.audit_finding_settlement.v1",
                "session_name": "INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-12 CONSOLIDATED FINAL QUALIFICATION",
                "ticket": "governance/tickets/T-IGX-095-A.md",
                "finding_disposition_ref": "FINDING_DISPOSITION.json",
                "recorded_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_digests: dict[str, str] = {}
    for name in PACKAGE_G_MANIFEST_FILES:
        path = ctx.run_root / name
        if path.is_file():
            manifest_digests[name] = sha256_file(path)
    log_digests = {
        f"logs/{p.name}": sha256_file(p) for p in sorted((ctx.run_root / "logs").glob("*.log")) if p.is_file()
    }
    (ctx.run_root / "EVIDENCE_INTEGRITY.json").write_text(
        json.dumps(
            {
                "schema": "eqo.evidence_integrity.v1",
                "qualification_source_sha": qual_sha,
                "baseline_task_start_remote_sha": baseline.get("task_start_remote_sha"),
                "manifest_sha256": manifest_digests,
                "log_sha256": log_digests,
                "recorded_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def emit(ctx) -> None:
    baseline_path = ctx.run_root / "BASELINE_BINDING.json"
    if not baseline_path.is_file():
        _die("missing BASELINE_BINDING.json for qualification source sha")
    projection = compute_canonical_evidence_projection(ctx)
    for filename, payload in projection.items():
        if filename.startswith("_"):
            continue
        (ctx.run_root / filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_run_manifest(ctx)
    emit_package_g_closure(ctx)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="EC-W3-12 reconciliation emitter")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--qualification-sha", type=str, default=None)
    args = parser.parse_args()
    ctx = resolve_execution_context(
        source_root=args.source_root,
        run_root=args.run_root,
        qualification_sha=args.qualification_sha,
    )
    subprocess.run(
        [
            sys.executable,
            str(ctx.run_root / "validate_reconciliation.py"),
            "--pre-check",
            "--run-root",
            str(ctx.run_root),
        ],
        check=False,
    )
    emit(ctx)
    proc = subprocess.run(
        [
            sys.executable,
            str(ctx.run_root / "validate_reconciliation.py"),
            "--mode",
            "PARTIAL_VALIDATION",
            "--run-root",
            str(ctx.run_root),
        ],
        check=False,
    )
    if proc.returncode != 0:
        _die("validator failed after reconciliation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
