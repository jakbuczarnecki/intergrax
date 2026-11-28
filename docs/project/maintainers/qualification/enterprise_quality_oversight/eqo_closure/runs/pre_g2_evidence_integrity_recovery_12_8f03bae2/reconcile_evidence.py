#!/usr/bin/env python3
"""EC-W3-12 reconciliation emitter — recompute dispositions (does not run validator)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from _evidence_common import (
    build_test_path_reverse_index,
    compute_obligation_disposition,
    ec08_dir,
    inventory_binding,
    parse_junit_cases,
    repo_root_from_run,
)

RUN_ROOT = Path(__file__).resolve().parent


def _die(msg: str) -> None:
    print(f"RECONCILE_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load_execution_records() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rec_dir = RUN_ROOT / "execution_records"
    if not rec_dir.is_dir():
        return out
    for pre_path in rec_dir.glob("*.pre.json"):
        batch_id = pre_path.name.replace(".pre.json", "")
        post_path = rec_dir / f"{batch_id}.post.json"
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        post = json.loads(post_path.read_text(encoding="utf-8")) if post_path.is_file() else {}
        out[batch_id] = {"pre": pre, "post": post}
    return out


def emit() -> None:
    repo_root = repo_root_from_run(RUN_ROOT)
    baseline_path = RUN_ROOT / "BASELINE_BINDING.json"
    if not baseline_path.is_file():
        _die("missing BASELINE_BINDING.json for qualification source sha")
    source_sha = json.loads(baseline_path.read_text(encoding="utf-8")).get("qualification_source_sha")
    if not source_sha:
        _die("BASELINE_BINDING.json missing qualification_source_sha")
    obl_src = json.loads((ec08_dir(repo_root) / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    queue_src = json.loads((ec08_dir(repo_root) / "REMAINING_QUALIFICATION_QUEUE.json").read_text(encoding="utf-8"))
    binding = inventory_binding(repo_root)
    (RUN_ROOT / "INVENTORY_BINDING.json").write_text(json.dumps(binding, indent=2, sort_keys=True), encoding="utf-8")
    records = _load_execution_records()
    file_outcomes: dict[str, list] = defaultdict(list)
    testcase_matrix: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []

    for batch_id, pair in records.items():
        post = pair.get("post", {})
        pre = pair.get("pre", {})
        if post.get("source_changed_during_batch"):
            _die(f"source changed during batch {batch_id}")
        junit_rel = post.get("junit_path")
        if not junit_rel:
            batch_manifest.append({"batch_id": batch_id, "status": "MISSING_JUNIT"})
            continue
        junit_path = RUN_ROOT / junit_rel
        cases = parse_junit_cases(junit_path, batch_id=batch_id, source_sha=pre.get("source_commit_sha", source_sha))
        for case in cases:
            file_outcomes[case.file_path].append(case)
            testcase_matrix.append(
                {
                    "node_id": case.node_id,
                    "file_path": case.file_path,
                    "outcome": case.outcome,
                    "batch_id": case.batch_id,
                    "source_commit_sha": case.source_commit_sha,
                    "evidence_reference": f"execution_records/{batch_id}.post.json",
                }
            )
        batch_manifest.append(
            {
                "batch_id": batch_id,
                "process_returncode": post.get("process_returncode"),
                "argv_digest": pre.get("command_digest_sha256"),
                "source_commit_sha": pre.get("source_commit_sha"),
                "junit_sha256": post.get("junit_sha256"),
                "log_sha256": post.get("stdout_stderr_sha256"),
            }
        )

    reverse = build_test_path_reverse_index(obl_src)
    obligation_rows: list[dict[str, Any]] = []
    for ob in obl_src["obligations"]:
        mapped_paths = (
            set(ob.get("mapped_test_file_paths") or [])
            | set(ob.get("remaining_qualification_test_paths") or [])
            | set(ob.get("remaining_application_host_test_paths") or [])
        )
        disposition, limits = compute_obligation_disposition(ob, file_outcomes=file_outcomes, source_sha=source_sha)
        obligation_rows.append(
            {
                "obligation_id": ob["obligation_id"],
                "recalculated_disposition": disposition,
                "coverage_limitations": limits,
                "mapped_test_file_paths": ob.get("mapped_test_file_paths", []),
                "current_testcase_count": sum(len(file_outcomes.get(p, [])) for p in mapped_paths),
                "evidence_validity": "CURRENT",
                "historical_disposition_reference": ob.get("execution_disposition"),
                "semantic_review_required": ob.get("semantic_review_required", False),
                "evidence_reference": RUN_ROOT.name,
            }
        )

    qual_rows: list[dict[str, Any]] = []
    host_rows: list[dict[str, Any]] = []
    for item in queue_src.get("qualification_queue", []):
        path = item["path"]
        cases = file_outcomes.get(path, [])
        if not cases:
            disp = "NOT_EXECUTED"
            reason = "no_current_batch"
        elif any(c.outcome in ("FAIL", "ERROR", "COLLECTION_ERROR") for c in cases):
            disp = "RED"
            reason = "executable_failure"
        elif all(c.outcome == "PASS" for c in cases):
            disp = "GREEN"
            reason = "all_pass"
        else:
            disp = "QUALIFICATION_PENDING"
            reason = "mixed_or_skip"
        row = {
            **{k: item[k] for k in item if k not in ("current_result", "execution_disposition")},
            "recalculated_disposition": disp,
            "disposition_reason": reason,
            "source_commit_sha": source_sha,
            "testcase_outcomes": [c.__dict__ for c in cases],
            "obligation_ids": sorted(reverse.get(path, [])),
        }
        qual_rows.append(row)
    for item in queue_src.get("application_host_queue", []):
        path = item["path"]
        cases = file_outcomes.get(path, [])
        if not cases:
            disp, reason = "NOT_EXECUTED", "no_current_batch"
        elif any(c.outcome in ("FAIL", "ERROR", "COLLECTION_ERROR") for c in cases):
            disp, reason = "RED", "executable_failure"
        elif all(c.outcome == "PASS" for c in cases):
            disp, reason = "GREEN", "all_pass"
        else:
            disp, reason = "QUALIFICATION_PENDING", "mixed_or_skip"
        host_rows.append(
            {
                **{k: item[k] for k in item if k not in ("current_result", "execution_disposition")},
                "recalculated_disposition": disp,
                "disposition_reason": reason,
                "source_commit_sha": source_sha,
                "testcase_outcomes": [c.__dict__ for c in cases],
                "obligation_ids": sorted(reverse.get(path, [])),
            }
        )

    (RUN_ROOT / "TESTCASE_EXECUTION_MATRIX.json").write_text(
        json.dumps(
            {"schema": "eqo.testcase_execution_matrix.v1", "source_sha": source_sha, "testcases": testcase_matrix},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (RUN_ROOT / "BATCH_MANIFEST.json").write_text(
        json.dumps({"schema": "eqo.batch_manifest.v1", "batches": batch_manifest}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (RUN_ROOT / "OBLIGATION_EXECUTION_MATRIX.json").write_text(
        json.dumps(
            {
                "schema": "eqo.obligation_execution_matrix.v2",
                "source_sha": source_sha,
                "obligation_count": len(obligation_rows),
                "reconciled": False,
                "obligations": obligation_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (RUN_ROOT / "REMAINING_QUALIFICATION_QUEUE.json").write_text(
        json.dumps(
            {
                "schema": "eqo.remaining_qualification_queue.v2",
                "source_sha": source_sha,
                "queue_count": len(qual_rows) + len(host_rows),
                "qualification_test_file_count": len(qual_rows),
                "application_host_test_file_count": len(host_rows),
                "qualification_queue": qual_rows,
                "application_host_queue": host_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> int:
    subprocess.run([sys.executable, str(RUN_ROOT / "validate_reconciliation.py"), "--pre-check"], check=False)
    emit()
    proc = subprocess.run(
        [sys.executable, str(RUN_ROOT / "validate_reconciliation.py"), "--mode", "PARTIAL_VALIDATION"],
        check=False,
    )
    if proc.returncode != 0:
        _die("validator failed after reconciliation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
