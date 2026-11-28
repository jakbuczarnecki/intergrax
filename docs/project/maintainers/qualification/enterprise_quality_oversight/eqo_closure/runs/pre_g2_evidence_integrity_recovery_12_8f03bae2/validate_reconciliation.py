#!/usr/bin/env python3
# ruff: noqa: E501
"""Pure read-only semantic evidence validator for EC-W3-12."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _evidence_common import (
    VALIDATOR_REASONS,
    build_test_path_reverse_index,
    compute_obligation_disposition,
    ec08_dir,
    git,
    inventory_binding,
    merge_junit_with_node_capture,
    repo_root_from_run,
    sha256_file,
)
from validation_execution import (
    validate_disposition_reconciliation,
    validate_evidence_references,
    validate_execution_inventory,
    validate_immutable_inventory,
    validate_queue_exact,
    validate_synthetic_isolation,
    validate_testcase_node_ids,
)

RUN_ROOT = Path(__file__).resolve().parent


def fail(reason: str, detail: str = "") -> int:
    if reason not in VALIDATOR_REASONS:
        reason = "STRUCTURAL_INCONSISTENCY"
    msg = f"VALIDATION_FAIL:{reason}"
    if detail:
        msg = f"{msg}:{detail}"
    print(msg, file=sys.stderr)
    return 1


def validate(run_root: Path | None = None, *, mode: str = "PARTIAL_VALIDATION") -> int:
    root = run_root or RUN_ROOT
    repo_root = repo_root_from_run(root)
    final_mode = mode == "FINAL_VALIDATION"
    if final_mode:
        rc = validate_synthetic_isolation(root, fail)
        if rc:
            return rc
    required = [
        "DES_PREREG.json",
        "BASELINE_BINDING.json",
        "ENVIRONMENT_MANIFEST.json",
        "INVENTORY_BINDING.json",
        "OBLIGATION_EXECUTION_MATRIX.json",
        "REMAINING_QUALIFICATION_QUEUE.json",
    ]
    for name in required:
        if not (root / name).is_file():
            return fail("STRUCTURAL_INCONSISTENCY", f"missing {name}")

    binding_disk = json.loads((root / "INVENTORY_BINDING.json").read_text(encoding="utf-8"))
    adversarial = "--adversarial" in sys.argv
    expected_binding = binding_disk if adversarial else inventory_binding(repo_root)
    if binding_disk.get("obligation_ids") != expected_binding["obligation_ids"]:
        return fail("INVENTORY_BINDING_MISMATCH", "obligation id set")
    if len(binding_disk.get("obligation_ids", [])) != 34:
        return fail("MISSING_OBLIGATION_ID", "count!=34")
    if len(set(binding_disk.get("obligation_ids", []))) != 34:
        return fail("DUPLICATE_OBLIGATION_ID", "duplicate obligation")

    obl = json.loads((root / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    queue = json.loads((root / "REMAINING_QUALIFICATION_QUEUE.json").read_text(encoding="utf-8"))
    baseline = json.loads((root / "BASELINE_BINDING.json").read_text(encoding="utf-8"))
    source_sha = baseline.get("qualification_source_sha") or obl.get("source_sha")
    if obl.get("source_sha") and obl.get("source_sha") != source_sha:
        return fail("SOURCE_SHA_MISMATCH", "obligation matrix source_sha")
    tree_sha = baseline.get("qualification_source_tree_sha")
    qual_sha = baseline.get("qualification_source_sha") or source_sha
    if tree_sha and qual_sha:
        pinned = git(repo_root, ["rev-parse", f"{qual_sha}^{{tree}}"])
        if tree_sha != pinned:
            return fail("SOURCE_TREE_MISMATCH", "baseline tree vs qualification sha")
    obl_ids = [o["obligation_id"] for o in obl.get("obligations", [])]
    if len(set(obl_ids)) != len(obl_ids):
        return fail("DUPLICATE_OBLIGATION_ID", "matrix duplicate")
    if len(obl_ids) != 34:
        return fail("MISSING_OBLIGATION_ID", f"matrix count {len(obl_ids)}")

    qual_count = len(queue.get("qualification_queue", []))
    host_count = len(queue.get("application_host_queue", []))
    if queue.get("queue_count") and queue.get("queue_count") != qual_count + host_count:
        return fail("REPLACEMENT_QUALIFICATION_PATH", "queue_count mismatch")
    if qual_count + host_count != 31:
        return fail("REPLACEMENT_QUALIFICATION_PATH", "inventory queue path count!=31")

    ec08_obl = json.loads((ec08_dir(repo_root) / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    ec08_ids = {o["obligation_id"] for o in ec08_obl["obligations"]}
    if set(obl_ids) != ec08_ids:
        return fail("REPLACEMENT_OBLIGATION_ID", "id set drift from inventory")
    forged = [o for o in obl_ids if o not in ec08_ids]
    if forged:
        return fail("REPLACEMENT_OBLIGATION_ID", forged[0])

    file_outcomes: dict[str, list] = {}
    rec_dir = root / "execution_records"
    if rec_dir.is_dir():
        for post_path in rec_dir.glob("*.post.json"):
            batch_id = post_path.name.replace(".post.json", "")
            if final_mode and batch_id.startswith("synthetic_"):
                continue
            post = json.loads(post_path.read_text(encoding="utf-8"))
            pre_path = rec_dir / post_path.name.replace(".post.json", ".pre.json")
            pre = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.is_file() else {}
            pre_sha = pre.get("source_commit_sha")
            post_sha_after = post.get("source_commit_sha_after")
            pre_tree = pre.get("source_tree_sha")
            post_tree_after = post.get("source_tree_sha_after")
            if pre_sha and post_sha_after and pre_sha != post_sha_after:
                return fail("SOURCE_CHANGED_DURING_EXECUTION", post_path.name)
            if (
                pre_sha
                and post_sha_after
                and pre_sha == post_sha_after
                and pre_tree
                and post_tree_after
                and pre_tree != post_tree_after
            ):
                return fail("SOURCE_TREE_MISMATCH", post_path.name)
            if post.get("finished_at_utc", "") < pre.get("started_at_utc", ""):
                return fail("INVALID_TIMESTAMPS", post_path.name)
            rc = post.get("process_returncode")
            disp = post.get("process_exit_disposition") or "PROCESS_COMPLETED"
            if disp == "PROCESS_EXIT_UNRESOLVED" or rc is None:
                return fail("PROCESS_EXIT_UNRESOLVED", post_path.name)
            if disp == "EXECUTION_TIMEOUT":
                return fail("INCOMPLETE_EXECUTION_RECORD", post_path.name)
            log_rel = post.get("stdout_stderr_path")
            junit_rel = post.get("junit_path")
            if log_rel and not (root / log_rel).is_file():
                return fail("MISSING_RAW_LOG", log_rel)
            if log_rel and (root / log_rel).is_file() and post.get("process_returncode") == 0:
                log_body = (root / log_rel).read_text(encoding="utf-8", errors="replace")
                if "FAILED" in log_body or "= FAIL" in log_body:
                    return fail("NONZERO_EXIT_AS_SUCCESS", post_path.name)
            if post.get("process_returncode") is not None and post.get("process_returncode") == 0:
                if junit_rel and not (root / junit_rel).is_file():
                    return fail("MISSING_RAW_JUNIT", junit_rel)
            if log_rel:
                digest = post.get("stdout_stderr_sha256")
                if digest and sha256_file(root / log_rel) != digest:
                    return fail("RAW_DIGEST_MISMATCH", log_rel)
            if junit_rel and (root / junit_rel).is_file():
                if post.get("junit_sha256") and sha256_file(root / junit_rel) != post.get("junit_sha256"):
                    return fail("RAW_DIGEST_MISMATCH", junit_rel)
                batch_id = post_path.name.replace(".post.json", "")
                for case in merge_junit_with_node_capture(
                    root / junit_rel,
                    batch_id=batch_id,
                    source_sha=pre.get("source_commit_sha", source_sha),
                    run_root=root,
                ):
                    file_outcomes.setdefault(case.file_path, []).append(case)
    testcase_doc = (
        json.loads((root / "TESTCASE_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
        if (root / "TESTCASE_EXECUTION_MATRIX.json").is_file()
        else {"testcases": []}
    )
    seen_nodes: dict[str, int] = {}
    for tc in testcase_doc.get("testcases", []):
        if tc.get("evidence_validity") == "HISTORICAL_UNVERIFIED" and tc.get("outcome") == "PASS":
            for row in queue.get("qualification_queue", []):
                if row.get("recalculated_disposition") == "GREEN" and row.get("path") == tc.get("file_path"):
                    return fail("HISTORICAL_AS_CURRENT_PASS", tc.get("node_id", ""))
        if tc.get("outcome") == "XFAIL" and tc.get("counted_as") == "PASS":
            return fail("XFAIL_AS_PASS", tc.get("node_id", ""))
        node = tc.get("node_id", "")
        if node:
            seen_nodes[node] = seen_nodes.get(node, 0) + 1
            if seen_nodes[node] > 1 and tc.get("counted_in_coverage") is not False:
                return fail("DUPLICATE_TESTCASE_COVERAGE", node)
        if tc.get("outcome") == "COLLECTION_ERROR" and tc.get("disposition") == "NOT_EXECUTED":
            return fail("COLLECTION_ERROR_MISREPRESENTED", tc.get("node_id", ""))

    reverse = build_test_path_reverse_index(ec08_obl)
    for path, cases in file_outcomes.items():
        if path not in reverse and not any(
            b.get("paths") and path in b.get("paths", [])
            for b in json.loads((root / "DES_PREREG.json").read_text(encoding="utf-8")).get("batches", [])
        ):
            batch_class = None
            prereg = json.loads((root / "DES_PREREG.json").read_text(encoding="utf-8"))
            for batch in prereg.get("batches", []):
                if path in batch.get("paths", []):
                    batch_class = batch.get("classification")
            if batch_class != "SUPPLEMENTARY_REGRESSION":
                return fail("UNMAPPED_TESTCASE", path)

    ec08_order = [o["obligation_id"] for o in ec08_obl["obligations"]]
    if obl_ids != ec08_order:
        return fail("INCORRECT_OBLIGATION_MAPPING", "obligation order drift")
    for ob_row, ec08_ob in zip(obl.get("obligations", []), ec08_obl["obligations"], strict=False):
        if ob_row.get("obligation_id") != ec08_ob.get("obligation_id"):
            return fail("INCORRECT_OBLIGATION_MAPPING", ob_row.get("obligation_id"))
        expected_disp, _ = compute_obligation_disposition(ec08_ob, file_outcomes=file_outcomes, source_sha=source_sha)
        actual = ob_row.get("recalculated_disposition")
        if actual in ("GREEN", "QUALIFICATION_COMPLETE") and expected_disp in ("RED", "INVALIDATED"):
            return fail("GREEN_DESPITE_FAILURE", ob_row.get("obligation_id"))
        if actual in ("GREEN", "QUALIFICATION_COMPLETE") and expected_disp == "PARTIAL":
            return fail("GREEN_MISSING_TESTCASE", ob_row.get("obligation_id"))
        if ob_row.get("recalculated_disposition") == "QUALIFICATION_COMPLETE" and ec08_ob.get(
            "semantic_review_required"
        ):
            return fail("PREREQUISITE_FALSE_SATISFIED", ob_row.get("obligation_id"))
        if actual in ("GREEN", "QUALIFICATION_COMPLETE") and expected_disp in (
            "OPEN",
            "DOCUMENTARY_EVIDENCE_ONLY",
            "PARTIAL",
        ):
            return fail("STALE_DISPOSITION", ob_row.get("obligation_id"))
        if (
            actual == ec08_ob.get("execution_disposition")
            and actual in ("GREEN", "QUALIFICATION_COMPLETE")
            and expected_disp in ("OPEN", "RED", "PARTIAL", "QUALIFICATION_PENDING")
        ):
            return fail("STALE_DISPOSITION", ob_row.get("obligation_id"))
        if ob_row.get("evidence_reference") == "unrelated_batch":
            return fail("EVIDENCE_REFERENCE_MISMATCH", ob_row.get("obligation_id"))
        if ob_row.get("obligation_id") not in ec08_ids:
            return fail("INCORRECT_OBLIGATION_MAPPING", ob_row.get("obligation_id"))

    for row in queue.get("qualification_queue", []):
        path = row.get("path")
        cases = file_outcomes.get(path, [])
        disp = row.get("recalculated_disposition")
        if disp == "GREEN":
            for outcome in row.get("testcase_outcomes", []):
                if outcome.get("outcome") == "SKIP":
                    return fail("SKIP_WITHOUT_DISPOSITION", row.get("path", ""))
                if outcome.get("evidence_validity") == "HISTORICAL_UNVERIFIED":
                    return fail("HISTORICAL_AS_CURRENT_PASS", row.get("path", ""))
        if disp == "GREEN" and not cases:
            return fail("GREEN_MISSING_TESTCASE", path)
        if cases and disp == "GREEN":
            for case in cases:
                if case.outcome in ("FAIL", "ERROR"):
                    return fail("GREEN_DESPITE_FAILURE", case.node_id)
                if case.outcome == "SKIP":
                    return fail("SKIP_WITHOUT_DISPOSITION", path)
                if getattr(case, "evidence_validity", None) == "HISTORICAL_UNVERIFIED":
                    return fail("HISTORICAL_AS_CURRENT_PASS", case.node_id)

    baseline = json.loads((root / "BASELINE_BINDING.json").read_text(encoding="utf-8"))
    qual_sha = baseline.get("qualification_source_sha") or source_sha
    qual_tree = baseline.get("qualification_source_tree_sha") or tree_sha
    adversarial = "--adversarial" in sys.argv

    if final_mode:
        for step, fn, kwargs in (
            ("synthetic", validate_synthetic_isolation, {}),
            ("inventory", validate_immutable_inventory, {"adversarial": adversarial}),
            ("queue", validate_queue_exact, {}),
            ("references", validate_evidence_references, {}),
            ("execution", validate_execution_inventory, {"qualification_tree": qual_tree}),
            ("dispositions", validate_disposition_reconciliation, {}),
            ("node_ids", validate_testcase_node_ids, {}),
        ):
            if fn == validate_immutable_inventory:
                rc = fn(root, repo_root, qual_sha, adversarial=adversarial, fail=fail)
            elif fn in (validate_queue_exact, validate_evidence_references):
                rc = (
                    fn(root, repo_root, qual_sha, fail=fail)
                    if fn == validate_queue_exact
                    else fn(root, repo_root, fail=fail)
                )
            elif fn == validate_execution_inventory:
                rc = fn(root, repo_root, qualification_sha=qual_sha, qualification_tree=qual_tree, fail=fail)
            elif fn == validate_disposition_reconciliation:
                rc = fn(root, repo_root, qual_sha, fail=fail)
            elif fn == validate_testcase_node_ids:
                rc = fn(root, fail=fail)
            else:
                rc = fn(root, fail=fail)
            if rc:
                return rc

    print("VALIDATION_OK:ACCEPT")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--pre-check", action="store_true")
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("PARTIAL_VALIDATION", "FINAL_VALIDATION"),
        default=None,
        help="PARTIAL_VALIDATION for adversarial fixtures; FINAL_VALIDATION for campaign acceptance.",
    )
    args = parser.parse_args()
    mode = args.mode
    if mode is None:
        mode = "PARTIAL_VALIDATION" if args.adversarial else "FINAL_VALIDATION"
    if args.pre_check:
        required = ["DES_PREREG.json", "BASELINE_BINDING.json", "ENVIRONMENT_MANIFEST.json"]
        root = args.run_root
        for name in required:
            if not (root / name).is_file():
                print(f"PRE_CHECK_FAIL:missing {name}", file=sys.stderr)
                return 1
        env = json.loads((root / "ENVIRONMENT_MANIFEST.json").read_text(encoding="utf-8"))
        if not env.get("fastmcp_import_ok"):
            print("PRE_CHECK_FAIL:fastmcp", file=sys.stderr)
            return 1
        print("PRE_CHECK_OK")
        return 0
    return validate(args.run_root, mode=mode)


if __name__ == "__main__":
    raise SystemExit(main())
