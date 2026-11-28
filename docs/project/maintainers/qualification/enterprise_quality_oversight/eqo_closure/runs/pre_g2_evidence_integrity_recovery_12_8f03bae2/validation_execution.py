"""FINAL_VALIDATION execution inventory checks (EC-W3-12 V02–V06)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _evidence_common import (
    IMMUTABLE_INVENTORY_SHA,
    ec08_dir,
    git,
    inventory_binding_at_sha,
    sha256_file,
)

FailFn = Callable[[str, str], int]

MANDATORY_PRE_FIELDS = (
    "source_commit_sha",
    "source_tree_sha",
    "argv",
    "command_digest_sha256",
    "started_at_utc",
    "batch_id",
)
MANDATORY_POST_FIELDS = (
    "source_commit_sha_after",
    "source_tree_sha_after",
    "process_returncode",
    "process_exit_disposition",
    "finished_at_utc",
    "monotonic_elapsed_seconds",
    "stdout_stderr_path",
    "stdout_stderr_sha256",
    "junit_path",
    "junit_sha256",
)


def _load_batch_inventory(run_root: Path) -> dict[str, Any]:
    prereg = json.loads((run_root / "DES_PREREG.json").read_text(encoding="utf-8"))
    return {b["id"]: b for b in prereg.get("batches", [])}


def validate_synthetic_isolation(run_root: Path, fail: FailFn) -> int:
    fixture_root = (run_root / "validator_fixtures").resolve()
    for post in run_root.glob("execution_records/*.post.json"):
        post_resolved = post.resolve()
        if fixture_root in post_resolved.parents:
            continue
        batch_id = post.name.replace(".post.json", "")
        if batch_id.startswith("synthetic_"):
            return fail("SYNTHETIC_IN_FINAL_QUALIFICATION", batch_id)
    return 0


def validate_execution_inventory(
    run_root: Path,
    repo_root: Path,
    *,
    qualification_sha: str,
    qualification_tree: str,
    fail: FailFn,
) -> int:
    batches = _load_batch_inventory(run_root)
    rec_dir = run_root / "execution_records"
    pre_files = {p.name.replace(".pre.json", ""): p for p in rec_dir.glob("*.pre.json")} if rec_dir.is_dir() else {}
    post_files = {p.name.replace(".post.json", ""): p for p in rec_dir.glob("*.post.json")} if rec_dir.is_dir() else {}
    if not pre_files and not post_files:
        return fail("EMPTY_EXECUTION_INVENTORY", "no execution_records")

    expected_tree = git(repo_root, ["rev-parse", f"{qualification_sha}^{{tree}}"])
    if expected_tree != qualification_tree:
        return fail("QUALIFICATION_SHA_MISMATCH", "baseline tree vs git tree")

    for batch_id, batch in batches.items():
        if batch.get("deferred"):
            continue
        has_pre = batch_id in pre_files
        has_post = batch_id in post_files
        if not has_pre and not has_post:
            return fail("MISSING_MANDATORY_BATCH", batch_id)
        if has_post and not has_pre:
            return fail("MISSING_PRE_RECORD", batch_id)
        if has_pre and not has_post:
            return fail("MISSING_POST_RECORD", batch_id)

    for batch_id in sorted(set(pre_files) | set(post_files)):
        if batch_id not in batches:
            if batch_id in post_files:
                return fail("STRUCTURAL_INCONSISTENCY", f"unregistered batch {batch_id}")
            continue
        pre = json.loads(pre_files[batch_id].read_text(encoding="utf-8"))
        post = json.loads(post_files[batch_id].read_text(encoding="utf-8"))
        if pre.get("batch_id") != batch_id or post.get("batch_id") != batch_id:
            return fail("STRUCTURAL_INCONSISTENCY", batch_id)
        for field in MANDATORY_PRE_FIELDS:
            if not pre.get(field):
                return fail("MISSING_SOURCE_BINDING_FIELDS", f"{batch_id}:pre:{field}")
        for field in MANDATORY_POST_FIELDS:
            if field not in post or post.get(field) is None:
                if field == "process_returncode" and post.get("process_exit_disposition") == "PROCESS_EXIT_UNRESOLVED":
                    continue
                return fail("MISSING_SOURCE_BINDING_FIELDS", f"{batch_id}:post:{field}")
        if post.get("process_exit_disposition") == "PROCESS_EXIT_UNRESOLVED":
            return fail("PROCESS_EXIT_UNRESOLVED", batch_id)
        if pre.get("source_commit_sha") != qualification_sha:
            return fail("QUALIFICATION_SHA_MISMATCH", f"{batch_id}:pre_sha")
        if pre.get("source_tree_sha") != qualification_tree:
            return fail("QUALIFICATION_SHA_MISMATCH", f"{batch_id}:pre_tree")
        if post.get("source_commit_sha_after") != qualification_sha:
            return fail("QUALIFICATION_SHA_MISMATCH", f"{batch_id}:post_sha")
        if post.get("source_tree_sha_after") != qualification_tree:
            return fail("QUALIFICATION_SHA_MISMATCH", f"{batch_id}:post_tree")
        pre_cons = pre.get("tracked_source_consistency") or {}
        if pre_cons.get("porcelain_empty") is not True:
            return fail("STRUCTURAL_INCONSISTENCY", f"{batch_id}:dirty_source")
        log_rel = post.get("stdout_stderr_path")
        junit_rel = post.get("junit_path")
        if log_rel and not (run_root / log_rel).is_file():
            return fail("MISSING_RAW_LOG", batch_id)
        if junit_rel and not (run_root / junit_rel).is_file():
            return fail("MISSING_RAW_JUNIT", batch_id)
        if (
            log_rel
            and post.get("stdout_stderr_sha256")
            and sha256_file(run_root / log_rel) != post["stdout_stderr_sha256"]
        ):
            return fail("RAW_DIGEST_MISMATCH", batch_id)
        if junit_rel and post.get("junit_sha256") and sha256_file(run_root / junit_rel) != post["junit_sha256"]:
            return fail("RAW_DIGEST_MISMATCH", batch_id)
        rc = post.get("process_returncode")
        if rc not in (0, 1, 2, 3, 4, 5):
            return fail("PROCESS_EXIT_UNRESOLVED", batch_id)

    return 0


def validate_queue_exact(run_root: Path, repo_root: Path, qualification_sha: str, fail: FailFn) -> int:
    binding = inventory_binding_at_sha(repo_root, qualification_sha)
    queue_disk = json.loads((run_root / "REMAINING_QUALIFICATION_QUEUE.json").read_text(encoding="utf-8"))
    qual_paths = [q["path"] for q in queue_disk.get("qualification_queue", [])]
    host_paths = [q["path"] for q in queue_disk.get("application_host_queue", [])]
    if len(qual_paths) != 29:
        return fail("REPLACEMENT_QUALIFICATION_PATH", f"qual count={len(qual_paths)}")
    if len(host_paths) != 2:
        return fail("REPLACEMENT_QUALIFICATION_PATH", f"host count={len(host_paths)}")
    if len(set(qual_paths)) != 29:
        return fail("DUPLICATE_QUEUE_PATH", "qualification_queue")
    if len(set(host_paths)) != 2:
        return fail("DUPLICATE_QUEUE_PATH", "application_host_queue")
    if sorted(qual_paths) != binding["qualification_paths"]:
        return fail("REPLACEMENT_QUALIFICATION_PATH", "qual paths drift")
    if sorted(host_paths) != binding["application_host_paths"]:
        return fail("REPLACEMENT_QUALIFICATION_PATH", "host paths drift")
    return 0


def validate_immutable_inventory(
    run_root: Path, repo_root: Path, qualification_sha: str, *, adversarial: bool, fail: FailFn
) -> int:
    if adversarial:
        return 0
    live = inventory_binding_at_sha(repo_root, IMMUTABLE_INVENTORY_SHA)
    disk = json.loads((run_root / "INVENTORY_BINDING.json").read_text(encoding="utf-8"))
    if disk.get("inventory_source") != live["inventory_source"]:
        return fail("INVENTORY_BINDING_MISMATCH", "source")
    for name, live_art in live["artifacts"].items():
        disk_art = (disk.get("artifacts") or {}).get(name)
        if not disk_art:
            return fail("INVENTORY_BINDING_MISMATCH", name)
        if disk_art.get("git_blob_sha") != live_art.get("git_blob_sha"):
            return fail("INVENTORY_BINDING_MISMATCH", f"{name}:blob")
        if disk_art.get("content_digest_sha256") != live_art.get("content_digest_sha256"):
            return fail("INVENTORY_BINDING_MISMATCH", f"{name}:digest")
    if len(live.get("obligation_ids", [])) != 34:
        return fail("MISSING_OBLIGATION_ID", "inventory obligations")
    if live.get("documentary_obligation_count") != 12:
        return fail("STRUCTURAL_INCONSISTENCY", "documentary count")
    if len(live.get("support_module_paths", [])) != 4:
        return fail("STRUCTURAL_INCONSISTENCY", "support modules")
    return 0


def _execution_batch_from_ref(ref: str) -> str | None:
    prefix = "execution_records/"
    suffix = ".post.json"
    if ref.startswith(prefix) and ref.endswith(suffix):
        return ref[len(prefix) : -len(suffix)]
    return None


def validate_evidence_references(run_root: Path, repo_root: Path, *, fail: FailFn) -> int:
    obl = json.loads((run_root / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    rec_dir = run_root / "execution_records"
    posts = {p.name.replace(".post.json", "") for p in rec_dir.glob("*.post.json")} if rec_dir.is_dir() else set()
    for ob in obl.get("obligations", []):
        disp = ob.get("recalculated_disposition")
        refs: list[str] = []
        if isinstance(ob.get("evidence_references"), list):
            refs.extend(str(x) for x in ob["evidence_references"])
        ref = ob.get("evidence_reference")
        if ref:
            refs.append(str(ref))
        if not refs:
            if disp in ("OPEN", "NOT_EXECUTED", "DEFERRED") and ob.get("deferral_reason_code"):
                continue
            if disp == "DOCUMENTARY_EVIDENCE_ONLY" or (
                ob.get("historical_disposition_reference") == "DOCUMENTARY_EVIDENCE_ONLY"
            ):
                continue
            continue
        for one in refs:
            if one.startswith("..") or one.startswith("/"):
                return fail("INVALID_EVIDENCE_REFERENCE", ob.get("obligation_id", ""))
            batch_id = _execution_batch_from_ref(one)
            if batch_id is None:
                if (run_root / one).is_file():
                    continue
                if disp in (
                    "OPEN",
                    "RED",
                    "PARTIAL",
                    "DEFERRED",
                    "NOT_EXECUTED",
                    "DOCUMENTARY_EVIDENCE_ONLY",
                ) and not one.startswith("execution_records/"):
                    continue
                if one.startswith("execution_records/"):
                    return fail("INVALID_EVIDENCE_REFERENCE", ob.get("obligation_id", ""))
                continue
            if batch_id not in posts:
                return fail("INVALID_EVIDENCE_REFERENCE", ob.get("obligation_id", ""))
    return 0


def validate_disposition_reconciliation(
    run_root: Path,
    repo_root: Path,
    qualification_sha: str,
    *,
    fail: FailFn,
) -> int:
    from _evidence_common import (
        compute_obligation_disposition,
        compute_queue_row_disposition,
        merge_junit_with_node_capture,
    )

    obl = json.loads((run_root / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    ec08 = json.loads((ec08_dir(repo_root) / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    file_outcomes: dict[str, list] = {}
    rec_dir = run_root / "execution_records"
    for post_path in rec_dir.glob("*.post.json"):
        batch_id = post_path.name.replace(".post.json", "")
        pre_path = rec_dir / f"{batch_id}.pre.json"
        if not pre_path.is_file():
            continue
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        post = json.loads(post_path.read_text(encoding="utf-8"))
        junit_rel = post.get("junit_path")
        if not junit_rel or not (run_root / junit_rel).is_file():
            continue
        for case in merge_junit_with_node_capture(
            run_root / junit_rel,
            batch_id=batch_id,
            source_sha=pre.get("source_commit_sha", qualification_sha),
            run_root=run_root,
        ):
            file_outcomes.setdefault(case.file_path, []).append(case)
    queue = json.loads((run_root / "REMAINING_QUALIFICATION_QUEUE.json").read_text(encoding="utf-8"))
    for row in queue.get("application_host_queue", []):
        path = row.get("path", "")
        disp = row.get("recalculated_disposition")
        cases = file_outcomes.get(path, [])
        expected, _ = compute_queue_row_disposition(path, cases)
        if disp != expected:
            if disp == "GREEN" and expected != "GREEN":
                return fail("UNSUPPORTED_GREEN_HOST_QUEUE", path)
            return fail("INCORRECT_DISPOSITION_GREEN", path)
    for row in queue.get("qualification_queue", []):
        path = row.get("path", "")
        disp = row.get("recalculated_disposition")
        cases = file_outcomes.get(path, [])
        expected, _ = compute_queue_row_disposition(path, cases)
        if disp != expected:
            return fail("INCORRECT_DISPOSITION_GREEN", path or row.get("obligation_id", ""))
    for row, ec08_ob in zip(obl.get("obligations", []), ec08.get("obligations", []), strict=False):
        expected, _ = compute_obligation_disposition(ec08_ob, file_outcomes=file_outcomes, source_sha=qualification_sha)
        actual = row.get("recalculated_disposition")
        if actual != expected:
            return fail("STALE_DISPOSITION", row.get("obligation_id", ""))
    return 0


def validate_testcase_node_ids(run_root: Path, *, fail: FailFn) -> int:
    from _evidence_common import merge_junit_with_node_capture

    tc_doc = json.loads((run_root / "TESTCASE_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    junit_cases: dict[str, set[str]] = {}
    rec_dir = run_root / "execution_records"
    for post_path in rec_dir.glob("*.post.json"):
        batch_id = post_path.name.replace(".post.json", "")
        pre_path = rec_dir / f"{batch_id}.pre.json"
        if not pre_path.is_file():
            continue
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        post = json.loads(post_path.read_text(encoding="utf-8"))
        junit_rel = post.get("junit_path")
        if not junit_rel:
            continue
        for case in merge_junit_with_node_capture(
            run_root / junit_rel,
            batch_id=batch_id,
            source_sha=pre.get("source_commit_sha", ""),
            run_root=run_root,
        ):
            junit_cases.setdefault(case.file_path, set()).add(case.node_id)
    for tc in tc_doc.get("testcases", []):
        path = tc.get("file_path")
        nid = tc.get("node_id")
        if path in junit_cases and nid not in junit_cases[path]:
            return fail("TESTCASE_NODE_ID_MISMATCH", nid or path)
    return 0
