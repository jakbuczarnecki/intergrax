"""Shared EQO EC-W3-12 semantic evidence primitives (read-only helpers)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

OutcomeKind = Literal[
    "PASS",
    "FAIL",
    "ERROR",
    "SKIP",
    "XFAIL",
    "XPASS",
    "NOT_EXECUTED",
    "COLLECTION_ERROR",
    "PROCESS_EXIT_UNRESOLVED",
    "INVALIDATED",
    "DEFERRED",
]

VALIDATOR_REASONS = (
    "ACCEPT",
    "MISSING_OBLIGATION_ID",
    "DUPLICATE_OBLIGATION_ID",
    "REPLACEMENT_OBLIGATION_ID",
    "REPLACEMENT_QUALIFICATION_PATH",
    "GREEN_DESPITE_FAILURE",
    "GREEN_MISSING_TESTCASE",
    "MISSING_RAW_LOG",
    "MISSING_RAW_JUNIT",
    "RAW_DIGEST_MISMATCH",
    "SOURCE_SHA_MISMATCH",
    "SOURCE_TREE_MISMATCH",
    "SOURCE_CHANGED_DURING_EXECUTION",
    "INVALID_TIMESTAMPS",
    "NONZERO_EXIT_AS_SUCCESS",
    "UNMAPPED_TESTCASE",
    "DUPLICATE_TESTCASE_COVERAGE",
    "STALE_DISPOSITION",
    "INCORRECT_OBLIGATION_MAPPING",
    "XFAIL_AS_PASS",
    "COLLECTION_ERROR_MISREPRESENTED",
    "HISTORICAL_AS_CURRENT_PASS",
    "VALIDATOR_MUTATED_INPUT",
    "PREREQUISITE_FALSE_SATISFIED",
    "SKIP_WITHOUT_DISPOSITION",
    "EVIDENCE_REFERENCE_MISMATCH",
    "INVENTORY_BINDING_MISMATCH",
    "STRUCTURAL_INCONSISTENCY",
    "EMPTY_EXECUTION_INVENTORY",
    "MISSING_MANDATORY_BATCH",
    "MISSING_PRE_RECORD",
    "MISSING_POST_RECORD",
    "MISSING_SOURCE_BINDING_FIELDS",
    "QUALIFICATION_SHA_MISMATCH",
    "DUPLICATE_QUEUE_PATH",
    "UNSUPPORTED_GREEN_HOST_QUEUE",
    "INCORRECT_DISPOSITION_GREEN",
    "INVALID_EVIDENCE_REFERENCE",
    "SYNTHETIC_IN_FINAL_QUALIFICATION",
    "TESTCASE_NODE_ID_MISMATCH",
    "PROCESS_EXIT_UNRESOLVED",
    "INCOMPLETE_EXECUTION_RECORD",
)


def repo_root_from_run(run_root: Path) -> Path:
    candidate = run_root.resolve()
    for _ in range(16):
        if (candidate / ".git").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return run_root.parents[7]


EC08_REL = (
    "docs/project/maintainers/qualification/enterprise_quality_oversight/"
    "eqo_closure/runs/pre_g2_qualification_execution_08_245d4717"
)
IMMUTABLE_INVENTORY_SHA = "4abc43faad149cc0ce7291468e07886efcdd1f13"
BATCH_TIMEOUT_SECONDS = 3600

Q04_LIGHTWEIGHT_TESTS = (
    "test_l3_plain_freeze_blocked_without_baseline",
    "test_l3_full_real_85_fail_on_check_filter",
    "test_l3_full_not_qualifying_without_opt_in_profile",
)


@dataclass(frozen=True)
class ExecutionContext:
    source_root: Path
    run_root: Path
    qualification_sha: str
    repo_root: Path

    @property
    def qualification_tree_sha(self) -> str:
        return git(self.repo_root, ["rev-parse", f"{self.qualification_sha}^{{tree}}"])


def resolve_execution_context(
    *,
    source_root: Path | None,
    run_root: Path | None,
    qualification_sha: str | None,
) -> ExecutionContext:
    resolved_run = (run_root or Path(__file__).resolve().parent).resolve()
    resolved_repo = repo_root_from_run(resolved_run)
    if source_root is None:
        if not qualification_sha:
            raise ValueError("source_root or qualification_sha required")
        source_root = resolved_repo
    resolved_source = source_root.resolve()
    if qualification_sha is None:
        qualification_sha = git(resolved_source, ["rev-parse", "HEAD"])
    return ExecutionContext(
        source_root=resolved_source,
        run_root=resolved_run,
        qualification_sha=qualification_sha,
        repo_root=resolved_repo,
    )


def ec08_dir(repo_root: Path) -> Path:
    return repo_root / EC08_REL


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def git(repo_root: Path, args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def inventory_binding_at_sha(repo_root: Path, sha: str) -> dict[str, Any]:
    """EC-W3-08 inventory pinned to qualification source commit (not moving HEAD)."""
    ec08_rel = EC08_REL
    files_rel = {
        "OBLIGATION_EXECUTION_MATRIX.json": f"{ec08_rel}/OBLIGATION_EXECUTION_MATRIX.json",
        "REMAINING_QUALIFICATION_QUEUE.json": f"{ec08_rel}/REMAINING_QUALIFICATION_QUEUE.json",
        "DES_PREREG.json": f"{ec08_rel}/DES_PREREG.json",
    }
    out: dict[str, Any] = {
        "schema": "eqo.inventory_binding.v1",
        "inventory_source": ec08_rel,
        "qualification_source_sha": sha,
        "artifacts": {},
    }
    payloads: dict[str, Any] = {}
    for name, rel in files_rel.items():
        blob = git(repo_root, ["rev-parse", f"{sha}:{rel}"])
        raw_bytes = subprocess.check_output(["git", "-C", str(repo_root), "cat-file", "blob", blob])
        raw = raw_bytes.decode("utf-8")
        payload = json.loads(raw)
        payloads[name] = payload
        out["artifacts"][name] = {
            "path": rel,
            "git_blob_sha": blob,
            "content_digest_sha256": sha256_bytes(raw_bytes),
            "json_canonical_digest_sha256": sha256_json_canonical(payload),
        }
    obl = payloads["OBLIGATION_EXECUTION_MATRIX.json"]
    queue = payloads["REMAINING_QUALIFICATION_QUEUE.json"]
    out["obligation_ids"] = sorted(o["obligation_id"] for o in obl["obligations"])
    out["qualification_paths"] = sorted(q["path"] for q in queue.get("qualification_queue", []))
    out["application_host_paths"] = sorted(q["path"] for q in queue.get("application_host_queue", []))
    out["support_module_paths"] = sorted({m for o in obl["obligations"] for m in o.get("support_modules", []) if m})
    out["documentary_obligation_count"] = sum(1 for o in obl["obligations"] if not o.get("mapped_test_file_paths"))
    return out


def inventory_binding(repo_root: Path) -> dict[str, Any]:
    ec08 = ec08_dir(repo_root)
    files = {
        "OBLIGATION_EXECUTION_MATRIX.json": ec08 / "OBLIGATION_EXECUTION_MATRIX.json",
        "REMAINING_QUALIFICATION_QUEUE.json": ec08 / "REMAINING_QUALIFICATION_QUEUE.json",
        "DES_PREREG.json": ec08 / "DES_PREREG.json",
    }
    out: dict[str, Any] = {
        "schema": "eqo.inventory_binding.v1",
        "inventory_source": EC08_REL,
        "artifacts": {},
    }
    for name, path in files.items():
        blob = git(repo_root, ["rev-parse", f"HEAD:{path.relative_to(repo_root).as_posix()}"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        out["artifacts"][name] = {
            "path": path.relative_to(repo_root).as_posix(),
            "git_blob_sha": blob,
            "content_digest_sha256": sha256_file(path),
            "json_canonical_digest_sha256": sha256_json_canonical(payload),
        }
    obl = json.loads(files["OBLIGATION_EXECUTION_MATRIX.json"].read_text(encoding="utf-8"))
    queue = json.loads(files["REMAINING_QUALIFICATION_QUEUE.json"].read_text(encoding="utf-8"))
    out["obligation_ids"] = sorted(o["obligation_id"] for o in obl["obligations"])
    out["qualification_paths"] = sorted(
        q["path"] for q in queue.get("qualification_queue", []) if q.get("category") != "application_host"
    )
    out["application_host_paths"] = sorted(q["path"] for q in queue.get("application_host_queue", []))
    out["support_module_paths"] = sorted({m for o in obl["obligations"] for m in o.get("support_modules", []) if m})
    out["documentary_obligation_count"] = sum(1 for o in obl["obligations"] if not o.get("mapped_test_file_paths"))
    return out


def build_test_path_reverse_index(obligation_matrix: dict[str, Any]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = {}
    for ob in obligation_matrix["obligations"]:
        oid = ob["obligation_id"]
        for key in (
            "mapped_test_file_paths",
            "remaining_qualification_test_paths",
            "remaining_application_host_test_paths",
        ):
            for path in ob.get(key, []) or []:
                reverse.setdefault(path, set()).add(oid)
    return reverse


@dataclass(frozen=True)
class TestcaseRecord:
    node_id: str
    file_path: str
    name: str
    outcome: OutcomeKind
    batch_id: str
    source_commit_sha: str


def _classify_testcase(case: ET.Element) -> OutcomeKind:
    skipped = case.find("skipped")
    if skipped is not None:
        msg = (skipped.get("message") or "") + (skipped.text or "")
        if "xfail" in msg.lower() or skipped.get("type") == "pytest.xfail":
            return "XFAIL"
        return "SKIP"
    if case.find("failure") is not None:
        failure = case.find("failure")
        msg = (failure.get("message") or "") + (failure.text or "")
        if "xfail" in msg.lower():
            return "XFAIL"
        return "FAIL"
    if case.find("error") is not None:
        err_text = (case.find("error").text or "") + (case.find("error").get("message") or "")
        if "collection" in err_text.lower():
            return "COLLECTION_ERROR"
        return "ERROR"
    return "PASS"


def _junit_properties(case: ET.Element) -> dict[str, str]:
    props: dict[str, str] = {}
    props_el = case.find("properties")
    if props_el is None:
        return props
    for prop in props_el.findall("property"):
        name = prop.get("name")
        value = prop.get("value")
        if name and value is not None:
            props[name] = value
    return props


def parse_junit_cases(path: Path, *, batch_id: str, source_sha: str) -> list[TestcaseRecord]:
    if not path.is_file():
        return []
    root = ET.parse(path).getroot()
    records: list[TestcaseRecord] = []
    for case in root.iter("testcase"):
        file_attr = case.get("file")
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        props = _junit_properties(case)
        if file_attr:
            file_path = file_attr.replace("\\", "/")
            if not file_path.startswith("tests/") and not file_path.startswith("applications/"):
                file_path = f"tests/{file_path}" if "." in file_path else file_path
        elif props.get("eqo_file_path"):
            file_path = props["eqo_file_path"]
        else:
            file_path = classname.replace(".", "/") + ".py" if classname else "unknown"
        node_id = props.get("eqo_node_id") or (f"{classname}::{name}" if classname else name)
        records.append(
            TestcaseRecord(
                node_id=node_id,
                file_path=file_path,
                name=name,
                outcome=_classify_testcase(case),
                batch_id=batch_id,
                source_commit_sha=source_sha,
            )
        )
    return records


def extract_pytest_paths_from_command(command: str) -> list[str]:
    m = re.search(r"pytest\s+(.+?)\s+-vv", command)
    if not m:
        return []
    return m.group(1).split()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_obligation_disposition(
    ob: dict[str, Any],
    *,
    file_outcomes: dict[str, list[TestcaseRecord]],
    source_sha: str,
) -> tuple[str, list[str]]:
    """Return (disposition, limitation_notes) without copying historical execution_disposition."""
    limitations: list[str] = []
    required_paths = list(ob.get("mapped_test_file_paths") or [])
    qual_paths = list(ob.get("remaining_qualification_test_paths") or [])
    host_paths = list(ob.get("remaining_application_host_test_paths") or [])
    all_paths = {p for p in required_paths + qual_paths + host_paths if p}
    if not all_paths and ob.get("documentary_evidence_paths"):
        return "DOCUMENTARY_EVIDENCE_ONLY", limitations
    if ob.get("semantic_review_required"):
        limitations.append("semantic_review_open")
    outcomes: list[TestcaseRecord] = []
    for path in all_paths:
        outcomes.extend(file_outcomes.get(path, []))
    if not outcomes:
        if ob.get("execution_disposition") in ("DEFERRED", "DEFERRED_FORMAL_QUALIFICATION_PROFILE"):
            return "DEFERRED", limitations
        return "OPEN", limitations
    for rec in outcomes:
        if rec.source_commit_sha != source_sha:
            return "INVALIDATED", limitations + ["source_binding_invalid"]
    if any(o.outcome in ("FAIL", "ERROR", "COLLECTION_ERROR") for o in outcomes):
        return "RED", limitations
    executed = {o.file_path for o in outcomes}
    if all_paths - executed:
        if ob.get("semantic_review_required"):
            return "GREEN_WITH_LIMITS", limitations + ["partial_executable_coverage"]
        return "PARTIAL", limitations + ["incomplete_path_coverage"]
    if any(o.outcome in ("SKIP", "NOT_EXECUTED", "DEFERRED") for o in outcomes):
        return "QUALIFICATION_PENDING", limitations
    if limitations:
        return "GREEN_WITH_LIMITS", limitations
    return "GREEN", limitations


def load_node_capture(run_root: Path, batch_id: str) -> dict[str, Any] | None:
    path = run_root / "node_reports" / f"{batch_id}_nodes.json"
    if not path.is_file():
        path = run_root / "logs" / f"{batch_id}_nodes.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def testcase_identity(batch_id: str, node_id: str, source_sha: str) -> tuple[str, str, str]:
    return (batch_id, node_id, source_sha)


def _outcome_rank(outcome: str) -> int:
    order = {
        "ERROR": 6,
        "FAIL": 5,
        "COLLECTION_ERROR": 5,
        "XPASS": 4,
        "XFAIL": 3,
        "SKIP": 2,
        "PASS": 1,
        "NOT_EXECUTED": 0,
    }
    return order.get(outcome, 0)


def merge_junit_with_node_capture(
    junit_path: Path,
    *,
    batch_id: str,
    source_sha: str,
    run_root: Path,
) -> list[TestcaseRecord]:
    records = parse_junit_cases(junit_path, batch_id=batch_id, source_sha=source_sha)
    capture = load_node_capture(run_root, batch_id)
    if not capture:
        return records
    by_node: dict[str, dict[str, Any]] = {}
    for rec in capture.get("node_records", []):
        nid = rec.get("node_id")
        if nid:
            by_node[nid] = rec
    by_suffix: dict[tuple[str, str], list[str]] = {}
    for nid in by_node:
        if "::" not in nid:
            continue
        file_part, func = nid.rsplit("::", 1)
        by_suffix.setdefault((file_part.split("/")[-1], func), []).append(nid)
    merged: dict[tuple[str, str, str], TestcaseRecord] = {}
    for rec in records:
        cap = by_node.get(rec.node_id)
        if cap is None:
            suffix_key = (rec.file_path.rsplit("/", 1)[-1], rec.name)
            candidates = by_suffix.get((suffix_key[0].removesuffix(".py"), suffix_key[1]), [])
            if len(candidates) == 1:
                cap = by_node[candidates[0]]
                rec = TestcaseRecord(
                    node_id=candidates[0],
                    file_path=cap.get("file_path") or rec.file_path,
                    name=rec.name,
                    outcome=rec.outcome,
                    batch_id=rec.batch_id,
                    source_commit_sha=rec.source_commit_sha,
                )
            elif len(candidates) > 1:
                rec = TestcaseRecord(
                    node_id=rec.node_id,
                    file_path=rec.file_path,
                    name=rec.name,
                    outcome="ERROR",
                    batch_id=rec.batch_id,
                    source_commit_sha=rec.source_commit_sha,
                )
        key = testcase_identity(rec.batch_id, rec.node_id, rec.source_commit_sha)
        existing = merged.get(key)
        if existing is None or _outcome_rank(rec.outcome) >= _outcome_rank(existing.outcome):
            merged[key] = rec
    return list(merged.values())


def load_execution_record_pairs(run_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rec_dir = run_root / "execution_records"
    if not rec_dir.is_dir():
        return out
    for pre_path in rec_dir.glob("*.pre.json"):
        batch_id = pre_path.name.replace(".pre.json", "")
        post_path = rec_dir / f"{batch_id}.post.json"
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        post = json.loads(post_path.read_text(encoding="utf-8")) if post_path.is_file() else {}
        out[batch_id] = {"pre": pre, "post": post}
    return out


def build_file_outcomes_from_records(
    run_root: Path,
    records: dict[str, dict[str, Any]],
    *,
    default_source_sha: str,
) -> dict[str, list[TestcaseRecord]]:
    file_outcomes: dict[str, list[TestcaseRecord]] = {}
    for batch_id, pair in records.items():
        post = pair.get("post") or {}
        pre = pair.get("pre") or {}
        junit_rel = post.get("junit_path")
        if not junit_rel:
            continue
        junit_path = run_root / junit_rel
        if not junit_path.is_file():
            continue
        source_sha = pre.get("source_commit_sha") or default_source_sha
        for case in merge_junit_with_node_capture(
            junit_path, batch_id=batch_id, source_sha=source_sha, run_root=run_root
        ):
            file_outcomes.setdefault(case.file_path, []).append(case)
    return file_outcomes


def _obligation_evidence_refs(
    ob: dict[str, Any],
    *,
    batch_ids_for_paths: dict[str, set[str]],
) -> list[str]:
    paths = set(ob.get("mapped_test_file_paths") or []) | set(ob.get("remaining_qualification_test_paths") or [])
    paths |= set(ob.get("remaining_application_host_test_paths") or [])
    refs: list[str] = []
    seen: set[str] = set()
    for path in sorted(paths):
        for batch_id in sorted(batch_ids_for_paths.get(path, [])):
            ref = f"execution_records/{batch_id}.post.json"
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def compute_queue_row_disposition(path: str, cases: list[TestcaseRecord]) -> tuple[str, str]:
    if not cases:
        return "NOT_EXECUTED", "no_current_batch"
    if any(c.outcome in ("FAIL", "ERROR", "COLLECTION_ERROR") for c in cases):
        return "RED", "executable_failure"
    if any(c.outcome == "XPASS" for c in cases):
        return "QUALIFICATION_PENDING", "xpass_not_pass"
    if any(c.outcome == "XFAIL" for c in cases):
        return "QUALIFICATION_PENDING", "xfail_not_pass"
    if any(c.outcome == "SKIP" for c in cases):
        return "QUALIFICATION_PENDING", "skip_requires_disposition"
    if all(c.outcome == "PASS" for c in cases):
        return "GREEN", "all_pass"
    return "QUALIFICATION_PENDING", "mixed_or_skip"


def compute_canonical_evidence_projection(ctx: ExecutionContext) -> dict[str, Any]:
    """Deterministic read-only matrices from inventory @ immutable sha, DES, and execution records."""
    binding = inventory_binding_at_sha(ctx.repo_root, ctx.qualification_sha)
    ec08 = inventory_binding_at_sha(ctx.repo_root, IMMUTABLE_INVENTORY_SHA)
    obl_src = json.loads(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ctx.repo_root),
                "show",
                f"{IMMUTABLE_INVENTORY_SHA}:{EC08_REL}/OBLIGATION_EXECUTION_MATRIX.json",
            ],
            text=True,
        )
    )
    queue_src = json.loads(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ctx.repo_root),
                "show",
                f"{IMMUTABLE_INVENTORY_SHA}:{EC08_REL}/REMAINING_QUALIFICATION_QUEUE.json",
            ],
            text=True,
        )
    )
    source_sha = ctx.qualification_sha
    records = load_execution_record_pairs(ctx.run_root)
    file_outcomes = build_file_outcomes_from_records(ctx.run_root, records, default_source_sha=source_sha)
    batch_ids_for_paths: dict[str, set[str]] = {}
    testcase_matrix: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []
    prereg = json.loads((ctx.run_root / "DES_PREREG.json").read_text(encoding="utf-8"))
    batch_order = [b["id"] for b in prereg.get("batches", [])]
    ordered_batch_ids = [bid for bid in batch_order if bid in records] + sorted(set(records) - set(batch_order))
    seen_node_ids: dict[str, int] = {}
    for batch_id in ordered_batch_ids:
        pair = records[batch_id]
        post = pair.get("post") or {}
        pre = pair.get("pre") or {}
        junit_rel = post.get("junit_path")
        if junit_rel and (ctx.run_root / junit_rel).is_file():
            for case in merge_junit_with_node_capture(
                ctx.run_root / junit_rel,
                batch_id=batch_id,
                source_sha=pre.get("source_commit_sha", source_sha),
                run_root=ctx.run_root,
            ):
                batch_ids_for_paths.setdefault(case.file_path, set()).add(batch_id)
                prior = seen_node_ids.get(case.node_id, 0)
                seen_node_ids[case.node_id] = prior + 1
                testcase_matrix.append(
                    {
                        "batch_id": case.batch_id,
                        "node_id": case.node_id,
                        "file_path": case.file_path,
                        "outcome": case.outcome,
                        "source_commit_sha": case.source_commit_sha,
                        "evidence_reference": f"execution_records/{batch_id}.post.json",
                        "counted_in_coverage": prior == 0,
                    }
                )
        batch_manifest.append(
            {
                "batch_id": batch_id,
                "process_returncode": post.get("process_returncode"),
                "process_exit_disposition": post.get("process_exit_disposition"),
                "argv_digest": pre.get("command_digest_sha256"),
                "source_commit_sha": pre.get("source_commit_sha"),
                "junit_sha256": post.get("junit_sha256"),
                "log_sha256": post.get("stdout_stderr_sha256"),
            }
        )
    batch_rank = {bid: idx for idx, bid in enumerate(batch_order)}
    testcase_matrix.sort(key=lambda r: (batch_rank.get(r["batch_id"], 999), r["node_id"], r["source_commit_sha"]))
    file_rows: list[dict[str, Any]] = []
    for path in sorted(file_outcomes.keys()):
        cases = file_outcomes[path]
        refs = sorted(
            {f"execution_records/{c.batch_id}.post.json" for c in cases if c.batch_id},
        )
        file_rows.append(
            {
                "path": path,
                "testcase_count": len(cases),
                "outcomes": sorted({c.outcome for c in cases}),
                "evidence_references": refs,
            }
        )
    reverse = build_test_path_reverse_index(obl_src)
    obligation_rows: list[dict[str, Any]] = []
    for ob in obl_src["obligations"]:
        disposition, limits = compute_obligation_disposition(ob, file_outcomes=file_outcomes, source_sha=source_sha)
        refs = _obligation_evidence_refs(ob, batch_ids_for_paths=batch_ids_for_paths)
        row = {
            "obligation_id": ob["obligation_id"],
            "recalculated_disposition": disposition,
            "coverage_limitations": limits,
            "mapped_test_file_paths": ob.get("mapped_test_file_paths", []),
            "current_testcase_count": sum(len(file_outcomes.get(p, [])) for p in ob.get("mapped_test_file_paths", [])),
            "evidence_validity": "CURRENT",
            "historical_disposition_reference": ob.get("execution_disposition"),
            "semantic_review_required": ob.get("semantic_review_required", False),
        }
        if refs:
            row["evidence_references"] = refs
            row["evidence_reference"] = refs[0]
        obligation_rows.append(row)
    qual_rows: list[dict[str, Any]] = []
    for item in queue_src.get("qualification_queue", []):
        path = item["path"]
        cases = file_outcomes.get(path, [])
        disp, reason = compute_queue_row_disposition(path, cases)
        qual_rows.append(
            {
                **{k: item[k] for k in item if k not in ("current_result", "execution_disposition")},
                "recalculated_disposition": disp,
                "disposition_reason": reason,
                "source_commit_sha": source_sha,
                "testcase_outcomes": [c.__dict__ for c in cases],
                "obligation_ids": sorted(reverse.get(path, [])),
            }
        )
    host_rows: list[dict[str, Any]] = []
    for item in queue_src.get("application_host_queue", []):
        path = item["path"]
        cases = file_outcomes.get(path, [])
        disp, reason = compute_queue_row_disposition(path, cases)
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
    return {
        "INVENTORY_BINDING.json": binding,
        "TESTCASE_EXECUTION_MATRIX.json": {
            "schema": "eqo.testcase_execution_matrix.v1",
            "source_sha": source_sha,
            "testcases": testcase_matrix,
        },
        "FILE_LEVEL_EXECUTION_MATRIX.json": {
            "schema": "eqo.file_level_execution_matrix.v1",
            "source_sha": source_sha,
            "files": file_rows,
        },
        "BATCH_MANIFEST.json": {"schema": "eqo.batch_manifest.v1", "batches": batch_manifest},
        "OBLIGATION_EXECUTION_MATRIX.json": {
            "schema": "eqo.obligation_execution_matrix.v2",
            "source_sha": source_sha,
            "obligation_count": len(obligation_rows),
            "reconciled": True,
            "obligations": obligation_rows,
        },
        "REMAINING_QUALIFICATION_QUEUE.json": {
            "schema": "eqo.remaining_qualification_queue.v2",
            "source_sha": source_sha,
            "queue_count": len(qual_rows) + len(host_rows),
            "qualification_test_file_count": len(qual_rows),
            "application_host_test_file_count": len(host_rows),
            "qualification_queue": qual_rows,
            "application_host_queue": host_rows,
        },
        "_immutable_inventory_sha": IMMUTABLE_INVENTORY_SHA,
        "_ec08_binding_for_disposition": ec08,
    }
