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
        raw = git(repo_root, ["show", f"{sha}:{rel}"])
        payload = json.loads(raw)
        payloads[name] = payload
        out["artifacts"][name] = {
            "path": rel,
            "git_blob_sha": blob,
            "content_digest_sha256": sha256_bytes(raw.encode("utf-8")),
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


def parse_junit_cases(path: Path, *, batch_id: str, source_sha: str) -> list[TestcaseRecord]:
    if not path.is_file():
        return []
    root = ET.parse(path).getroot()
    records: list[TestcaseRecord] = []
    for case in root.iter("testcase"):
        file_attr = case.get("file")
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        if file_attr:
            file_path = file_attr.replace("\\", "/")
            if not file_path.startswith("tests/") and not file_path.startswith("applications/"):
                file_path = f"tests/{file_path}" if "." in file_path else file_path
        else:
            file_path = classname.replace(".", "/") + ".py" if classname else "unknown"
        node_id = f"{classname}::{name}" if classname else name
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
    path = run_root / "logs" / f"{batch_id}_nodes.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    by_classname: dict[str, str] = {}
    for rec in capture.get("node_records", []):
        nid = rec.get("node_id")
        if nid:
            by_classname[nid.rsplit("::", 1)[-1] if "::" in nid else nid] = nid
    merged: list[TestcaseRecord] = []
    for rec in records:
        cap_id = by_classname.get(rec.name)
        if cap_id and "::" in cap_id:
            merged.append(
                TestcaseRecord(
                    node_id=cap_id,
                    file_path=rec.file_path,
                    name=rec.name,
                    outcome=rec.outcome,
                    batch_id=rec.batch_id,
                    source_commit_sha=rec.source_commit_sha,
                )
            )
        else:
            merged.append(rec)
    return merged
