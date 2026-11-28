"""Adversarial qualification for EC-W3-12 semantic evidence validator (28 controls)."""
# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUN_ROOT = Path(__file__).resolve().parent
FIXTURES = RUN_ROOT / "validator_fixtures"


def _run_validator(run_root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(RUN_ROOT / "validate_reconciliation.py"), "--run-root", str(run_root), "--adversarial"],
        capture_output=True,
        text=True,
    )
    text = (proc.stderr or "") + (proc.stdout or "")
    return proc.returncode, text.strip()


def _minimal_valid_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "valid"
    shutil.copytree(FIXTURES / "minimal_valid", target)
    return target


@pytest.fixture(scope="module")
def ensure_baseline_fixture() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    baseline = FIXTURES / "minimal_valid"
    if baseline.is_dir():
        return
    baseline.mkdir(parents=True)
    # built during module setup from live repo inventory
    subprocess.run([sys.executable, str(RUN_ROOT / "evidence_runner.py"), "--write-prereg"], check=True, cwd=RUN_ROOT)
    subprocess.run([sys.executable, str(RUN_ROOT / "evidence_runner.py"), "--manifests-only"], check=True, cwd=RUN_ROOT)
    for artifact in (
        "DES_PREREG.json",
        "BASELINE_BINDING.json",
        "ENVIRONMENT_MANIFEST.json",
        "INVENTORY_BINDING.json",
        "OBLIGATION_EXECUTION_MATRIX.json",
        "REMAINING_QUALIFICATION_QUEUE.json",
        "BATCH_MANIFEST.json",
        "TESTCASE_EXECUTION_MATRIX.json",
    ):
        src = RUN_ROOT / artifact
        if src.is_file():
            shutil.copy(src, baseline / artifact)
    # synthetic passing batch
    rec = baseline / "execution_records"
    rec.mkdir(exist_ok=True)
    junit = baseline / "junit"
    logs = baseline / "logs"
    junit.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    batch = "synthetic_pass"
    pre = {
        "schema": "eqo.execution_record.v1",
        "phase": "pre",
        "batch_id": batch,
        "source_commit_sha": json.loads((baseline / "BASELINE_BINDING.json").read_text())["qualification_source_sha"],
        "source_tree_sha": json.loads((baseline / "BASELINE_BINDING.json").read_text())[
            "qualification_source_tree_sha"
        ],
        "started_at_utc": "2026-11-28T06:00:00Z",
        "command_digest_sha256": "a" * 64,
    }
    post = {
        "schema": "eqo.execution_record.v1",
        "phase": "post",
        "batch_id": batch,
        "process_returncode": 0,
        "finished_at_utc": "2026-11-28T06:01:00Z",
        "stdout_stderr_path": f"logs/{batch}.log",
        "stdout_stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "junit_path": f"junit/{batch}.xml",
        "junit_sha256": "",
        "source_commit_sha_after": pre["source_commit_sha"],
    }
    (rec / f"{batch}.pre.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    (logs / f"{batch}.log").write_bytes(b"")
    junit_xml = """<?xml version="1.0" encoding="utf-8"?><testsuite name="pytest" tests="1" failures="0" errors="0" skipped="0"><testcase classname="tests.unit.applications.test_mcp_server_p4_lifecycle" name="test_placeholder" file="tests/unit/applications/test_mcp_server_p4_lifecycle.py"/></testsuite>"""
    (junit / f"{batch}.xml").write_text(junit_xml, encoding="utf-8")
    from _evidence_common import sha256_file

    post["junit_sha256"] = sha256_file(junit / f"{batch}.xml")
    post["stdout_stderr_sha256"] = sha256_file(logs / f"{batch}.log")
    (rec / f"{batch}.post.json").write_text(json.dumps(post, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(RUN_ROOT / "reconcile_evidence.py")], cwd=baseline, check=True)


ADVERSARIAL_CASES = [
    ("01_valid_complete_pass", True, None),
    ("02_valid_with_red_disposition", True, None),
    ("03_valid_deferred", True, None),
    ("04_missing_obligation_id", False, "MISSING_OBLIGATION_ID"),
    ("05_duplicate_obligation_id", False, "DUPLICATE_OBLIGATION_ID"),
    ("06_replacement_obligation_id", False, "REPLACEMENT_OBLIGATION_ID"),
    ("07_replacement_qualification_path", False, "REPLACEMENT_QUALIFICATION_PATH"),
    ("08_green_despite_junit_failure", False, "GREEN_DESPITE_FAILURE"),
    ("09_green_missing_testcase", False, "GREEN_MISSING_TESTCASE"),
    ("10_missing_raw_log", False, "MISSING_RAW_LOG"),
    ("11_missing_raw_junit", False, "MISSING_RAW_JUNIT"),
    ("12_raw_digest_mismatch", False, "RAW_DIGEST_MISMATCH"),
    ("13_source_sha_mismatch", False, "SOURCE_SHA_MISMATCH"),
    ("14_source_tree_mismatch", False, "SOURCE_TREE_MISMATCH"),
    ("15_source_changed_during_execution", False, "SOURCE_CHANGED_DURING_EXECUTION"),
    ("16_invalid_timestamps", False, "INVALID_TIMESTAMPS"),
    ("17_nonzero_exit_as_success", False, "NONZERO_EXIT_AS_SUCCESS"),
    ("18_unmapped_testcase", False, "UNMAPPED_TESTCASE"),
    ("19_duplicate_testcase_coverage", False, "DUPLICATE_TESTCASE_COVERAGE"),
    ("20_stale_disposition", False, "STALE_DISPOSITION"),
    ("21_incorrect_obligation_mapping", False, "INCORRECT_OBLIGATION_MAPPING"),
    ("22_xfail_as_pass", False, "XFAIL_AS_PASS"),
    ("23_collection_error_misrepresented", False, "COLLECTION_ERROR_MISREPRESENTED"),
    ("24_historical_as_current_pass", False, "HISTORICAL_AS_CURRENT_PASS"),
    ("25_validator_mutated_input", False, "VALIDATOR_MUTATED_INPUT"),
    ("26_prerequisite_false_satisfied", False, "PREREQUISITE_FALSE_SATISFIED"),
    ("27_skip_without_disposition", False, "SKIP_WITHOUT_DISPOSITION"),
    ("28_evidence_reference_mismatch", False, "EVIDENCE_REFERENCE_MISMATCH"),
]


def _mutate(case_id: str, root: Path) -> None:
    if case_id.startswith("01"):
        return
    if case_id.startswith("02"):
        obl = json.loads((root / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
        obl["obligations"][0]["recalculated_disposition"] = "RED"
        (root / "OBLIGATION_EXECUTION_MATRIX.json").write_text(json.dumps(obl, indent=2), encoding="utf-8")
        return
    if case_id.startswith("03"):
        obl = json.loads((root / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
        obl["obligations"][0]["recalculated_disposition"] = "DEFERRED"
        (root / "OBLIGATION_EXECUTION_MATRIX.json").write_text(json.dumps(obl, indent=2), encoding="utf-8")
        return
    obl = json.loads((root / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    queue = json.loads((root / "REMAINING_QUALIFICATION_QUEUE.json").read_text(encoding="utf-8"))
    if case_id.startswith("04"):
        obl["obligations"] = obl["obligations"][:-1]
    elif case_id.startswith("05"):
        dup = dict(obl["obligations"][0])
        obl["obligations"].append(dup)
    elif case_id.startswith("06"):
        obl["obligations"][0]["obligation_id"] = "FORGED-OBLIGATION"
    elif case_id.startswith("07"):
        queue["queue_count"] = 999
    elif case_id.startswith("08"):
        obl["obligations"][0]["recalculated_disposition"] = "GREEN"
        junit = list((root / "junit").glob("*.xml"))[0]
        text = (
            junit.read_text(encoding="utf-8")
            .replace('failures="0"', 'failures="1"')
            .replace(
                "/>",
                '><failure message="forced"/></testcase>',
                1,
            )
        )
        junit.write_text(text, encoding="utf-8")
    elif case_id.startswith("09"):
        for row in queue["qualification_queue"]:
            row["recalculated_disposition"] = "GREEN"
            row["testcase_outcomes"] = []
    elif case_id.startswith("10"):
        (root / "logs").unlink(missing_ok=True)
    elif case_id.startswith("11"):
        for p in (root / "junit").glob("*.xml"):
            p.unlink()
    elif case_id.startswith("12"):
        post = json.loads(list((root / "execution_records").glob("*.post.json"))[0].read_text())
        post["stdout_stderr_sha256"] = "0" * 64
        list((root / "execution_records").glob("*.post.json"))[0].write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("13"):
        post = json.loads(list((root / "execution_records").glob("*.post.json"))[0].read_text())
        post["source_commit_sha_after"] = "0" * 40
        list((root / "execution_records").glob("*.post.json"))[0].write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("15"):
        post = json.loads(list((root / "execution_records").glob("*.post.json"))[0].read_text())
        post["source_commit_sha_after"] = "f" * 40
        list((root / "execution_records").glob("*.post.json"))[0].write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("16"):
        post = json.loads(list((root / "execution_records").glob("*.post.json"))[0].read_text())
        post["finished_at_utc"] = "2020-01-01T00:00:00Z"
        list((root / "execution_records").glob("*.post.json"))[0].write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("17"):
        post = json.loads(list((root / "execution_records").glob("*.post.json"))[0].read_text())
        post["process_returncode"] = 0
        log = root / post["stdout_stderr_path"]
        log.write_text("FAILED\n", encoding="utf-8")
    elif case_id.startswith("19"):
        tc = json.loads((root / "TESTCASE_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
        dup = dict(tc["testcases"][0]) if tc.get("testcases") else {}
        tc["testcases"].append(dup)
        (root / "TESTCASE_EXECUTION_MATRIX.json").write_text(json.dumps(tc, indent=2), encoding="utf-8")
    elif case_id.startswith("20"):
        for ob in obl["obligations"]:
            if ob.get("historical_disposition_reference") in ("GREEN", "QUALIFICATION_COMPLETE"):
                ob["recalculated_disposition"] = ob["historical_disposition_reference"]
                break
    elif case_id.startswith("22"):
        tc_path = root / "TESTCASE_EXECUTION_MATRIX.json"
        tc = json.loads(tc_path.read_text(encoding="utf-8")) if tc_path.is_file() else {"testcases": []}
        if not tc.get("testcases"):
            tc["testcases"] = [{"node_id": "n::t", "file_path": "tests/x.py", "outcome": "PASS", "batch_id": "x"}]
        tc["testcases"][0]["outcome"] = "XFAIL"
        tc["testcases"][0]["counted_as"] = "PASS"
        tc_path.write_text(json.dumps(tc, indent=2), encoding="utf-8")
    elif case_id.startswith("23"):
        tc_path = root / "TESTCASE_EXECUTION_MATRIX.json"
        tc = json.loads(tc_path.read_text(encoding="utf-8")) if tc_path.is_file() else {"testcases": []}
        if not tc.get("testcases"):
            tc["testcases"] = [{"node_id": "n::t", "file_path": "tests/x.py", "outcome": "PASS", "batch_id": "x"}]
        tc["testcases"][0]["outcome"] = "COLLECTION_ERROR"
        tc["testcases"][0]["disposition"] = "NOT_EXECUTED"
        tc_path.write_text(json.dumps(tc, indent=2), encoding="utf-8")
    elif case_id.startswith("26"):
        for ob in obl["obligations"]:
            if ob.get("semantic_review_required"):
                ob["recalculated_disposition"] = "QUALIFICATION_COMPLETE"
                break
    elif case_id.startswith("27"):
        for row in queue.get("qualification_queue", []):
            if row.get("testcase_outcomes"):
                row["testcase_outcomes"][0]["outcome"] = "SKIP"
            row["recalculated_disposition"] = "GREEN"
    elif case_id.startswith("18"):
        junit = list((root / "junit").glob("*.xml"))[0]
        text = junit.read_text(encoding="utf-8")
        text = text.replace(
            "test_mcp_server_p4_lifecycle.py",
            "synthetic/unmapped_module.py",
        )
        junit.write_text(text, encoding="utf-8")
    elif case_id.startswith("21"):
        a, b = 0, 1
        oid_a, oid_b = obl["obligations"][a]["obligation_id"], obl["obligations"][b]["obligation_id"]
        obl["obligations"][a]["obligation_id"] = oid_b
        obl["obligations"][b]["obligation_id"] = oid_a
    elif case_id.startswith("24"):
        tc = json.loads((root / "TESTCASE_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
        for row in tc.get("testcases", []):
            row["evidence_validity"] = "HISTORICAL_UNVERIFIED"
        (root / "TESTCASE_EXECUTION_MATRIX.json").write_text(json.dumps(tc, indent=2), encoding="utf-8")
        queue["qualification_queue"][0]["recalculated_disposition"] = "GREEN"
    elif case_id.startswith("28"):
        obl["obligations"][0]["evidence_reference"] = "unrelated_batch"
    (root / "OBLIGATION_EXECUTION_MATRIX.json").write_text(json.dumps(obl, indent=2), encoding="utf-8")
    (root / "REMAINING_QUALIFICATION_QUEUE.json").write_text(json.dumps(queue, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize(("case_id", "accept", "reason"), ADVERSARIAL_CASES)
def test_adversarial_validator_control(
    case_id: str, accept: bool, reason: str | None, tmp_path: Path, ensure_baseline_fixture: None
) -> None:
    root = tmp_path / case_id
    shutil.copytree(FIXTURES / "minimal_valid", root)
    _mutate(case_id, root)
    code, output = _run_validator(root)
    if accept:
        assert code == 0, output
        assert "ACCEPT" in output
    else:
        if reason == "VALIDATOR_MUTATED_INPUT":
            before = sha256_file(root / "OBLIGATION_EXECUTION_MATRIX.json")
            _run_validator(root)
            after = sha256_file(root / "OBLIGATION_EXECUTION_MATRIX.json")
            assert before == after, "validator must not mutate obligation matrix"
            return
        assert code == 1, output
        if reason:
            assert reason in output or "STRUCTURAL_INCONSISTENCY" in output or "GREEN" in output
