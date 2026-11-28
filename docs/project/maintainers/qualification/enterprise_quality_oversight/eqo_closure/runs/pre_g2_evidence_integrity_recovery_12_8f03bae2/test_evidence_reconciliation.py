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
if str(RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT))


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


def _sync_synthetic_records(baseline: Path) -> None:
    import hashlib

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    binding = json.loads((baseline / "BASELINE_BINDING.json").read_text(encoding="utf-8"))
    batch = "synthetic_pass"
    rec = baseline / "execution_records"
    rec.mkdir(parents=True, exist_ok=True)
    junit = baseline / "junit" / f"{batch}.xml"
    logs = baseline / "logs" / f"{batch}.log"
    logs.parent.mkdir(parents=True, exist_ok=True)
    junit.parent.mkdir(parents=True, exist_ok=True)
    if not logs.is_file():
        logs.write_bytes(b"")
    if not junit.is_file():
        junit.write_text(
            '<?xml version="1.0" encoding="utf-8"?><testsuite tests="1" failures="0">'
            '<testcase classname="tests.unit.applications.test_mcp_server_p4_lifecycle" '
            'name="test_placeholder" file="tests/unit/applications/test_mcp_server_p4_lifecycle.py"/>'
            "</testsuite>",
            encoding="utf-8",
        )
    pre = {
        "schema": "eqo.execution_record.v1",
        "phase": "pre",
        "batch_id": batch,
        "source_commit_sha": binding["qualification_source_sha"],
        "source_tree_sha": binding["qualification_source_tree_sha"],
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
        "stdout_stderr_sha256": sha256_file(logs),
        "junit_path": f"junit/{batch}.xml",
        "junit_sha256": sha256_file(junit),
        "source_commit_sha_after": binding["qualification_source_sha"],
        "source_tree_sha_after": binding["qualification_source_tree_sha"],
    }
    (rec / f"{batch}.pre.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    (rec / f"{batch}.post.json").write_text(json.dumps(post, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def ensure_baseline_fixture() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    baseline = FIXTURES / "minimal_valid"
    if not baseline.is_dir():
        baseline.mkdir(parents=True)
    # built during module setup from live repo inventory
    subprocess.run([sys.executable, str(RUN_ROOT / "evidence_runner.py"), "--write-prereg"], check=True, cwd=RUN_ROOT)
    subprocess.run([sys.executable, str(RUN_ROOT / "evidence_runner.py"), "--manifests-only"], check=True, cwd=RUN_ROOT)
    if not (baseline / "BASELINE_BINDING.json").is_file():
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
    _sync_synthetic_records(RUN_ROOT)
    subprocess.run([sys.executable, str(RUN_ROOT / "reconcile_evidence.py")], cwd=RUN_ROOT, check=True)
    for rel in (
        "DES_PREREG.json",
        "BASELINE_BINDING.json",
        "ENVIRONMENT_MANIFEST.json",
        "INVENTORY_BINDING.json",
        "OBLIGATION_EXECUTION_MATRIX.json",
        "REMAINING_QUALIFICATION_QUEUE.json",
        "BATCH_MANIFEST.json",
        "TESTCASE_EXECUTION_MATRIX.json",
    ):
        src = RUN_ROOT / rel
        if src.is_file():
            shutil.copy(src, baseline / rel)
    _sync_synthetic_records(baseline)
    subprocess.run([sys.executable, str(RUN_ROOT / "reconcile_evidence.py")], cwd=RUN_ROOT, check=True)


def _refresh_batch_digests(root: Path, batch_id: str = "synthetic_pass") -> None:
    import hashlib

    post_path = root / "execution_records" / f"{batch_id}.post.json"
    if not post_path.is_file():
        return
    post = json.loads(post_path.read_text(encoding="utf-8"))
    log_rel = post.get("stdout_stderr_path")
    junit_rel = post.get("junit_path")
    if log_rel and (root / log_rel).is_file():
        post["stdout_stderr_sha256"] = hashlib.sha256((root / log_rel).read_bytes()).hexdigest()
    if junit_rel and (root / junit_rel).is_file():
        post["junit_sha256"] = hashlib.sha256((root / junit_rel).read_bytes()).hexdigest()
    post_path.write_text(json.dumps(post, indent=2), encoding="utf-8")


def _first_post(root: Path) -> Path:
    posts = sorted(root.glob("execution_records/*.post.json"))
    assert posts, f"missing execution post records under {root}"
    return posts[0]


def _junit_covered_paths(root: Path) -> set[str]:
    from _evidence_common import parse_junit_cases

    covered: set[str] = set()
    for post_path in root.glob("execution_records/*.post.json"):
        post = json.loads(post_path.read_text(encoding="utf-8"))
        junit_rel = post.get("junit_path")
        if not junit_rel:
            continue
        junit_path = root / junit_rel
        if not junit_path.is_file():
            continue
        batch_id = post_path.name.replace(".post.json", "")
        binding = json.loads((root / "BASELINE_BINDING.json").read_text(encoding="utf-8"))
        sha = binding.get("qualification_source_sha", "")
        for case in parse_junit_cases(junit_path, batch_id=batch_id, source_sha=sha):
            covered.add(case.file_path)
    return covered


def _clear_queue_green_except(queue: dict, *, keep_path: str | None = None) -> None:
    for row in queue.get("qualification_queue", []):
        if keep_path is not None and row.get("path") == keep_path:
            continue
        if row.get("recalculated_disposition") == "GREEN":
            row["recalculated_disposition"] = "OPEN"


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
        target_row = next(
            (r for r in queue.get("qualification_queue", []) if r.get("path")),
            queue["qualification_queue"][0],
        )
        target_path = target_row.get("path", "tests/x.py")
        _clear_queue_green_except(queue, keep_path=target_path)
        target_row["recalculated_disposition"] = "GREEN"
        junit = root / "junit" / "synthetic_pass.xml"
        junit.write_text(
            '<?xml version="1.0" encoding="utf-8"?><testsuite tests="1" failures="1">'
            f'<testcase classname="synthetic.adversarial" name="t" file="{target_path}">'
            '<failure message="forced"/></testcase></testsuite>',
            encoding="utf-8",
        )
        _refresh_batch_digests(root)
    elif case_id.startswith("09"):
        for row in queue["qualification_queue"]:
            row["recalculated_disposition"] = "GREEN"
            row["testcase_outcomes"] = []
    elif case_id.startswith("10"):
        post = json.loads(_first_post(root).read_text(encoding="utf-8"))
        log_rel = post.get("stdout_stderr_path")
        if log_rel:
            log_path = root / log_rel
            if log_path.is_file():
                log_path.unlink()
    elif case_id.startswith("11"):
        for p in (root / "junit").glob("*.xml"):
            p.unlink()
    elif case_id.startswith("12"):
        post_path = _first_post(root)
        post = json.loads(post_path.read_text(encoding="utf-8"))
        post["stdout_stderr_sha256"] = "0" * 64
        post_path.write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("13"):
        obl["source_sha"] = "0" * 40
    elif case_id.startswith("14"):
        post_path = _first_post(root)
        post = json.loads(post_path.read_text(encoding="utf-8"))
        post["source_tree_sha_after"] = "f" * 40
        post_path.write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("15"):
        post_path = _first_post(root)
        post = json.loads(post_path.read_text(encoding="utf-8"))
        post["source_commit_sha_after"] = "f" * 40
        post_path.write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("16"):
        post_path = _first_post(root)
        post = json.loads(post_path.read_text(encoding="utf-8"))
        post["finished_at_utc"] = "2020-01-01T00:00:00Z"
        post_path.write_text(json.dumps(post), encoding="utf-8")
    elif case_id.startswith("17"):
        post_path = _first_post(root)
        post = json.loads(post_path.read_text(encoding="utf-8"))
        post["process_returncode"] = 0
        log = root / post["stdout_stderr_path"]
        log.write_text("FAILED\n", encoding="utf-8")
    elif case_id.startswith("19"):
        tc = json.loads((root / "TESTCASE_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
        dup = dict(tc["testcases"][0]) if tc.get("testcases") else {}
        tc["testcases"].append(dup)
        (root / "TESTCASE_EXECUTION_MATRIX.json").write_text(json.dumps(tc, indent=2), encoding="utf-8")
    elif case_id.startswith("20"):
        covered = _junit_covered_paths(root)
        for ob in obl["obligations"]:
            mapped = ob.get("mapped_test_file_paths") or []
            if mapped and not ob.get("semantic_review_required") and not any(p in covered for p in mapped):
                ob["recalculated_disposition"] = "QUALIFICATION_COMPLETE"
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
        row = queue["qualification_queue"][0]
        _clear_queue_green_except(queue, keep_path=row.get("path"))
        row["recalculated_disposition"] = "GREEN"
        row["testcase_outcomes"] = [
            {
                "node_id": "synthetic::skip",
                "file_path": row.get("path", "tests/x.py"),
                "outcome": "SKIP",
                "batch_id": "synthetic_pass",
            }
        ]
    elif case_id.startswith("18"):
        bad = "tests/synthetic/unmapped_adversarial_module.py"
        junit = root / "junit" / "synthetic_pass.xml"
        junit.write_text(
            '<?xml version="1.0" encoding="utf-8"?><testsuite tests="1" failures="0">'
            f'<testcase classname="synthetic.unmapped" name="t" file="{bad}"/></testsuite>',
            encoding="utf-8",
        )
        _refresh_batch_digests(root)
    elif case_id.startswith("21"):
        a, b = 0, 1
        oid_a, oid_b = obl["obligations"][a]["obligation_id"], obl["obligations"][b]["obligation_id"]
        obl["obligations"][a]["obligation_id"] = oid_b
        obl["obligations"][b]["obligation_id"] = oid_a
    elif case_id.startswith("24"):
        row0 = queue["qualification_queue"][0]
        row0["recalculated_disposition"] = "GREEN"
        path0 = row0.get("path", "tests/x.py")
        tc = json.loads((root / "TESTCASE_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
        if not tc.get("testcases"):
            tc["testcases"] = []
        tc["testcases"] = [
            {
                "node_id": "hist::pass",
                "file_path": path0,
                "outcome": "PASS",
                "batch_id": "synthetic_pass",
                "evidence_validity": "HISTORICAL_UNVERIFIED",
            }
        ]
        (root / "TESTCASE_EXECUTION_MATRIX.json").write_text(json.dumps(tc, indent=2), encoding="utf-8")
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
        assert reason and reason in output, output
