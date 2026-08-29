# © Artur Czarnecki. All rights reserved.

"""HARDEN-2C — real cross-process ProblemLifecycleEngine.reconcile() OCC proof."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_TENANT = "harden-2c-tenant"
_OTHER_TENANT = "harden-2c-other-tenant"
_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_harden_2c"
_COLLECTION_PREFIX = "harden_2c_"
_WORKER = Path(__file__).with_name("harden_2a_concurrency_proof_worker.py")
_REPO_ROOT = Path(__file__).resolve().parents[4]

_EXIT_SKIP = 2
_DONE_TIMEOUT_SECONDS = 60.0
_DONE_POLL_SECONDS = 0.1
_MAX_CONFLICT_ROUNDS = 15


def _mint_task_id_str() -> str:
    return f"task_{uuid.uuid4().hex}"


def _mint_run_id_str() -> str:
    return f"run_{uuid.uuid4().hex}"


def _resolve_mongodb_uri() -> str:
    return os.environ.get("INTERGRAX_MONGODB_URI", _DEFAULT_URI).strip() or _DEFAULT_URI


def _proof_env(*, collection_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env["INTERGRAX_MONGODB_URI"] = _resolve_mongodb_uri()
    env["INTERGRAX_MONGODB_DATABASE"] = _DEFAULT_DATABASE
    env["INTERGRAX_MONGODB_COLLECTION"] = collection_name
    pythonpath = [
        str(_REPO_ROOT),
        str(_REPO_ROOT / "agents"),
        str(_REPO_ROOT / "applications"),
    ]
    if existing := env.get("PYTHONPATH", "").strip():
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _run_worker(
    command: str,
    *,
    env: dict[str, str],
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_WORKER), command, *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _wait_for_done(done_path: Path) -> dict[str, object]:
    deadline = time.monotonic() + _DONE_TIMEOUT_SECONDS
    while not done_path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"worker did not finish: {done_path}")
        time.sleep(_DONE_POLL_SECONDS)
    return json.loads(done_path.read_text(encoding="utf-8"))


def _wait_for_snapshot(snapshot_path: Path) -> dict[str, object]:
    deadline = time.monotonic() + _DONE_TIMEOUT_SECONDS
    while not snapshot_path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"worker snapshot missing: {snapshot_path}")
        time.sleep(_DONE_POLL_SECONDS)
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _subject_execution_ids(subject_ref: dict[str, object]) -> tuple[str, str]:
    subject = subject_ref["subject"]
    assert isinstance(subject, dict)
    return str(subject["task_id"]), str(subject["run_id"])


def _spawn_lifecycle_worker(
    *,
    env: dict[str, str],
    sync_dir: Path,
    worker_label: str,
    problem_id: str | None,
    subject_task_id: str,
    subject_run_id: str,
) -> subprocess.Popen[str]:
    start_path = sync_dir / "start.signal"
    update_path = sync_dir / "update.signal"
    read_snapshot_path = sync_dir / f"read_snapshot.{worker_label}.json"
    done_path = sync_dir / f"worker_{worker_label}.done"
    args = [
        "concurrent-lifecycle-reconcile",
        "--tenant-id",
        _TENANT,
        "--worker-label",
        worker_label,
        "--subject-task-id",
        subject_task_id,
        "--subject-run-id",
        subject_run_id,
        "--start-path",
        str(start_path),
        "--update-path",
        str(update_path),
        "--done-path",
        str(done_path),
    ]
    if problem_id is not None:
        args.extend(["--problem-id", problem_id, "--read-snapshot-path", str(read_snapshot_path)])
    return subprocess.Popen(
        [sys.executable, str(_WORKER), *args],
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _run_lifecycle_update_round(
    *,
    env: dict[str, str],
    sync_dir: Path,
    problem_id: str,
    baseline_record_version: int,
    subject_a_task_id: str,
    subject_a_run_id: str,
    subject_b_task_id: str,
    subject_b_run_id: str,
) -> tuple[dict[str, object], dict[str, object], subprocess.Popen[str], subprocess.Popen[str]]:
    start_path = sync_dir / "start.signal"
    update_path = sync_dir / "update.signal"
    read_a = sync_dir / "read_snapshot.a.json"
    read_b = sync_dir / "read_snapshot.b.json"
    done_a = sync_dir / "worker_a.done"
    done_b = sync_dir / "worker_b.done"

    for path in (start_path, update_path, read_a, read_b, done_a, done_b):
        if path.exists():
            path.unlink()

    process_a = _spawn_lifecycle_worker(
        env=env,
        sync_dir=sync_dir,
        worker_label="a",
        problem_id=problem_id,
        subject_task_id=subject_a_task_id,
        subject_run_id=subject_a_run_id,
    )
    process_b = _spawn_lifecycle_worker(
        env=env,
        sync_dir=sync_dir,
        worker_label="b",
        problem_id=problem_id,
        subject_task_id=subject_b_task_id,
        subject_run_id=subject_b_run_id,
    )

    time.sleep(0.2)
    start_path.write_text("go\n", encoding="utf-8")

    snapshot_a = _wait_for_snapshot(read_a)
    snapshot_b = _wait_for_snapshot(read_b)
    assert snapshot_a["record_version"] == baseline_record_version
    assert snapshot_b["record_version"] == baseline_record_version
    update_path.write_text("go\n", encoding="utf-8")

    stdout_a, stderr_a = process_a.communicate(timeout=_DONE_TIMEOUT_SECONDS)
    stdout_b, stderr_b = process_b.communicate(timeout=_DONE_TIMEOUT_SECONDS)
    assert process_a.returncode == 0, stderr_a or stdout_a
    assert process_b.returncode == 0, stderr_b or stdout_b

    payload_a = _wait_for_done(done_a)
    payload_b = _wait_for_done(done_b)
    return payload_a, payload_b, process_a, process_b


def test_harden_2c_cross_process_lifecycle_reconcile_on_mongodb(tmp_path: Path) -> None:
    """
    HARDEN-2C lifecycle proof:

    baseline occurrence_count=1; two independent processes reconcile distinct
    occurrences through ProblemLifecycleEngine; fresh process C verifies count=3.
    """
    collection_name = f"{_COLLECTION_PREFIX}{uuid.uuid4().hex}"
    env = _proof_env(collection_name=collection_name)
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)

    probe = _run_worker("probe", env=env, args=[])
    if probe.returncode == _EXIT_SKIP:
        pytest.skip(probe.stderr.strip())
    assert probe.returncode == 0, probe.stderr

    seed = _run_worker(
        "seed-baseline",
        env=env,
        args=["--tenant-id", _TENANT, "--other-tenant-id", _OTHER_TENANT],
    )
    assert seed.returncode == 0, seed.stderr
    seed_payload = json.loads(seed.stdout)
    assert seed_payload["phase"] == "seed"
    problem_id = seed_payload["problem_id"]
    baseline_count = seed_payload["baseline_occurrence_count"]
    baseline_version = seed_payload["baseline_record_version"]
    assert baseline_count == 1
    assert baseline_version == 1

    baseline_subject_refs = seed_payload["baseline_subject_refs"]
    assert len(baseline_subject_refs) == 1
    baseline_task_id, baseline_run_id = _subject_execution_ids(baseline_subject_refs[0])

    subject_a_task_id = _mint_task_id_str()
    subject_a_run_id = _mint_run_id_str()
    subject_b_task_id = _mint_task_id_str()
    subject_b_run_id = _mint_run_id_str()

    payload_a: dict[str, object] | None = None
    payload_b: dict[str, object] | None = None
    process_a: subprocess.Popen[str] | None = None
    process_b: subprocess.Popen[str] | None = None

    for _round in range(_MAX_CONFLICT_ROUNDS):
        payload_a, payload_b, process_a, process_b = _run_lifecycle_update_round(
            env=env,
            sync_dir=sync_dir,
            problem_id=problem_id,
            baseline_record_version=baseline_version,
            subject_a_task_id=subject_a_task_id,
            subject_a_run_id=subject_a_run_id,
            subject_b_task_id=subject_b_task_id,
            subject_b_run_id=subject_b_run_id,
        )
        total_conflicts = int(payload_a["conflicts_observed"]) + int(payload_b["conflicts_observed"])
        if total_conflicts >= 1:
            break

        cleanup = _run_worker(
            "cleanup",
            env=env,
            args=["--tenant-id", _TENANT, "--tenant-id", _OTHER_TENANT],
        )
        assert cleanup.returncode == 0, cleanup.stderr
        seed = _run_worker(
            "seed-baseline",
            env=env,
            args=["--tenant-id", _TENANT, "--other-tenant-id", _OTHER_TENANT],
        )
        assert seed.returncode == 0, seed.stderr
        seed_payload = json.loads(seed.stdout)
        problem_id = seed_payload["problem_id"]
        baseline_subject_refs = seed_payload["baseline_subject_refs"]
        baseline_task_id, baseline_run_id = _subject_execution_ids(baseline_subject_refs[0])
        subject_a_task_id = _mint_task_id_str()
        subject_a_run_id = _mint_run_id_str()
        subject_b_task_id = _mint_task_id_str()
        subject_b_run_id = _mint_run_id_str()

    assert payload_a is not None and payload_b is not None
    assert process_a is not None and process_b is not None
    assert process_a.pid != process_b.pid

    total_conflicts = int(payload_a["conflicts_observed"]) + int(payload_b["conflicts_observed"])
    assert total_conflicts >= 1, (
        "HARDEN-2C requires observed CAS conflict evidence; "
        f"writer_a={payload_a['conflicts_observed']} writer_b={payload_b['conflicts_observed']}"
    )
    assert payload_a["status"] == "updated"
    assert payload_b["status"] == "updated"

    read_final = _run_worker(
        "read-final",
        env=env,
        args=[
            "--tenant-id",
            _TENANT,
            "--problem-id",
            problem_id,
            "--other-tenant-id",
            _OTHER_TENANT,
        ],
    )
    assert read_final.returncode == 0, read_final.stderr
    final_payload = json.loads(read_final.stdout)
    reader_pid = final_payload["pid"]
    assert reader_pid != process_a.pid
    assert reader_pid != process_b.pid

    assert final_payload["phase"] == "read"
    assert final_payload["listed_problem_ids"] == [problem_id]
    assert final_payload["reconciliation_lookup_problem_id"] == problem_id
    assert final_payload["other_tenant_problem_count"] == 0
    assert final_payload["occurrence_count"] == 3
    assert final_payload["record_version"] == 3

    occurrence_subject_refs = final_payload["occurrence_subject_refs"]
    assert len(occurrence_subject_refs) == 3
    observed_pairs = {
        _subject_execution_ids(subject_ref) for subject_ref in occurrence_subject_refs
    }
    assert observed_pairs == {
        (baseline_task_id, baseline_run_id),
        (subject_a_task_id, subject_a_run_id),
        (subject_b_task_id, subject_b_run_id),
    }

    cleanup = _run_worker(
        "cleanup",
        env=env,
        args=["--tenant-id", _TENANT, "--tenant-id", _OTHER_TENANT],
    )
    assert cleanup.returncode == 0, cleanup.stderr


def test_harden_2c_cross_process_lifecycle_create_race_on_mongodb(tmp_path: Path) -> None:
    """
    HARDEN-2C create-race proof:

    two processes reconcile different subjects with the same reconciliation
    identity before any Problem exists; converge to one Problem with two occurrences.
    """
    collection_name = f"{_COLLECTION_PREFIX}{uuid.uuid4().hex}"
    env = _proof_env(collection_name=collection_name)
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)

    probe = _run_worker("probe", env=env, args=[])
    if probe.returncode == _EXIT_SKIP:
        pytest.skip(probe.stderr.strip())
    assert probe.returncode == 0, probe.stderr

    subject_a_task_id = _mint_task_id_str()
    subject_a_run_id = _mint_run_id_str()
    subject_b_task_id = _mint_task_id_str()
    subject_b_run_id = _mint_run_id_str()

    payload_a: dict[str, object] | None = None
    payload_b: dict[str, object] | None = None
    process_a: subprocess.Popen[str] | None = None
    process_b: subprocess.Popen[str] | None = None
    problem_id: str | None = None

    for _round in range(_MAX_CONFLICT_ROUNDS):
        start_path = sync_dir / "start.signal"
        update_path = sync_dir / "update.signal"
        done_a = sync_dir / "worker_a.done"
        done_b = sync_dir / "worker_b.done"
        for path in (start_path, update_path, done_a, done_b):
            if path.exists():
                path.unlink()

        purge = _run_worker(
            "cleanup",
            env=env,
            args=["--tenant-id", _TENANT, "--tenant-id", _OTHER_TENANT],
        )
        assert purge.returncode == 0, purge.stderr

        process_a = _spawn_lifecycle_worker(
            env=env,
            sync_dir=sync_dir,
            worker_label="a",
            problem_id=None,
            subject_task_id=subject_a_task_id,
            subject_run_id=subject_a_run_id,
        )
        process_b = _spawn_lifecycle_worker(
            env=env,
            sync_dir=sync_dir,
            worker_label="b",
            problem_id=None,
            subject_task_id=subject_b_task_id,
            subject_run_id=subject_b_run_id,
        )

        time.sleep(0.2)
        start_path.write_text("go\n", encoding="utf-8")
        update_path.write_text("go\n", encoding="utf-8")

        stdout_a, stderr_a = process_a.communicate(timeout=_DONE_TIMEOUT_SECONDS)
        stdout_b, stderr_b = process_b.communicate(timeout=_DONE_TIMEOUT_SECONDS)
        if process_a.returncode != 0 or process_b.returncode != 0:
            subject_a_task_id = _mint_task_id_str()
            subject_a_run_id = _mint_run_id_str()
            subject_b_task_id = _mint_task_id_str()
            subject_b_run_id = _mint_run_id_str()
            continue
        assert process_a.pid != process_b.pid

        payload_a = _wait_for_done(done_a)
        payload_b = _wait_for_done(done_b)
        if payload_a["status"] not in {"created", "updated"}:
            continue
        if payload_b["status"] not in {"created", "updated"}:
            continue

        problem_ids = {str(payload_a["problem_id"]), str(payload_b["problem_id"])}
        if len(problem_ids) != 1:
            subject_a_task_id = _mint_task_id_str()
            subject_a_run_id = _mint_run_id_str()
            subject_b_task_id = _mint_task_id_str()
            subject_b_run_id = _mint_run_id_str()
            continue
        problem_id = next(iter(problem_ids))
        break

    assert payload_a is not None and payload_b is not None
    assert process_a is not None and process_b is not None
    assert problem_id is not None
    assert process_a.returncode == 0
    assert process_b.returncode == 0

    read_final = _run_worker(
        "read-final",
        env=env,
        args=[
            "--tenant-id",
            _TENANT,
            "--problem-id",
            problem_id,
            "--other-tenant-id",
            _OTHER_TENANT,
        ],
    )
    assert read_final.returncode == 0, read_final.stderr
    final_payload = json.loads(read_final.stdout)
    assert final_payload["listed_problem_ids"] == [problem_id]
    assert final_payload["reconciliation_lookup_problem_id"] == problem_id
    assert final_payload["other_tenant_problem_count"] == 0
    assert final_payload["occurrence_count"] == 2
    assert final_payload["record_version"] == 2
    observed_pairs = {
        _subject_execution_ids(subject_ref) for subject_ref in final_payload["occurrence_subject_refs"]
    }
    assert observed_pairs == {
        (subject_a_task_id, subject_a_run_id),
        (subject_b_task_id, subject_b_run_id),
    }

    cleanup = _run_worker(
        "cleanup",
        env=env,
        args=["--tenant-id", _TENANT, "--tenant-id", _OTHER_TENANT],
    )
    assert cleanup.returncode == 0, cleanup.stderr
