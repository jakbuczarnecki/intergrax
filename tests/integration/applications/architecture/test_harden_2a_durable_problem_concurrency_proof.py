# © Artur Czarnecki. All rights reserved.

"""HARDEN-2A — real cross-process concurrent Problem persistence update proof."""

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

_TENANT = "harden-2a-tenant"
_OTHER_TENANT = "harden-2a-other-tenant"
_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_harden_2a"
_COLLECTION_PREFIX = "harden_2a_"
_WORKER = Path(__file__).with_name("harden_2a_concurrency_proof_worker.py")
_REPO_ROOT = Path(__file__).resolve().parents[4]

_EXIT_SKIP = 2
_DONE_TIMEOUT_SECONDS = 60.0
_DONE_POLL_SECONDS = 0.1


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


def test_harden_2a_cross_process_concurrent_update_on_mongodb(tmp_path: Path) -> None:
    """
    HARDEN-2A process proof:

    two concurrent worker processes, each with its own IntegrationProfile store client,
    append distinct occurrences to the same baseline Problem on MongoDB.
    """
    collection_name = f"{_COLLECTION_PREFIX}{uuid.uuid4().hex}"
    env = _proof_env(collection_name=collection_name)
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    start_path = sync_dir / "start.signal"
    update_path = sync_dir / "update.signal"
    read_a = sync_dir / "read_snapshot.a.json"
    read_b = sync_dir / "read_snapshot.b.json"
    done_a = sync_dir / "worker_a.done"
    done_b = sync_dir / "worker_b.done"

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
    assert baseline_count == 1

    worker_args_common = [
        "--tenant-id",
        _TENANT,
        "--problem-id",
        problem_id,
        "--start-path",
        str(start_path),
        "--update-path",
        str(update_path),
    ]
    process_a = subprocess.Popen(
        [
            sys.executable,
            str(_WORKER),
            "concurrent-update",
            *worker_args_common,
            "--worker-label",
            "a",
            "--read-snapshot-path",
            str(read_a),
            "--done-path",
            str(done_a),
        ],
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process_b = subprocess.Popen(
        [
            sys.executable,
            str(_WORKER),
            "concurrent-update",
            *worker_args_common,
            "--worker-label",
            "b",
            "--read-snapshot-path",
            str(read_b),
            "--done-path",
            str(done_b),
        ],
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    time.sleep(0.2)
    start_path.write_text("go\n", encoding="utf-8")

    _wait_for_done(read_a)
    _wait_for_done(read_b)
    snapshot_a = json.loads(read_a.read_text(encoding="utf-8"))
    snapshot_b = json.loads(read_b.read_text(encoding="utf-8"))
    assert snapshot_a["record_version"] == seed_payload["baseline_record_version"]
    assert snapshot_b["record_version"] == seed_payload["baseline_record_version"]
    update_path.write_text("go\n", encoding="utf-8")

    stdout_a, stderr_a = process_a.communicate(timeout=_DONE_TIMEOUT_SECONDS)
    stdout_b, stderr_b = process_b.communicate(timeout=_DONE_TIMEOUT_SECONDS)
    assert process_a.returncode == 0, stderr_a
    assert process_b.returncode == 0, stderr_b

    payload_a = _wait_for_done(done_a)
    payload_b = _wait_for_done(done_b)
    statuses = {payload_a["status"], payload_b["status"]}
    assert statuses == {"updated", "conflict"}

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
    assert final_payload["phase"] == "read"
    assert final_payload["listed_problem_ids"] == [problem_id]
    assert final_payload["reconciliation_lookup_problem_id"] == problem_id
    assert final_payload["other_tenant_problem_count"] == 0
    assert final_payload["occurrence_count"] == baseline_count + 1
    assert final_payload["record_version"] == seed_payload["baseline_record_version"] + 1
    assert len(final_payload["occurrence_subject_refs"]) == baseline_count + 1

    cleanup = _run_worker(
        "cleanup",
        env=env,
        args=["--tenant-id", _TENANT, "--tenant-id", _OTHER_TENANT],
    )
    assert cleanup.returncode == 0, cleanup.stderr
