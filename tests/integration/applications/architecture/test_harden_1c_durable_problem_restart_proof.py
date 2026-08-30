# © Artur Czarnecki. All rights reserved.

"""HARDEN-1C — durable central Problem store survives real process restart."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_TENANT = "harden-1c-tenant"
_OTHER_TENANT = "harden-1c-other-tenant"
_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_harden_1c"
_COLLECTION_PREFIX = "harden_1c_"
_WORKER = Path(__file__).with_name("harden_1c_restart_proof_worker.py")
_REPO_ROOT = Path(__file__).resolve().parents[4]

_EXIT_SKIP = 2


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


def test_harden_1c_durable_problem_survives_real_process_restart(tmp_path: Path) -> None:
    """
    HARDEN-1C process proof:

    parent pytest process spawns phase A (write + store close + exit) then phase B
    (fresh IntegrationProfile resolution + DiagnosticReadService read).
    """
    collection_name = f"{_COLLECTION_PREFIX}{uuid.uuid4().hex}"
    env = _proof_env(collection_name=collection_name)

    probe = _run_worker("probe", env=env, args=[])
    if probe.returncode == _EXIT_SKIP:
        pytest.skip(probe.stderr.strip())
    assert probe.returncode == 0, probe.stderr

    phase_a = _run_worker(
        "phase-a",
        env=env,
        args=[
            "--work-dir",
            str(tmp_path / "phase_a"),
            "--tenant-id",
            _TENANT,
        ],
    )
    assert phase_a.returncode == 0, phase_a.stderr
    phase_a_payload = json.loads(phase_a.stdout)
    assert phase_a_payload["phase"] == "a"
    assert phase_a_payload["pid"] > 0

    expect_path = tmp_path / "phase_a_expect.json"
    expect_path.write_text(phase_a.stdout, encoding="utf-8")

    phase_b = _run_worker(
        "phase-b",
        env=env,
        args=[
            "--work-dir",
            str(tmp_path / "phase_b"),
            "--tenant-id",
            _TENANT,
            "--other-tenant-id",
            _OTHER_TENANT,
            "--expect-file",
            str(expect_path),
        ],
    )
    assert phase_b.returncode == 0, phase_b.stderr
    phase_b_payload = json.loads(phase_b.stdout)
    assert phase_b_payload["phase"] == "b"
    assert phase_b_payload["problem_id"] == phase_a_payload["problem_id"]
    assert phase_b_payload["tenant_id"] == _TENANT
    assert phase_b_payload["pid"] != phase_a_payload["pid"]

    cleanup = _run_worker(
        "cleanup",
        env=env,
        args=["--tenant-id", _TENANT, "--tenant-id", _OTHER_TENANT],
    )
    assert cleanup.returncode == 0, cleanup.stderr
