# © Artur Czarnecki. All rights reserved.

"""Shared helpers for MEM-FINAL-AUDIT-5G SQLite durable restart qualification."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER_MODULE = "tests.integration.memory.e2e.mem_final_audit_5g_restart_worker"


def write_fixture(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_restart_worker(
    domain: str,
    *,
    db_path: Path,
    fixture_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            _WORKER_MODULE,
            domain,
            "--db",
            str(db_path),
            "--fixture",
            str(fixture_path),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_worker_json(stdout: str) -> dict[str, object]:
    line = stdout.strip().splitlines()[-1]
    return json.loads(line)
