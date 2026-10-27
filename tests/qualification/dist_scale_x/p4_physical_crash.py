# © Artur Czarnecki. All rights reserved.

"""P4 dimension G crash matrix (WL-5 worker boundaries)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._p4_support import (
    new_run_id,
    qual_pg_schema,
    require_p4_pg_infra,
    resolve_pg_dsn,
)

pytestmark = [pytest.mark.qualification, pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_MODULE = "tests.qualification.dist_scale_x.p4_crash_worker"


def _kill_at_sync(
    *,
    boundary: str,
    expect_sync: str,
    dsn: str,
    schema: str,
) -> subprocess.Popen[bytes]:
    session = _REPO_ROOT / ".tmp" / "session" / f"p4-g-{boundary}-{new_run_id('g')}"
    session.mkdir(parents=True, exist_ok=True)
    sync = session / "sync"
    state = session / "state.json"
    proc = subprocess.Popen(
        [
            _PYTHON,
            "-m",
            _MODULE,
            "--dsn",
            dsn,
            "--schema",
            schema,
            "--tenant-id",
            "tenant-g",
            "--boundary",
            boundary,
            "--sync-file",
            str(sync),
            "--state-file",
            str(state),
        ],
        cwd=_REPO_ROOT,
    )
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        if sync.exists() and sync.read_text(encoding="utf-8") == expect_sync:
            proc.kill()
            proc.wait(timeout=10)
            return proc
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=10)
    raise AssertionError(f"sync {expect_sync} not reached for {boundary}")


@pytest.mark.parametrize(
    ("boundary", "expect_sync"),
    [
        ("G-01", "armed"),
        ("G-02", "after_f01"),
        ("G-03", "after_G-03"),
        ("G-04", "after_G-04"),
        ("G-05", "after_G-05"),
        ("G-06", "after_G-06"),
    ],
)
def test_p4_g_crash_boundary_reached(boundary: str, expect_sync: str) -> None:
    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(f"crash-{boundary.lower()}")
    proc = _kill_at_sync(boundary=boundary, expect_sync=expect_sync, dsn=dsn, schema=schema)
    assert proc.returncode != 0

