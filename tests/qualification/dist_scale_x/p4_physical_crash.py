# © Artur Czarnecki. All rights reserved.

"""P4 dimension G — staged crash matrix with post-kill PostgreSQL inspection."""

from __future__ import annotations

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
from tests.qualification.dist_scale_x.p4_pg_state_inspection import (
    assert_boundary_snapshot,
    inspect_p4_durable_state,
)
from tests.qualification.dist_scale_x.p4_staged_persistence import (
    BOUNDARY_SYNC,
    BoundaryId,
    handle_from_json,
)

pytestmark = [pytest.mark.qualification, pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_MODULE = "tests.qualification.dist_scale_x.p4_crash_worker"


def _run_crash_scenario(boundary: BoundaryId, schema: str) -> Path:
    dsn = resolve_pg_dsn()
    session = _REPO_ROOT / ".tmp" / "session" / f"p4-g-{boundary}-{new_run_id('g')}"
    session.mkdir(parents=True, exist_ok=True)
    sync = session / "sync"
    state = session / "state.json"
    expect_sync = BOUNDARY_SYNC[boundary]
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
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if sync.exists() and sync.read_text(encoding="utf-8") == expect_sync:
            proc.kill()
            proc.wait(timeout=15)
            assert proc.returncode != 0
            return state
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=15)
    raise AssertionError(f"sync {expect_sync} not reached for {boundary}")


@pytest.mark.parametrize("boundary", ["G-01", "G-02", "G-03", "G-04", "G-05", "G-06"])
def test_p4_g_crash_boundary_durable_state(boundary: BoundaryId) -> None:
    require_p4_pg_infra()
    schema = qual_pg_schema(f"crash-{boundary.lower()}-{new_run_id('g')}")
    state_path = _run_crash_scenario(boundary, schema)
    handle = handle_from_json(state_path.read_text(encoding="utf-8"))
    snap = inspect_p4_durable_state(
        dsn=handle.dsn,
        schema=handle.schema,
        tenant_id=handle.tenant_id,
        task_id=handle.task_id,
        run_id=handle.run_id,
        decision_id=handle.decision_id,
        decision_subject=handle.decision_subject,
        event_id=handle.event_id,
    )
    assert_boundary_snapshot(boundary, snap)
