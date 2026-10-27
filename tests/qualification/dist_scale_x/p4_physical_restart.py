# © Artur Czarnecki. All rights reserved.

"""P4 dimension H — crash at G-06 boundary and recovery in new PID."""

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
from tests.qualification.dist_scale_x.p4_staged_persistence import BOUNDARY_SYNC

pytestmark = [pytest.mark.qualification, pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_H_MODULE = "tests.qualification.dist_scale_x.p4_h_crash_recovery_worker"


def test_p4_h_sigkill_then_recovery_continuation() -> None:
    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(new_run_id("p4-h-kill"))
    session = _REPO_ROOT / ".tmp" / "session" / f"p4-h-kill-{schema}"
    session.mkdir(parents=True, exist_ok=True)
    sync = session / "sync"
    state = session / "state.json"
    recover_out = session / "recover.json"
    writer = subprocess.Popen(
        [
            _PYTHON,
            "-m",
            _H_MODULE,
            "--dsn",
            dsn,
            "--schema",
            schema,
            "--mode",
            "write",
            "--sync-file",
            str(sync),
            "--state-file",
            str(state),
        ],
        cwd=_REPO_ROOT,
    )
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if sync.exists() and sync.read_text(encoding="utf-8") == BOUNDARY_SYNC["G-06"]:
            writer.kill()
            writer.wait(timeout=15)
            break
        time.sleep(0.05)
    else:
        writer.kill()
        writer.wait(timeout=15)
        raise AssertionError("H write boundary not reached")
    assert writer.returncode != 0
    recover = subprocess.run(
        [
            _PYTHON,
            "-m",
            _H_MODULE,
            "--dsn",
            dsn,
            "--schema",
            schema,
            "--mode",
            "recover",
            "--state-file",
            str(state),
            "--out",
            str(recover_out),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert recover.returncode == 0, recover.stdout + recover.stderr
    payload = json.loads(recover_out.read_text(encoding="utf-8"))
    assert payload["before_revision"] == 1
    assert payload["after_revision"] == 2
    assert payload["f03_present"] is True
    assert payload["f07_event_present"] is True
    assert payload["f09_present"] is True
