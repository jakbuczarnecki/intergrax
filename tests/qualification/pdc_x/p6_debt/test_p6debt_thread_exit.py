# © Artur Czarnecki. All rights reserved.

"""B-P6-I item 1 (R-05): the ``pdc_x`` pytest process exits by itself after its summary line."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest
from tests.qualification.pdc_x.conftest import (
    describe_thread,
    find_leaked_non_daemon_threads,
    live_non_daemon_threads,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
_XD03 = "tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py"
_EXIT_LIMIT_SECONDS = 120  # the file needs about 8 s; the limit is more than ten times that


def test_guard_detects_a_leaked_non_daemon_thread_by_name() -> None:
    release = threading.Event()
    before = live_non_daemon_threads()
    leaker = threading.Thread(target=release.wait, name="p6debt-leaker", daemon=False)
    leaker.start()
    try:
        leaked = find_leaked_non_daemon_threads(before, grace_seconds=0.2)
        assert leaked == [leaker]
        assert "p6debt-leaker" in describe_thread(leaker)
        assert "wait" in describe_thread(leaker)
    finally:
        release.set()
        leaker.join(timeout=5)
    assert not leaker.is_alive()


def test_guard_ignores_daemon_threads_and_threads_that_finished() -> None:
    release = threading.Event()
    before = live_non_daemon_threads()
    daemon = threading.Thread(target=release.wait, name="p6debt-daemon", daemon=True)
    finished = threading.Thread(target=lambda: None, name="p6debt-finished", daemon=False)
    daemon.start()
    finished.start()
    finished.join(timeout=5)
    try:
        assert find_leaked_non_daemon_threads(before, grace_seconds=0.2) == []
    finally:
        release.set()
        daemon.join(timeout=5)


def test_guard_waits_for_a_thread_that_ends_within_the_grace_window() -> None:
    before = live_non_daemon_threads()
    short = threading.Thread(target=lambda: threading.Event().wait(0.3), name="p6debt-short", daemon=False)
    short.start()
    try:
        assert find_leaked_non_daemon_threads(before, grace_seconds=5.0) == []
    finally:
        short.join(timeout=5)


def test_xd03_pytest_process_exits_by_itself_with_code_zero() -> None:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", _XD03, "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_EXIT_LIMIT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the pytest process of {_XD03} did not exit within {_EXIT_LIMIT_SECONDS} s (leaked thread)")
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
