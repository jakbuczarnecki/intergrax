# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package F — durable external-effect admission across OS processes."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]
_CHILD = Path(__file__).with_name("_arch_freeze_f_idempotency_child.py")


def test_process_a_admit_crash_process_b_observes_unknown_without_second_backend_call(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "backend_calls.txt"
    env_key = f"idem-{uuid.uuid4().hex}"
    base_env = {
        **os.environ,
        "ARCH_FREEZE_F_DB": str(tmp_path / "idempotency.sqlite3"),
        "ARCH_FREEZE_F_KEY": env_key,
        "ARCH_FREEZE_F_COUNTER": str(counter),
        "ARCH_FREEZE_F_AUTHORITY": str(tmp_path / "authority.txt"),
    }
    counter.write_text("0", encoding="utf-8")
    proc_a = subprocess.run(
        [sys.executable, str(_CHILD)],
        cwd=str(_REPO),
        env={**base_env, "ARCH_FREEZE_F_MODE": "process_a_after_admit_crash"},
        check=True,
    )
    assert proc_a.returncode == 0
    assert int(counter.read_text(encoding="utf-8")) == 1
    proc_b = subprocess.run(
        [sys.executable, str(_CHILD)],
        cwd=str(_REPO),
        env={**base_env, "ARCH_FREEZE_F_MODE": "process_b_observe_uncertain"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "uncertain" in proc_b.stdout
    assert int(counter.read_text(encoding="utf-8")) == 1


def test_death_before_admit_allows_single_retry_in_second_process(tmp_path: Path) -> None:
    counter = tmp_path / "backend_calls.txt"
    env_key = f"idem-{uuid.uuid4().hex}"
    base_env = {
        **os.environ,
        "ARCH_FREEZE_F_DB": str(tmp_path / "idempotency.sqlite3"),
        "ARCH_FREEZE_F_KEY": env_key,
        "ARCH_FREEZE_F_COUNTER": str(counter),
        "ARCH_FREEZE_F_AUTHORITY": str(tmp_path / "authority.txt"),
    }
    counter.write_text("0", encoding="utf-8")
    proc_a = subprocess.run(
        [sys.executable, str(_CHILD)],
        cwd=str(_REPO),
        env={**base_env, "ARCH_FREEZE_F_MODE": "process_a_before_admit_exit"},
        check=False,
    )
    assert proc_a.returncode == 0
    assert int(counter.read_text(encoding="utf-8")) == 0
    proc_b = subprocess.run(
        [sys.executable, str(_CHILD)],
        cwd=str(_REPO),
        env={**base_env, "ARCH_FREEZE_F_MODE": "process_b_reconcile_and_call_once"},
        check=True,
    )
    assert proc_b.returncode == 0
    assert int(counter.read_text(encoding="utf-8")) == 1


def test_completed_replay_does_not_call_backend_again(tmp_path: Path) -> None:
    counter = tmp_path / "backend_calls.txt"
    env_key = f"idem-{uuid.uuid4().hex}"
    base_env = {
        **os.environ,
        "ARCH_FREEZE_F_DB": str(tmp_path / "idempotency.sqlite3"),
        "ARCH_FREEZE_F_KEY": env_key,
        "ARCH_FREEZE_F_COUNTER": str(counter),
        "ARCH_FREEZE_F_AUTHORITY": str(tmp_path / "authority.txt"),
    }
    counter.write_text("0", encoding="utf-8")
    subprocess.run(
        [sys.executable, str(_CHILD)],
        cwd=str(_REPO),
        env={**base_env, "ARCH_FREEZE_F_MODE": "process_a_complete"},
        check=True,
    )
    assert int(counter.read_text(encoding="utf-8")) == 1
    subprocess.run(
        [sys.executable, str(_CHILD)],
        cwd=str(_REPO),
        env={**base_env, "ARCH_FREEZE_F_MODE": "process_b_replay_completed"},
        check=True,
    )
    assert int(counter.read_text(encoding="utf-8")) == 1
