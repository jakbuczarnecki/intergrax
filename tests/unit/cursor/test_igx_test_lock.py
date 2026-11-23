"""Tests for the cross-worktree pytest lock wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parents[3] / "scripts/cursor/igx_test_lock.py"


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(LOCK), *args], cwd=cwd, text=True, capture_output=True, check=False)


def test_runs_command_tees_log_and_returns_code(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    log = tmp_path / "out" / "t.log"
    done = run(tmp_path, "--log", str(log), "--", sys.executable, "-c", "print('hello'); raise SystemExit(3)")
    assert done.returncode == 3
    assert "hello" in done.stdout
    assert "hello" in log.read_text(encoding="utf-8")
    assert (tmp_path / ".tmp/factory/pytest.lock").is_file()


def test_requires_command(tmp_path: Path) -> None:
    assert run(tmp_path).returncode == 2
    assert run(tmp_path, "--").returncode == 2


def test_lock_keyed_on_main_checkout(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(main)], check=True)
    subprocess.run(["git", *ident, "commit", "-q", "--allow-empty", "-m", "i"], cwd=main, check=True)
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "w", str(wt)], cwd=main, check=True)
    assert run(wt, "--", sys.executable, "-c", "pass").returncode == 0
    assert (main / ".tmp/factory/pytest.lock").is_file()
    assert not (wt / ".tmp/factory/pytest.lock").exists()
