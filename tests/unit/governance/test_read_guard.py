# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[3] / "scripts" / "hooks" / "igx_read_guard.py"


def _run(tmp: Path, payload: dict, *args: str, env: dict[str, str] | None = None) -> dict:
    base = {k: v for k, v in os.environ.items() if k not in ("IGX_HOOKS", "IGX_ROLE")}
    full = {**base, "IGX_HOOK_STATE_DIR": str(tmp / "state"), **(env or {})}
    done = subprocess.run(
        [sys.executable, str(HOOK), *args],
        input=json.dumps({"session_id": "s1", **payload}),
        capture_output=True,
        text=True,
        env=full,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout) if done.stdout else {}


def _read(path: Path, **extra: int) -> dict:
    return {"tool_name": "Read", "tool_input": {"file_path": str(path), **extra}}


def _denied(out: dict) -> bool:
    return out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


@pytest.fixture
def big(tmp_path: Path) -> Path:
    path = tmp_path / "big.md"
    path.write_text("# Title\n" + "x" * 100 + "\n" + "\n".join(f"## H{i}" for i in range(50)) + "\n" * 5 + "y" * 90_000)
    return path


def test_large_unranged_read_denied_with_outline(tmp_path: Path, big: Path) -> None:
    out = _run(tmp_path, _read(big))
    assert _denied(out)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "1: # Title" in reason
    assert reason.count(": ## H") + reason.count(": # ") <= 30


def test_ranged_read_allowed(tmp_path: Path, big: Path) -> None:
    assert not _denied(_run(tmp_path, _read(big, offset=1, limit=20)))


def test_insist_allows_identical_retry(tmp_path: Path, big: Path) -> None:
    assert _denied(_run(tmp_path, _read(big)))
    assert not _denied(_run(tmp_path, _read(big)))


def test_repeat_denied_until_epoch_advances(tmp_path: Path) -> None:
    small = tmp_path / "small.txt"
    small.write_text("hello\n")
    assert not _denied(_run(tmp_path, _read(small)))
    assert _denied(_run(tmp_path, _read(small)))
    _run(tmp_path, {}, "epoch")
    assert not _denied(_run(tmp_path, _read(small)))


def test_changed_file_is_not_a_repeat(tmp_path: Path) -> None:
    small = tmp_path / "small.txt"
    small.write_text("one\n")
    assert not _denied(_run(tmp_path, _read(small)))
    small.write_text("one two three\n")
    assert not _denied(_run(tmp_path, _read(small)))


def test_auditor_exempt_from_repeat(tmp_path: Path) -> None:
    small = tmp_path / "small.txt"
    small.write_text("hello\n")
    env = {"IGX_ROLE": "auditor"}
    assert not _denied(_run(tmp_path, _read(small), env=env))
    assert not _denied(_run(tmp_path, _read(small), env=env))


def test_tool_results_always_large(tmp_path: Path) -> None:
    folder = tmp_path / "tool-results"
    folder.mkdir()
    tiny = folder / "out.txt"
    tiny.write_text("tiny\n")
    assert _denied(_run(tmp_path, _read(tiny)))


def test_kill_switch(tmp_path: Path, big: Path) -> None:
    assert not _denied(_run(tmp_path, _read(big), env={"IGX_HOOKS": "off"}))
