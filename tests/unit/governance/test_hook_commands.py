# © Artur Czarnecki. All rights reserved.
"""The settings.json hook commands must work without CLAUDE_PROJECT_DIR (Cursor runs them without it)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _commands() -> list[str]:
    hooks = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))["hooks"]
    return [h["command"] for entries in hooks.values() for entry in entries for h in entry["hooks"]]


def _run(command: str, cwd: Path, payload: dict, tmp: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_PROJECT_DIR", "IGX_HOOKS")}
    env["IGX_HOOK_STATE_DIR"] = str(tmp / "state")
    return subprocess.run(
        command, shell=True, input=json.dumps(payload), capture_output=True, text=True, cwd=cwd, env=env, check=False
    )


def test_no_command_depends_on_claude_project_dir_expansion() -> None:
    assert all("$CLAUDE_PROJECT_DIR" not in c and "%CLAUDE_PROJECT_DIR%" not in c for c in _commands())


@pytest.mark.parametrize("subdir", ["", "docs"])
def test_read_hook_denies_large_file_from_any_subdirectory(tmp_path: Path, subdir: str) -> None:
    big = tmp_path / "big.md"
    big.write_text("# T\n" + "x" * 100_000)
    command = next(c for c in _commands() if "igx_read_guard.py" in c and not c.endswith("epoch"))
    payload = {"session_id": "s", "tool_name": "Read", "tool_input": {"file_path": str(big)}}
    done = _run(command, ROOT / subdir, payload, tmp_path)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_every_hook_tolerates_a_payload_without_a_path(tmp_path: Path) -> None:
    for command in _commands():
        done = _run(command, tmp_path, {"session_id": "s", "tool_name": "Read"}, tmp_path)
        assert done.returncode == 0, (command, done.stderr)
        assert done.stdout.strip() == ""
