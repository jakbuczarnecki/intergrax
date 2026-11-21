# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[3] / "scripts" / "hooks" / "igx_bash_guard.py"


def _run(command: str, cwd: Path, **env: str) -> dict:
    base = {k: v for k, v in os.environ.items() if k != "IGX_HOOKS"}
    done = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd)}),
        capture_output=True,
        text=True,
        env={**base, **env},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout) if done.stdout else {}


def _denied(out: dict) -> bool:
    return out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


def test_cat_large_file_denied(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 90_000)
    assert _denied(_run("cat big.txt", tmp_path))
    assert _denied(_run("git status && Get-Content big.txt", tmp_path))


def test_cat_small_file_allowed(tmp_path: Path) -> None:
    (tmp_path / "small.txt").write_text("x")
    assert not _denied(_run("cat small.txt", tmp_path))


def test_sed_wide_range_denied(tmp_path: Path) -> None:
    assert _denied(_run("sed -n '1,700p' file.py", tmp_path))
    assert not _denied(_run("sed -n '1,600p' file.py", tmp_path))


def test_other_commands_untouched(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 90_000)
    assert not _denied(_run("grep -n foo big.txt", tmp_path))
    assert not _denied(_run("ls -la", tmp_path))


def test_kill_switch(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 90_000)
    assert not _denied(_run("cat big.txt", tmp_path, IGX_HOOKS="off"))
