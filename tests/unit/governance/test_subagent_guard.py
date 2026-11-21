# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / "scripts" / "hooks" / "igx_subagent_guard.py"
HOOKS_JSON = ROOT / ".cursor" / "hooks.json"
ROLES = ("igx-architect", "igx-implementer", "igx-researcher", "igx-reviewer")


def _run(stdin: str, **env: str) -> dict:
    base = {k: v for k, v in os.environ.items() if k != "IGX_HOOKS"}
    done = subprocess.run(
        [sys.executable, str(HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        env={**base, **env},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)  # always valid JSON: failClosed blocks an empty stdout


def _start(subagent_type: object, **env: str) -> dict:
    return _run(json.dumps({"hook_event_name": "subagentStart", "subagent_type": subagent_type}), **env)


def test_general_purpose_is_denied_and_names_the_roles() -> None:
    out = _start("generalPurpose")
    assert out["permission"] == "deny"
    for role in ROLES:
        assert role in out["user_message"]


@pytest.mark.parametrize("subagent_type", [*ROLES, "explore", "general-purpose", "GeneralPurpose"])
def test_other_types_are_allowed(subagent_type: str) -> None:
    assert _start(subagent_type) == {"permission": "allow"}


@pytest.mark.parametrize("stdin", ["{}", "", "not json", "[]", json.dumps({"subagent_type": ""})])
def test_missing_type_is_denied_with_valid_json(stdin: str) -> None:
    assert _run(stdin)["permission"] == "deny"


def test_non_string_type_is_denied() -> None:
    assert _start(None)["permission"] == "deny"
    assert _start(["generalPurpose"])["permission"] == "deny"


def test_kill_switch_allows_every_spawn() -> None:
    assert _start("generalPurpose", IGX_HOOKS="off") == {"permission": "allow"}
    assert _run("not json", IGX_HOOKS="off") == {"permission": "allow"}


def test_hooks_json_registers_only_a_fail_closed_subagent_start() -> None:
    config = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    assert config["version"] == 1
    assert list(config["hooks"]) == ["subagentStart"]
    assert "stop" not in config["hooks"]
    (entry,) = config["hooks"]["subagentStart"]
    assert entry["failClosed"] is True
    assert entry["matcher"] == "^generalPurpose$"
    assert entry["command"] == "python scripts/hooks/igx_subagent_guard.py"
    assert (ROOT / "scripts" / "hooks" / "igx_subagent_guard.py").is_file()
