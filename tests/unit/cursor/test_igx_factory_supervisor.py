"""Tests for the headless factory supervisor (fake agent script, temp git repo) and cli.json."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/cursor/igx_factory_supervisor.py"
spec = importlib.util.spec_from_file_location("igx_factory_supervisor_under_test", SCRIPT)
assert spec and spec.loader
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)

FAKE_AGENT = """
import json, os, sys
from pathlib import Path

state_path = Path(os.environ["FAKE_STATE"])
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
mode = os.environ["FAKE_MODE"]
state = json.loads(state_path.read_text(encoding="utf-8"))
steps = state["steps"]
if mode == "complete":
    for s in steps.values():
        s["status"] = "integrated"
elif mode == "parked":
    for s in steps.values():
        s["status"] = "parked"
        s["park"] = {"type": "missing access", "reason": "r", "options": "o", "recommendation": "x"}
elif mode == "progress":
    for s in steps.values():
        if s["status"] == "pending":
            s["status"] = "integrated"
            break
state_path.write_text(json.dumps(state), encoding="utf-8")
print(json.dumps({"session_id": "chat-1", "result": "ok"}))
"""


def step(status: str = "pending") -> dict[str, Any]:
    return {
        "title": "T",
        "deps": [],
        "agent": "igx-implementer",
        "status": status,
        "background": False,
        "corrections": 0,
        "relaunches": 0,
        "park": None,
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "i"],
        cwd=repo,
        check=True,
    )
    plan = repo / "p.md"
    plan.write_text("# p\n", encoding="utf-8")
    agent = tmp_path / "fake_agent.py"
    agent.write_text(FAKE_AGENT, encoding="utf-8")
    state_file = repo / ".tmp" / "factory" / "p" / "state.json"
    log = tmp_path / "agent.log"
    monkeypatch.setenv("FAKE_STATE", str(state_file))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.chdir(repo)
    return {"repo": repo, "plan": plan, "agent": agent, "state_file": state_file, "log": log}


def seed(env: dict[str, Any], *statuses: str) -> None:
    state = {
        "plan_id": "p",
        "root": str(env["repo"]),
        "int_worktree": str(env["repo"]),
        "int_branch": "factory/p/int",
        "owner_conversation": None,
        "stall": {"signature": None, "count": 0},
        "steps": {f"S{i + 1}": step(s) for i, s in enumerate(statuses)},
    }
    env["state_file"].parent.mkdir(parents=True, exist_ok=True)
    env["state_file"].write_text(json.dumps(state), encoding="utf-8")


def run(env: dict[str, Any], *extra: str, agent: str | None = None) -> int:
    return sup.main(
        [
            "--plan",
            str(env["plan"]),
            "--model",
            "m1",
            "--agent-bin",
            agent or str(env["agent"]),
            "--workspace",
            str(env["repo"]),
            *extra,
        ]
    )


def calls(env: dict[str, Any]) -> list[list[str]]:
    return [json.loads(line) for line in env["log"].read_text(encoding="utf-8").splitlines()]


def test_complete_exits_zero_and_prints_complete_output(env: dict[str, Any], monkeypatch, capsys) -> None:
    seed(env, "pending", "pending")
    monkeypatch.setenv("FAKE_MODE", "complete")
    assert run(env) == 0
    assert "PLAN COMPLETE" in capsys.readouterr().out
    argv = calls(env)[0]
    assert argv[:5] == ["-p", "--model", "m1", "--output-format", "json"] and "--resume" not in argv
    assert "FACTORY CONTINUE" in argv[-1]
    log = (env["state_file"].parent / "supervisor.log").read_text(encoding="utf-8").splitlines()
    record = json.loads(log[0])
    assert record["iteration"] == 1 and record["exit_code"] == 0 and record["next"]["done"] is True


def test_parked_exits_three(env: dict[str, Any], monkeypatch, capsys) -> None:
    seed(env, "pending")
    monkeypatch.setenv("FAKE_MODE", "parked")
    assert run(env) == 3
    assert "ESCALATION" in capsys.readouterr().out


def test_stalled_exits_four(env: dict[str, Any], monkeypatch) -> None:
    seed(env, "pending")
    monkeypatch.setenv("FAKE_MODE", "noop")
    assert run(env, "--max-iterations", "10") == 4
    assert len(calls(env)) == 3


def test_max_iterations_exits_five_and_resume_id_is_passed(env: dict[str, Any], monkeypatch) -> None:
    seed(env, "pending", "pending", "pending", "pending")
    monkeypatch.setenv("FAKE_MODE", "progress")
    assert run(env, "--max-iterations", "2") == 5
    first, second = calls(env)
    assert "--resume" not in first
    assert second[second.index("--resume") + 1] == "chat-1"


def test_missing_agent_binary_exits_two(env: dict[str, Any]) -> None:
    assert run(env, agent="definitely-not-an-agent-binary") == 2


def test_first_prompt_starts_the_run_when_no_state(env: dict[str, Any], monkeypatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "noop")
    seed(env, "pending")
    env["state_file"].unlink()
    # fake agent cannot read a missing state: it fails, supervisor still passes the start prompt
    assert run(env, "--max-iterations", "1") == 5
    assert "igx_factory.py start" in calls(env)[0][-1] and "/igx-native-run-plan" in calls(env)[0][-1]


def test_extract_chat_id_is_tolerant() -> None:
    assert sup.extract_chat_id('{"chatId": "a"}') == "a"
    assert sup.extract_chat_id('noise\n{"chat_id": "b"}\n') == "b"
    assert sup.extract_chat_id("not json") is None
    assert sup.extract_chat_id('{"x": 1}') is None


def test_help_mentions_powershell(capsys) -> None:
    with pytest.raises(SystemExit):
        sup.main(["--help"])
    assert "PowerShell" in capsys.readouterr().out


def test_cli_json_is_valid_and_denies_push() -> None:
    cfg = json.loads((ROOT / ".cursor/cli.json").read_text(encoding="utf-8"))
    assert "Shell(git push)" in cfg["permissions"]["deny"]
    for token in ("Shell(git reset)", "Shell(gh)", "Shell(curl)"):
        assert token in cfg["permissions"]["deny"]
    assert "Shell(git push)" not in cfg["permissions"]["allow"]
