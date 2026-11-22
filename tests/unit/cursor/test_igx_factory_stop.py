"""Tests for the Cursor state-file stop hook (real subprocess, temp git repos)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / "scripts/hooks/igx_factory_stop.py"
spec = importlib.util.spec_from_file_location("igx_factory_for_stop_tests", ROOT / "scripts/cursor/igx_factory.py")
assert spec and spec.loader
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


def mk_step(status: str = "pending", deps: list[str] | None = None, background: bool = False) -> dict[str, Any]:
    return {
        "title": "T",
        "deps": deps or [],
        "agent": "igx-implementer",
        "status": status,
        "background": background,
        "corrections": 0,
        "relaunches": 0,
        "park": None,
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def make_state(root: Path, steps: dict[str, dict[str, Any]], owner: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "plan_id": "p",
        "root": str(root),
        "owner_conversation": owner,
        "stall": {"signature": None, "count": 0},
        "steps": steps,
    }
    factory.save_state(state)
    return state


def fire(
    payload: Any, env_extra: dict[str, str] | None = None, raw: str | None = None, cwd: Path | None = None
) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k != "IGX_HOOKS"}
    env.update(env_extra or {})
    data = raw if raw is not None else json.dumps(payload)
    done = subprocess.run(
        [sys.executable, str(HOOK)], input=data, text=True, capture_output=True, env=env, cwd=cwd, check=False
    )
    return done.returncode, done.stdout


def payload(root: Path, conv: str = "c1", gen: str = "g1", loop: int = 0, status: str = "completed") -> dict[str, Any]:
    return {
        "conversation_id": conv,
        "generation_id": gen,
        "hook_event_name": "stop",
        "loop_count": loop,
        "status": status,
        "workspace_roots": [str(root)],
    }


def stored(root: Path) -> dict[str, Any]:
    state = factory.load_state(root, "p")
    assert state is not None
    return state


def test_no_active_run_allows(repo: Path) -> None:
    assert fire(payload(repo)) == (0, "")


def test_completed_run_allows(repo: Path) -> None:
    make_state(repo, {"S1": mk_step("integrated")}, owner="c1")
    assert fire(payload(repo)) == (0, "")


def test_non_completed_status_allows(repo: Path) -> None:
    make_state(repo, {"S1": mk_step()}, owner="c1")
    for status in ("aborted", "error"):
        assert fire(payload(repo, status=status)) == (0, "")


def test_claims_unowned_run_and_continues(repo: Path) -> None:
    make_state(repo, {"S1": mk_step(), "S2": mk_step(), "S3": mk_step("received")})
    code, out = fire(payload(repo, conv="mine"))
    assert code == 0
    text = json.loads(out)["followup_message"]
    assert stored(repo)["owner_conversation"] == "mine"
    assert text.startswith("FACTORY CONTINUE: run python scripts/cursor/igx_factory.py next")
    assert "READY: S1, S2" in text and "PENDING: S3 (review)" in text
    assert list(json.loads(out)) == ["followup_message"]


def test_other_owner_allows(repo: Path) -> None:
    make_state(repo, {"S1": mk_step()}, owner="someone-else")
    assert fire(payload(repo, conv="mine")) == (0, "")
    assert stored(repo)["owner_conversation"] == "someone-else"


def test_background_wait_allows_but_ready_continues(repo: Path) -> None:
    make_state(repo, {"S1": mk_step("running", background=True), "S2": mk_step("pending", deps=["S1"])}, owner="c1")
    assert fire(payload(repo)) == (0, "")
    make_state(repo, {"S1": mk_step("running", background=True), "S2": mk_step()}, owner="c1")
    assert "READY: S2" in json.loads(fire(payload(repo))[1])["followup_message"]


def test_stall_after_three_identical_signatures_and_reset(repo: Path) -> None:
    make_state(repo, {"S1": mk_step()}, owner="c1")
    assert fire(payload(repo, loop=0))[1]
    assert fire(payload(repo, loop=1))[1]
    assert fire(payload(repo, loop=2)) == (0, "")
    st = stored(repo)["stall"]
    assert st["stalled"] is True and st["count"] == 3
    # progress changes the signature: counter resets and the hook continues again
    state = stored(repo)
    state["steps"]["S2"] = mk_step()
    factory.save_state(state)
    assert fire(payload(repo, loop=3))[1]
    st = stored(repo)["stall"]
    assert st["count"] == 1 and st["stalled"] is False


def test_replay_repeats_output_without_counting(repo: Path) -> None:
    make_state(repo, {"S1": mk_step()}, owner="c1")
    first = fire(payload(repo, loop=0))
    again = fire(payload(repo, loop=0))
    assert first == again and first[1]
    assert stored(repo)["stall"]["count"] == 1


def test_kill_switch_and_malformed_input_fail_open(repo: Path) -> None:
    make_state(repo, {"S1": mk_step()}, owner="c1")
    assert fire(payload(repo), env_extra={"IGX_HOOKS": "off"}) == (0, "")
    assert fire(None, raw="not json") == (0, "")
    assert fire(None, raw="[1, 2]") == (0, "")
    assert fire(None, raw="") == (0, "")
    assert fire({"status": "completed", "conversation_id": "c1", "workspace_roots": ["Z:/nope/missing"]}) == (0, "")


def test_cwd_key_takes_precedence_over_workspace_roots(repo: Path, tmp_path: Path) -> None:
    make_state(repo, {"S1": mk_step()}, owner="c1")
    data = payload(tmp_path / "elsewhere")
    data["cwd"] = str(repo)
    assert fire(data)[1]


def test_root_discovery_from_subdirectory_and_pointer(repo: Path, tmp_path: Path) -> None:
    make_state(repo, {"S1": mk_step()}, owner="c1")
    sub = repo / "pkg" / "deep"
    sub.mkdir(parents=True)
    assert fire(payload(sub))[1]
    other = tmp_path / "wt"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    assert fire(payload(other)) == (0, "")  # no state there, no pointer yet
    pointer = other / ".tmp" / "factory_root"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(str(repo), encoding="utf-8")
    assert fire(payload(other))[1]
