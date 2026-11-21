# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[3] / "scripts" / "hooks" / "igx_stop_guard.py"
ACTIVE = Path(".tmp/session/autonomy/active-plan")


def _env(**extra: str) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if k not in ("IGX_HOOKS", "CLAUDE_PROJECT_DIR")}
    return {**base, **extra}


def _stop(root: Path, message: str, *, cwd: Path | None = None, **env: str) -> dict:
    payload = {"hook_event_name": "Stop", "cwd": str(cwd or root), "last_assistant_message": message}
    done = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=_env(**env),
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout) if done.stdout.strip() else {}


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK), *args], capture_output=True, text=True, cwd=root, env=_env(), check=False
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n- [x] step one\n- [ ] step two\n- [ ] step three\n", encoding="utf-8")
    assert _cli(tmp_path, "start", str(plan)).returncode == 0
    return tmp_path


def _blocked(out: dict) -> bool:
    return out.get("decision") == "block"


def test_no_active_plan_allows(tmp_path: Path) -> None:
    assert _stop(tmp_path, "Shall I continue?") == {}


def test_question_with_open_steps_is_blocked_and_names_next_step(root: Path) -> None:
    out = _stop(root, "Do you want me to continue?")
    assert _blocked(out)
    assert "step two" in out["reason"] and "ESCALATION:" in out["reason"]


def test_escalation_is_allowed(root: Path) -> None:
    assert _stop(root, "ESCALATION: missing access\n- need a token") == {}
    assert (root / ACTIVE).is_file()


def test_plan_complete_is_allowed_and_clears_active_plan(root: Path) -> None:
    assert _stop(root, "PLAN COMPLETE\nall done") == {}
    assert not (root / ACTIVE).exists()


def test_no_open_steps_allows_and_clears(root: Path) -> None:
    (root / "plan.md").write_text("- [x] a\n- [x] b\n", encoding="utf-8")
    assert _stop(root, "finished") == {}
    assert not (root / ACTIVE).exists()


def test_third_block_without_progress_is_allowed_once_with_warning(root: Path) -> None:
    assert _blocked(_stop(root, "question?"))
    assert _blocked(_stop(root, "question?"))
    third = _stop(root, "question?")
    assert not _blocked(third) and "igx stop guard" in third["systemMessage"]
    assert _blocked(_stop(root, "question?"))  # counter reset after the single allow


def test_checkbox_change_resets_the_counter(root: Path) -> None:
    assert _blocked(_stop(root, "q"))
    assert _blocked(_stop(root, "q"))
    (root / "plan.md").write_text("- [x] one\n- [x] two\n- [ ] three\n", encoding="utf-8")
    assert _blocked(_stop(root, "q"))
    assert _blocked(_stop(root, "q"))


def test_stop_hook_active_is_still_bounded_by_the_loop_limit(root: Path) -> None:
    payload = {"cwd": str(root), "stop_hook_active": True, "last_assistant_message": "q"}
    outs = []
    for _ in range(3):
        done = subprocess.run(
            [sys.executable, str(HOOK)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=_env(),
            check=False,
        )
        outs.append(json.loads(done.stdout) if done.stdout.strip() else {})
    assert _blocked(outs[0]) and _blocked(outs[1]) and not _blocked(outs[2])


def test_works_from_a_subdirectory(root: Path) -> None:
    sub = root / "docs"
    sub.mkdir()
    assert _blocked(_stop(root, "q", cwd=sub))


def test_kill_switch_and_fail_open(root: Path) -> None:
    assert _stop(root, "q", IGX_HOOKS="off") == {}
    (root / "plan.md").unlink()  # unreadable plan -> allow
    assert _stop(root, "q") == {}
    (root / ACTIVE).write_text("", encoding="utf-8")  # empty active-plan -> allow
    assert _stop(root, "q") == {}
    done = subprocess.run(
        [sys.executable, str(HOOK)], input="not json", capture_output=True, text=True, env=_env(), check=False
    )
    assert done.returncode == 0 and done.stdout == ""


def test_cli_start_validates_and_stop_clears(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    empty = tmp_path / "empty.md"
    empty.write_text("no steps here\n", encoding="utf-8")
    assert _cli(tmp_path, "start", str(empty)).returncode == 1
    assert _cli(tmp_path, "start", str(tmp_path / "missing.md")).returncode == 1
    plan = tmp_path / "p.md"
    plan.write_text("- [ ] only\n", encoding="utf-8")
    assert _cli(tmp_path, "start", str(plan)).returncode == 0
    assert "only" in _cli(tmp_path, "status").stdout
    assert _cli(tmp_path, "stop").returncode == 0
    assert "no active plan" in _cli(tmp_path, "status").stdout


def test_hook_is_fast(root: Path) -> None:
    import time

    start = time.perf_counter()
    _stop(root, "q")
    assert time.perf_counter() - start < 1.0  # generous: process start on a loaded Windows host
