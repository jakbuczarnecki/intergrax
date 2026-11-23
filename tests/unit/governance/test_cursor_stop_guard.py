# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[3] / "scripts" / "hooks" / "igx_stop_guard.py"
CURSOR = HOOK.with_name("igx_cursor_stop.py")
ACTIVE = Path(".tmp/session/autonomy/active-plan")
DIAG = Path(".tmp/session/autonomy/cursor-stop-last.json")
STATE = Path(".tmp/session/autonomy/cursor-stop-state.json")
START_CMD = "python scripts/hooks/igx_stop_guard.py start .tmp/x/plan.md"


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("IGX_HOOKS", "CLAUDE_PROJECT_DIR")}


def _cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK), *args], capture_output=True, text=True, cwd=cwd, env=_env(), check=False
    )


def _user(text: str) -> dict:
    return {"role": "user", "message": {"content": [{"type": "text", "text": text}]}}


def _assistant(text: str = "", *tools: dict) -> dict:
    content: list[dict] = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", "name": t["name"], "input": t["input"]} for t in tools]
    return {"role": "assistant", "message": {"content": content}}


SHELL_START = {"name": "Shell", "input": {"command": START_CMD}}
SPAWN = {"name": "Task", "input": {"prompt": "work", "run_in_background": True}}


class Session:
    """One Cursor conversation: transcript file plus a stop-event sender."""

    def __init__(self, root: Path, name: str = "c1") -> None:
        self.root = root
        self.conv = name
        self.path = root / f"{name}.jsonl"
        self.gen = 0
        self.loop = 0

    def write(self, *lines: dict | str) -> None:
        self.path.write_text(
            "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n", encoding="utf-8"
        )

    def turn(self, final: str, *, start: bool = True, extra: tuple[dict, ...] = ()) -> None:
        lines: list[dict] = [_user("run the plan")]
        if start:
            lines.append(_assistant("", SHELL_START))
        lines += list(extra)
        if final:
            lines.append(_assistant(final))
        lines.append({"type": "turn_ended", "status": "success"})
        self.write(*lines)

    def payload(self, **over: object) -> dict:
        self.gen += 1
        self.loop += 1
        base = {
            "conversation_id": self.conv,
            "generation_id": f"g{self.gen}",
            "hook_event_name": "stop",
            "loop_count": self.loop,
            "session_id": "s1",
            "status": "completed",
            "transcript_path": str(self.path),
            "workspace_roots": [str(self.root)],
        }
        return {**base, **over}

    def stop(self, **over: object) -> dict:
        return send(self.root, self.payload(**over))


def send(root: Path, payload: dict, cwd: Path | None = None) -> dict:
    other = cwd or root.parent / "elsewhere"
    other.mkdir(exist_ok=True)
    done = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=other,
        env=_env(),
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout) if done.stdout.strip() else {}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "plan.md").write_text("# Plan\n- [x] step one\n- [ ] step two\n- [ ] step three\n", encoding="utf-8")
    assert _cli(repo, "start", str(repo / "plan.md")).returncode == 0
    return repo


@pytest.fixture
def sess(root: Path) -> Session:
    return Session(root)


def _diag(root: Path) -> dict:
    return json.loads((root / DIAG).read_text(encoding="utf-8"))


def _state(root: Path) -> dict:
    return json.loads((root / STATE).read_text(encoding="utf-8"))


def _tick(root: Path, done: int) -> None:
    lines = [f"- [{'x' if i <= done else ' '}] step {i}" for i in range(1, 5)]
    (root / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_owner_is_claimed_by_start_marker_and_followup_is_only_followup_message(sess: Session, root: Path) -> None:
    sess.turn("I did a thing.")
    out = sess.stop()
    assert set(out) == {"followup_message"}
    assert "step two" in out["followup_message"]
    assert "ESCALATION:" in out["followup_message"] and "WAITING:" in out["followup_message"]
    assert "PLAN COMPLETE" in out["followup_message"]
    assert _state(root)["owner"] == "c1"
    assert _diag(root) == {
        "decision": "followup",
        "reason": "followup",
        "loop_count": 1,
        "status": "completed",
        "open": 2,
    }


def test_no_start_marker_allows_and_claims_nothing(sess: Session, root: Path) -> None:
    sess.turn("I did a thing.", start=False)
    assert sess.stop() == {}
    assert not (root / STATE).exists()
    assert _diag(root)["reason"] == "not_owner"
    assert (root / ACTIVE).is_file()


def test_other_conversation_is_allowed(sess: Session, root: Path) -> None:
    sess.turn("working")
    assert sess.stop()
    other = Session(root, "c2")
    other.turn("chatting", start=False)
    assert other.stop() == {}
    assert _diag(root)["reason"] == "other_conversation"
    # a second conversation that itself mentions the start command does not steal ownership
    other.turn("chatting again", start=True)
    assert other.stop() == {}
    assert _state(root)["owner"] == "c1"


def test_session_id_is_used_without_conversation_id(sess: Session) -> None:
    sess.turn("working")
    out = send(sess.root, {**sess.payload(), "conversation_id": ""})
    assert set(out) == {"followup_message"}


def test_duplicate_event_replays_without_counting(sess: Session, root: Path) -> None:
    sess.turn("working")
    payload = sess.payload()
    first = send(root, payload)
    assert send(root, payload) == first
    assert send(root, payload) == first
    assert _state(root)["n"] == 1 and _state(root)["followups"] == 1


def test_progress_resets_the_counter(sess: Session, root: Path) -> None:
    _tick(root, 1)
    sess.turn("working")
    assert "step 2" in sess.stop()["followup_message"]
    assert "No progress" in sess.stop()["followup_message"]
    _tick(root, 2)
    out = sess.stop()
    assert out["followup_message"].startswith("Open plan step: step 3")
    assert _state(root)["n"] == 1


def test_stall_sequence_followup_warning_allow_then_silent_until_progress(sess: Session, root: Path) -> None:
    _tick(root, 1)
    sess.turn("working")
    assert sess.stop()["followup_message"].startswith("Open plan step: step 2")
    warning = sess.stop()["followup_message"]
    assert "step 2" in warning and "plan.md" in warning and "ESCALATION: <type>" in warning
    assert "marks the plan stalled" in warning
    assert sess.stop() == {}
    assert _state(root)["stalled"] is True and _diag(root)["reason"] == "stall"
    assert sess.stop() == {}
    assert _diag(root)["reason"] == "stalled"
    _tick(root, 2)
    assert set(sess.stop()) == {"followup_message"}
    assert _state(root)["stalled"] is False
    assert (root / ACTIVE).is_file()


@pytest.mark.parametrize(
    "final",
    ["Blocked.\nESCALATION: missing access", "## Result\n> **ESCALATION:** forbidden", "-  ESCALATION: x"],
)
def test_escalation_marker_on_a_later_line_is_allowed(sess: Session, root: Path, final: str) -> None:
    sess.turn(final)
    assert sess.stop() == {}
    assert _diag(root)["reason"] == "escalation"
    assert (root / ACTIVE).is_file()


def test_escalation_keeps_counters(sess: Session, root: Path) -> None:
    sess.turn("working")
    sess.stop()
    before = _state(root)
    sess.turn("ESCALATION: need access")
    assert sess.stop() == {}
    after = _state(root)
    assert (after["n"], after["followups"]) == (before["n"], before["followups"])


def test_plan_complete_on_last_line_with_no_open_step_clears(sess: Session, root: Path) -> None:
    (root / "plan.md").write_text("- [x] a\n- [x] b\n", encoding="utf-8")
    sess.turn("All verified.\n\nPLAN COMPLETE: everything merged")
    assert sess.stop() == {}
    assert not (root / ACTIVE).exists()


def test_plan_complete_with_open_steps_is_premature_and_keeps_plan(sess: Session, root: Path) -> None:
    sess.turn("Summary.\nPLAN COMPLETE")
    out = sess.stop()
    assert "premature" in out["followup_message"] and "step two" in out["followup_message"]
    assert (root / ACTIVE).is_file()
    assert _diag(root)["reason"] == "premature_complete"


def test_waiting_with_background_spawn_is_allowed_without_counting(sess: Session, root: Path) -> None:
    sess.turn("Spawned.\nWAITING: task-1, task-2", extra=(_assistant("", SPAWN),))
    assert sess.stop() == {}
    assert sess.stop() == {}
    assert _diag(root)["reason"] == "waiting"
    assert "n" not in _state(root) and "followups" not in _state(root)


def test_background_text_on_a_non_tool_use_line_does_not_back_waiting(sess: Session, root: Path) -> None:
    quoted = _user('{"prompt": "work", "run_in_background": true}')
    result = {"role": "tool", "content": [{"type": "tool_result", "content": '"run_in_background": true'}]}
    sess.turn("Nothing running.\nWAITING: task-1", extra=(quoted, result))
    out = sess.stop()
    assert "background worker" in out["followup_message"]
    assert _diag(root)["reason"] == "waiting_unbacked"


def test_waiting_without_spawn_is_a_followup(sess: Session, root: Path) -> None:
    sess.turn("Nothing running.\nWAITING: task-1")
    out = sess.stop()
    assert "background worker" in out["followup_message"] and "step two" in out["followup_message"]
    assert _diag(root)["reason"] == "waiting_unbacked"


def test_stale_text_from_a_previous_turn_does_not_count(sess: Session, root: Path) -> None:
    sess.write(
        _user("start"),
        _assistant("", SHELL_START),
        _assistant("ESCALATION: old problem"),
        _user("carry on"),
        _assistant("doing more"),
        {"type": "turn_ended", "status": "success"},
    )
    assert set(sess.stop()) == {"followup_message"}
    # a turn without any assistant text after the last user line allows
    sess.write(_user("start"), _assistant("", SHELL_START), _assistant("earlier text"), _user("again"))
    assert sess.stop() == {}
    assert _diag(root)["reason"] == "no_text"


@pytest.mark.parametrize("over", [{"status": "aborted"}, {"status": "error"}, {"status": None}])
def test_non_completed_status_is_allowed(sess: Session, root: Path, over: dict) -> None:
    sess.turn("working")
    payload = sess.payload(**over)
    if over["status"] is None:
        del payload["status"]
    assert send(root, payload) == {}
    assert _diag(root)["reason"] == "status"
    assert (root / ACTIVE).is_file()


def test_missing_or_malformed_transcript_is_allowed(sess: Session, root: Path) -> None:
    assert sess.stop() == {}  # no file yet
    assert _diag(root)["reason"] == "no_transcript"
    assert send(root, sess.payload(transcript_path="")) == {}
    sess.write("not json", "[1, 2]", "{broken")
    assert sess.stop() == {}
    assert (root / ACTIVE).is_file()


def test_followup_cap_marks_stalled(sess: Session, root: Path) -> None:
    (root / "plan.md").write_text("- [ ] a\n- [ ] b\n", encoding="utf-8")
    assert _cli(root, "start", str(root / "plan.md")).returncode == 0
    sess.turn("working")
    for i in range(10):  # cap = max(10, 3 * 2); each plan change is progress, so no per-signature stall
        (root / "plan.md").write_text("- [x] a\n- [ ] b\n" if i % 2 == 0 else "- [ ] a\n- [ ] b\n", encoding="utf-8")
        assert set(sess.stop()) == {"followup_message"}, i
    (root / "plan.md").write_text("- [x] a\n- [ ] b\n", encoding="utf-8")
    assert sess.stop() == {}
    assert _diag(root)["reason"] == "cap" and _state(root)["stalled"] is True


def test_new_start_resets_state_and_owner(sess: Session, root: Path) -> None:
    sess.turn("working")
    sess.stop()
    assert _cli(root, "start", str(root / "plan.md")).returncode == 0
    other = Session(root, "c2")
    other.turn("fresh owner")
    assert set(other.stop()) == {"followup_message"}
    assert _state(root)["owner"] == "c2"


def test_cursor_plan_file_with_frontmatter_todos(sess: Session, root: Path) -> None:
    plan = root / "cursor.plan.md"
    plan.write_text(
        "---\nname: demo\ntodos:\n  - id: fetch-repin\n    content: git fetch and confirm ...\n    status: pending\n"
        '  - id: other\n    content: "quoted content: with colon"\n    status: completed\n'
        "isProject: false\n---\n# Body\nno checkboxes here\n",
        encoding="utf-8",
    )
    started = _cli(root, "start", str(plan))
    assert started.returncode == 0 and "1 open of 2" in started.stdout
    sess.turn("working")
    assert "git fetch and confirm ..." in sess.stop()["followup_message"]


def test_frontmatter_todos_parser() -> None:
    spec = importlib.util.spec_from_file_location("igx_cursor_stop_t", CURSOR)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = (
        "---\ntodos:\n  - id: a\n    content: 'single: quoted'\n    status: in_progress\n"
        "  - id: b\n    content: done one\n    status: cancelled\n  - id: c\n    content: no status\n"
        "other: 1\n---\n- [ ] ignored by the parser\n"
    )
    assert mod.frontmatter_todos(text) == [(False, "single: quoted"), (False, "done one"), (False, "no status")]
    assert mod.frontmatter_todos("# no frontmatter\n- [ ] x\n") == []
    assert mod.frontmatter_todos("---\nname: x\n---\n") == []



def test_cursor_cancelled_todo_does_not_complete_plan(sess: Session, root: Path) -> None:
    plan = root / "cancelled.plan.md"
    plan.write_text(
        "---\ntodos:\n  - id: cancelled-task\n    content: Review the cancelled worker and reschedule\n    status: cancelled\n---\n",
        encoding="utf-8",
    )
    assert _cli(root, "start", str(plan)).returncode == 0
    sess.turn("PLAN COMPLETE")
    result = sess.stop()
    assert "premature" in result["followup_message"]
    assert "Review the cancelled worker" in result["followup_message"]
    assert (root / ACTIVE).is_file()


def test_checkbox_plan_still_parses_as_before(root: Path) -> None:
    done = _cli(root, "status")
    assert "next open step: step two" in done.stdout
    (root / "mixed.md").write_text(
        "---\ntodos:\n  - id: a\n    content: from todos\n    status: pending\n---\n- [ ] from box\n",
        encoding="utf-8",
    )
    assert _cli(root, "start", str(root / "mixed.md")).returncode == 0
    assert "from box" in _cli(root, "status").stdout


def test_runpy_invocation_with_native_payload(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p).resolve() != HOOK.parent.resolve()])
    monkeypatch.delitem(sys.modules, "igx_cursor_stop", raising=False)
    ns = runpy.run_path(str(HOOK), run_name="igx_stop_guard_runpy")
    sess = Session(root)
    sess.turn("working")
    out = ns["decide"](root, sess.payload())
    assert out is not None and set(out) == {"followup_message"}
    monkeypatch.delitem(sys.modules, "igx_cursor_stop", raising=False)


def test_owner_is_not_claimed_by_start_phrase_in_plain_text(sess: Session, root: Path) -> None:
    sess.write(
        _user(f"please run {START_CMD}"),
        _assistant(f"I read the skill: {START_CMD}"),
        _assistant("done for now"),
    )
    assert sess.stop() == {}
    assert _diag(root)["reason"] == "not_owner"
    assert not (root / STATE).exists()
    sess.write(_user("go"), _assistant("", SHELL_START), _assistant("working"))
    assert set(sess.stop()) == {"followup_message"}
    assert _state(root)["owner"] == "c1"


def test_escalation_placeholder_is_an_ordinary_stop(sess: Session, root: Path) -> None:
    sess.turn("Rules say: end with\nESCALATION: <type>")
    out = sess.stop()
    assert set(out) == {"followup_message"}
    assert _diag(root)["reason"] == "followup"


def test_waiting_with_background_false_is_a_followup(sess: Session, root: Path) -> None:
    fg = {"name": "Task", "input": {"prompt": "work", "run_in_background": False}}
    sess.turn("Spawned.\nWAITING: task-1", extra=(_assistant("", fg),))
    out = sess.stop()
    assert "background worker" in out["followup_message"]
    assert _diag(root)["reason"] == "waiting_unbacked"


def test_waiting_and_plan_complete_together_is_premature(sess: Session, root: Path) -> None:
    sess.turn("WAITING: task-1\nPLAN COMPLETE", extra=(_assistant("", SPAWN),))
    out = sess.stop()
    assert "premature" in out["followup_message"]
    assert _diag(root)["reason"] == "premature_complete"
    assert (root / ACTIVE).is_file()


def test_plan_file_deleted_after_start_is_allowed(sess: Session, root: Path) -> None:
    sess.turn("working")
    (root / "plan.md").unlink()
    assert sess.stop() == {}


def test_claude_payload_with_cwd_still_blocks(root: Path) -> None:
    payload = {"cwd": str(root), "last_assistant_message": "still working", "hook_event_name": "Stop"}
    out = send(root, payload)
    assert out["decision"] == "block" and "step two" in out["reason"]
    assert "followup_message" not in out


def test_root_discovery_from_uri_style_workspace_root(sess: Session, root: Path) -> None:
    sess.turn("working")
    assert set(sess.stop(workspace_roots=[root.as_uri()])) == {"followup_message"}


def test_root_discovery_tries_later_workspace_roots(sess: Session, root: Path) -> None:
    empty = root.parent / "empty-root"
    empty.mkdir()
    sess.turn("working")
    assert set(sess.stop(workspace_roots=[str(empty), str(root)])) == {"followup_message"}


def test_root_discovery_falls_back_to_process_cwd(sess: Session, root: Path) -> None:
    empty = root.parent / "empty-root"
    empty.mkdir()
    sess.turn("working")
    out = send(root, sess.payload(workspace_roots=[str(empty)]), cwd=root)
    assert set(out) == {"followup_message"}


def test_root_discovery_without_any_match_allows(sess: Session, root: Path) -> None:
    empty = root.parent / "empty-root"
    empty.mkdir()
    sess.turn("working")
    assert sess.stop(workspace_roots=[str(empty)]) == {}


def _parser() -> object:
    spec = importlib.util.spec_from_file_location("igx_cursor_stop_p", CURSOR)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frontmatter_bom_crlf_block_scalars_and_nested_keys() -> None:
    mod = _parser()
    text = (
        "\ufeff---\r\ntodos:\r\n  - id: a\r\n    content: >-\r\n      first part\r\n      second part\r\n"
        "    status: pending\r\n  - id: b\r\n    content: |\r\n      literal one\r\n      literal two\r\n"
        "    status: completed\r\n    deps:\r\n      - id: nested\r\n        content: not a todo\r\n"
        "  - id: c\r\n    content: plain\r\nname: x\r\n---\r\n"
    )
    assert mod.frontmatter_todos(text) == [  # type: ignore[attr-defined]
        (False, "first part second part"),
        (True, "literal one literal two"),
        (False, "plain"),
    ]


def _decide(mod: object, sess: Session, **over: object) -> dict | None:
    steps = [(True, "step one"), (False, "step two"), (False, "step three")]
    result, _clear = mod.decide_cursor(  # type: ignore[attr-defined]
        sess.root, sess.payload(**over), sess.root / "plan.md", steps, sess.root / ACTIVE, 3
    )
    return result


def _filler(count: int = 40) -> list[dict]:
    return [_assistant("filler " + "x" * 60) for _ in range(count)]


@pytest.fixture
def small_tail(monkeypatch: pytest.MonkeyPatch) -> object:
    mod = _parser()
    monkeypatch.setattr(mod, "_SCAN_TAIL", 512)
    return mod


def test_start_tool_use_beyond_the_tail_is_still_claimed(sess: Session, root: Path, small_tail: object) -> None:
    sess.write(_user("run"), _assistant("", SHELL_START), *_filler(), _assistant("progress"))
    assert sess.path.stat().st_size > 512 * 4
    result = _decide(small_tail, sess)
    assert result and "followup_message" in result
    assert _state(root)["owner"] == "c1"


def test_background_spawn_beyond_the_tail_still_honours_waiting(sess: Session, root: Path, small_tail: object) -> None:
    sess.write(
        _user("run"),
        _assistant("", SHELL_START),
        _assistant("", SPAWN),
        *_filler(),
        _assistant("Spawned.\nWAITING: task-1"),
    )
    assert _decide(small_tail, sess) is None
    assert _diag(root)["reason"] == "waiting"


def test_aborted_first_stop_claims_ownership_and_next_completed_stop_follows_up(
    sess: Session, root: Path, small_tail: object
) -> None:
    sess.write(_user("run"), _assistant("", SHELL_START), *_filler(), _assistant("working"))
    assert _decide(small_tail, sess, status="aborted") is None
    assert _diag(root)["reason"] == "status"
    assert _state(root)["owner"] == "c1"
    result = _decide(small_tail, sess)
    assert result and "followup_message" in result
    assert _diag(root)["reason"] == "followup"


def test_plain_text_mention_beyond_the_tail_does_not_claim(sess: Session, root: Path, small_tail: object) -> None:
    sess.write(_user(f"please run {START_CMD}"), _assistant(f"read: {START_CMD}"), *_filler(), _assistant("done"))
    assert _decide(small_tail, sess) is None
    assert _diag(root)["reason"] == "not_owner"
    assert not (root / STATE).exists()
