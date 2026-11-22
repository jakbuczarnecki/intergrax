"""Tests for the Cursor software factory state machine (temp git repos, no network)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/cursor/igx_factory.py"
spec = importlib.util.spec_from_file_location("igx_factory_under_test", SCRIPT)
assert spec and spec.loader
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)

T1 = "governance/tickets/T-IGX-900-A.md"


def sh(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, text=True, capture_output=True, check=True
    )
    return done.stdout.strip()


def plan_text(*steps: str) -> str:
    return "# Plan\n\n" + "\n".join(steps) + "\n"


def step(sid: str, deps: str = "", write: str = "src/a/**", ticket: str = T1, tests: str = "tests/a") -> str:
    lines = [f"- [ ] {sid} Title {sid}"]
    if deps:
        lines.append(f"  - deps: {deps}")
    lines += ["  - agent: igx-implementer", f"  - ticket: {ticket}", f"  - write: {write}", f"  - tests: {tests}"]
    return "\n".join(lines)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "main"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    (root / "governance/tickets").mkdir(parents=True)
    (root / T1).write_text("# T\n\nApproved: operator @ now\n", encoding="utf-8")
    (root / "src/a").mkdir(parents=True)
    (root / "src/a/base.txt").write_text("base\n", encoding="utf-8")
    (root / ".gitignore").write_text(".tmp/\n", encoding="utf-8")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "init")
    monkeypatch.chdir(root)
    return root


def write_plan(root: Path, text: str, name: str = "myplan") -> Path:
    path = root / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    capsys.readouterr()
    code = factory.main(list(argv))
    return code, capsys.readouterr().out


def jrun(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code, out = run(capsys, *argv)
    return code, json.loads(out)


def worker_commit(st: dict, fname: str = "src/a/f.txt", message: str | None = None, body: str = "x\n") -> None:
    wt = Path(st["worktree"])
    (wt / fname).parent.mkdir(parents=True, exist_ok=True)
    (wt / fname).write_text(body, encoding="utf-8")
    sh(wt, "add", "-A")
    sh(wt, "commit", "-q", "-m", message if message is not None else f"work\n\nRef: {T1}")


def write_log(st: dict, sid: str) -> None:
    p = Path(st["worktree"]) / ".tmp/factory/myplan" / sid / "tests.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")


def maybe_state(root: Path, plan_id: str = "myplan") -> dict | None:
    return factory.load_state(root, plan_id)


def state_of(root: Path, plan_id: str = "myplan") -> dict:
    state = maybe_state(root, plan_id)
    assert state is not None
    return state


def started(repo: Path, capsys: pytest.CaptureFixture[str], text: str | None = None) -> None:
    p = write_plan(repo, text or plan_text(step("S1")))
    assert jrun(capsys, "start", str(p))[0] == 0


# ------------------------------------------------------------------- parsing
def test_parse_plan_accepts_checkbox_and_metadata() -> None:
    text = plan_text(step("S1"), step("S2", deps="S1, S3").replace("[ ]", "[x]"))
    steps = factory.parse_plan(text)
    assert [s["id"] for s in steps] == ["S1", "S2"]
    assert steps[1]["deps"] == ["S1", "S3"] and steps[1]["checked"] is True
    assert steps[0]["write"] == ["src/a/**"] and steps[0]["ticket"] == T1


def test_glob_overlap() -> None:
    assert factory.globs_overlap("src/a/**", "src/a/x.py")
    assert factory.globs_overlap("src/**", "src/a/**")
    assert not factory.globs_overlap("src/a/**", "src/b/**")
    assert not factory.globs_overlap("a/b.py", "a/b.pyc")


# ----------------------------------------------------------------- preflight
def test_start_refuses_cycle(repo: Path, capsys) -> None:
    p = write_plan(repo, plan_text(step("S1", deps="S2"), step("S2", deps="S1", write="src/b/**")))
    code, out = jrun(capsys, "start", str(p))
    assert code == 1 and "cycle" in " ".join(out["errors"])
    assert maybe_state(repo) is None


def test_start_refuses_missing_approval_and_reports_all(repo: Path, capsys) -> None:
    (repo / "governance/tickets/T-IGX-901-A.md").write_text("# no approval\n", encoding="utf-8")
    p = write_plan(
        repo,
        plan_text(
            step("S1", ticket="governance/tickets/T-IGX-901-A.md"),
            step("S2", ticket="governance/tickets/none.md", write="src/b/**"),
        ),
    )
    code, out = jrun(capsys, "start", str(p))
    assert code == 1 and len(out["errors"]) == 2
    assert "Approved" in out["errors"][0]


def test_start_refuses_parallel_overlap_but_allows_chain(repo: Path, capsys) -> None:
    p = write_plan(repo, plan_text(step("S1"), step("S2", write="src/a/x/**")))
    code, out = jrun(capsys, "start", str(p))
    assert code == 1 and "overlap" in out["errors"][0]
    p = write_plan(repo, plan_text(step("S1"), step("S2", deps="S1", write="src/a/x/**")), "chain")
    assert jrun(capsys, "start", str(p))[0] == 0


def test_start_creates_integration_worktree_and_is_idempotent(repo: Path, capsys) -> None:
    p = write_plan(repo, plan_text(step("S1")))
    code, out = jrun(capsys, "start", str(p))
    assert code == 0 and out["int_branch"] == "factory/myplan/int"
    st = state_of(repo)
    assert Path(st["int_worktree"]).is_dir() and st["owner_conversation"] is None
    assert st["base_sha"] == sh(repo, "rev-parse", "HEAD")
    assert Path(st["int_worktree"]).parent == repo.parent / "_wt" / "main"
    code, again = jrun(capsys, "start", str(p))
    assert code == 0 and again["already_started"] is True


# --------------------------------------------------------------- next / park
def test_ready_cap_and_park_propagation(repo: Path, capsys) -> None:
    started(
        repo,
        capsys,
        plan_text(
            step("S1", write="src/1/**"),
            step("S2", write="src/2/**"),
            step("S3", write="src/3/**"),
            step("S4", deps="S1", write="src/4/**"),
        ),
    )
    _, nxt = jrun(capsys, "next")
    assert [e["id"] for e in nxt["ready"]] == ["S1", "S2"]
    run(capsys, "launch", "S1")
    run(capsys, "launch", "S2", "--background")
    _, nxt = jrun(capsys, "next")
    assert nxt["ready"] == [] and [e["id"] for e in nxt["running"]] == ["S1"]
    assert [e["id"] for e in nxt["awaiting_background"]] == ["S2"] and nxt["done"] is False
    code, out = jrun(capsys, "launch", "S3")
    assert code == 1 and "cap" in out["errors"][0]
    _, out = jrun(capsys, "park", "S1", "--type", "missing access", "--reason", "no token")
    assert out["parked"] == ["S1", "S4"]
    assert state_of(repo)["steps"]["S4"]["park"]["reason"] == "blocked by S1"
    assert jrun(capsys, "park", "S3", "--type", "bogus", "--reason", "x")[0] == 1


def test_launch_prints_self_contained_brief(repo: Path, capsys) -> None:
    started(repo, capsys, plan_text(step("S1", tests="tests/a tests/b")))
    code, out = run(capsys, "launch", "S1")
    st = state_of(repo)["steps"]["S1"]
    assert code == 0 and st["status"] == "running"
    for needle in (
        st["worktree"],
        T1,
        "src/a/**",
        "igx_test_lock.py --log .tmp/factory/myplan/S1/tests.log",
        "uv run pytest tests/a tests/b -q --basetemp=",
        "DONE: <branch> <sha>",
        "Co-authored-by: Cursor",
        "git log -1 --format=%B",
        "BLOCKED:",
        "git push",
    ):
        assert needle in out
    assert st["launch_base"] == sh(repo, "rev-parse", "factory/myplan/int")


# ---------------------------------------------------- receive / review / merge
def test_receive_pass_review_integrate_and_complete(repo: Path, capsys) -> None:
    started(repo, capsys)
    run(capsys, "launch", "S1")
    st = state_of(repo)["steps"]["S1"]
    worker_commit(st)
    write_log(st, "S1")
    assert jrun(capsys, "integrate", "S1")[0] == 1  # not accepted yet
    code, out = jrun(capsys, "receive", "S1", "--agent-id", "w1")
    assert code == 0 and out["status"] == "received"
    assert jrun(capsys, "integrate", "S1")[0] == 1  # review gate
    assert jrun(capsys, "review", "S1", "--agent-id", "r1", "--verdict", "ACCEPT")[1]["status"] == "accepted"
    code, out = jrun(capsys, "integrate", "S1")
    assert code == 0 and out["status"] == "integrated"
    assert "- [x] S1 Title S1" in (repo / "myplan.md").read_text(encoding="utf-8")
    int_wt = Path(state_of(repo)["int_worktree"])
    assert (int_wt / "src/a/f.txt").is_file()
    assert f"Ref: {T1}" in sh(int_wt, "log", "-1", "--format=%B")
    code, text = run(capsys, "complete")
    assert code == 0 and text.startswith("PLAN COMPLETE") and "WITH ESCALATIONS" not in text
    assert "git push origin factory/myplan/int" in text
    assert jrun(capsys, "next")[1]["done"] is True


def test_receive_rejects_out_of_scope_missing_ref_and_log(repo: Path, capsys) -> None:
    started(repo, capsys)
    run(capsys, "launch", "S1")
    worker_commit(state_of(repo)["steps"]["S1"], fname="other/x.txt", message="no trailer")
    code, out = jrun(capsys, "receive", "S1", "--agent-id", "w1")
    joined = " ".join(out["reasons"])
    assert code == 1 and "outside write set" in joined and "lacks trailer" in joined and "test log missing" in joined
    assert "CORRECTION REQUIRED" in out["correction_brief"] and out["status"] == "correction"


def test_receive_rejects_ai_attribution_and_empty_branch(repo: Path, capsys) -> None:
    started(repo, capsys)
    run(capsys, "launch", "S1")
    code, out = jrun(capsys, "receive", "S1", "--agent-id", "w1")
    assert code == 1 and "no commits" in out["reasons"][0]
    run(capsys, "launch", "S1")
    st = state_of(repo)["steps"]["S1"]
    worker_commit(st, message=f"work\n\nRef: {T1}\nCo-Authored-By: Claude <x@y>")
    write_log(st, "S1")
    code, out = jrun(capsys, "receive", "S1", "--agent-id", "w1")
    assert code == 1 and "AI attribution" in " ".join(out["reasons"])


def test_correction_budget_relaunch_then_park(repo: Path, capsys) -> None:
    started(repo, capsys, plan_text(step("S1"), step("S2", deps="S1", write="src/b/**")))
    statuses = []
    for _ in range(6):
        run(capsys, "launch", "S1")
        _, out = jrun(capsys, "receive", "S1", "--agent-id", "w")  # empty branch fails each time
        statuses.append(out["status"])
        if out["status"] == "parked":
            break
    assert statuses == ["correction", "correction", "relaunch", "parked"]
    st = state_of(repo)
    assert st["steps"]["S1"]["park"]["type"] == "scope/cost/reversibility decision"
    assert st["steps"]["S2"]["park"]["reason"] == "blocked by S1"
    assert st["steps"]["S1"]["worker_branch"].endswith("S1-r1")


def test_review_gating_and_reject_counts_as_correction(repo: Path, capsys) -> None:
    started(repo, capsys)
    assert jrun(capsys, "review", "S1", "--agent-id", "r", "--verdict", "ACCEPT")[0] == 1
    run(capsys, "launch", "S1")
    st = state_of(repo)["steps"]["S1"]
    worker_commit(st)
    write_log(st, "S1")
    jrun(capsys, "receive", "S1", "--agent-id", "w")
    _, out = jrun(capsys, "review", "S1", "--agent-id", "r", "--verdict", "REJECT", "--findings", "bad naming")
    assert out["status"] == "correction" and "bad naming" in out["correction_brief"]
    _, text = run(capsys, "launch", "S1")
    assert "bad naming" in text


def test_integrate_conflict_parks_step(repo: Path, capsys) -> None:
    started(repo, capsys, plan_text(step("S1"), step("S2", write="src/b/**"), step("S3", deps="S2", write="src/c/**")))
    run(capsys, "launch", "S1")
    run(capsys, "launch", "S2")
    st = state_of(repo)
    st["steps"]["S2"]["write"] = ["src/a/**"]  # emulate overlapping scope to provoke a real merge conflict
    factory.save_state(st)
    for sid, body in (("S1", "one\n"), ("S2", "two\n")):
        cur = state_of(repo)["steps"][sid]
        worker_commit(cur, fname="src/a/base.txt", body=body)
        write_log(cur, sid)
        assert jrun(capsys, "receive", sid, "--agent-id", "w")[0] == 0
        jrun(capsys, "review", sid, "--agent-id", "r", "--verdict", "ACCEPT")
    assert jrun(capsys, "integrate", "S1")[0] == 0
    code, out = jrun(capsys, "integrate", "S2")
    assert code == 1 and out["status"] == "parked"
    st = state_of(repo)
    assert st["steps"]["S2"]["park"]["type"] == "plan/rules/code contradiction"
    assert st["steps"]["S3"]["park"]["reason"] == "blocked by S2"
    assert sh(Path(st["int_worktree"]), "status", "--porcelain") == ""


def test_complete_with_escalations_and_incomplete(repo: Path, capsys) -> None:
    started(repo, capsys, plan_text(step("S1"), step("S2", write="src/b/**")))
    code, text = run(capsys, "complete")
    assert code == 1 and "S1" in text and "S2" in text
    jrun(
        capsys,
        "park",
        "S1",
        "--type",
        "missing access",
        "--reason",
        "no token",
        "--options",
        "grant; skip",
        "--recommendation",
        "grant",
    )
    jrun(capsys, "park", "S2", "--type", "forbidden action", "--reason", "needs push")
    code, text = run(capsys, "complete")
    assert code == 0 and text.startswith("PLAN COMPLETE WITH ESCALATIONS")
    assert "ESCALATION:" in text and "S1 [missing access]: no token" in text and "recommendation: grant" in text


def test_progress_signature_changes_with_status(repo: Path, capsys) -> None:
    started(repo, capsys)
    before = factory.progress_signature(state_of(repo))
    run(capsys, "launch", "S1")
    assert factory.progress_signature(state_of(repo)) != before
    assert len(factory.find_active_runs(repo)) == 1


# ------------------------------------------- tip binding, independence, recovery
def received(repo: Path, capsys, sid: str = "S1", worker: str = "w") -> dict:
    run(capsys, "launch", sid)
    st = state_of(repo)["steps"][sid]
    worker_commit(st)
    write_log(st, sid)
    assert jrun(capsys, "receive", sid, "--agent-id", worker)[0] == 0
    return state_of(repo)["steps"][sid]


def test_receive_stores_tip_and_integrate_refuses_moved_branch(repo: Path, capsys) -> None:
    started(repo, capsys)
    st = received(repo, capsys)
    assert st["tip"] == sh(repo, "rev-parse", st["worker_branch"])
    assert jrun(capsys, "review", "S1", "--agent-id", "r", "--verdict", "ACCEPT")[0] == 0
    worker_commit(st, fname="src/a/g.txt", message=f"more\n\nRef: {T1}")
    code, out = jrun(capsys, "integrate", "S1")
    assert code == 1 and "receive again" in out["errors"][0]
    assert state_of(repo)["steps"]["S1"]["status"] == "accepted"
    assert not (Path(state_of(repo)["int_worktree"]) / "src/a/f.txt").exists()


def test_review_refuses_moved_branch_and_same_agent(repo: Path, capsys) -> None:
    started(repo, capsys)
    st = received(repo, capsys, worker="w")
    code, out = jrun(capsys, "review", "S1", "--agent-id", "w", "--verdict", "ACCEPT")
    assert code == 1 and "independent" in out["errors"][0]
    worker_commit(st, fname="src/a/g.txt", message=f"more\n\nRef: {T1}")
    code, out = jrun(capsys, "review", "S1", "--agent-id", "r", "--verdict", "ACCEPT")
    assert code == 1 and "receive again" in out["errors"][0]
    assert state_of(repo)["steps"]["S1"]["status"] == "received"


def test_fresh_worker_first_failure_parks(repo: Path, capsys) -> None:
    started(repo, capsys)
    outs = []
    for _ in range(4):
        run(capsys, "launch", "S1")
        outs.append(jrun(capsys, "receive", "S1", "--agent-id", "w")[1]["status"])
        if outs[-1] == "parked":
            break
    assert outs == ["correction", "correction", "relaunch", "parked"]


def test_lost_worker_relaunches_without_consuming_a_correction(repo: Path, capsys) -> None:
    started(repo, capsys)
    run(capsys, "launch", "S1")
    code, out = jrun(capsys, "receive", "S1", "--agent-id", "lost")
    st = state_of(repo)["steps"]["S1"]
    assert out["status"] == "relaunch" and st["corrections"] == 0 and st["relaunches"] == 0
    run(capsys, "launch", "S1")
    st2 = state_of(repo)["steps"]["S1"]
    assert st2["worker_branch"] != st["worker_branch"] and st2["status"] == "running"
    worker_commit(st2)
    write_log(st2, "S1")
    assert jrun(capsys, "receive", "S1", "--agent-id", "lost")[0] == 0  # commits exist: normal validation


def test_integrate_tests_run_without_shell_and_revert_blocks_remerge(repo: Path, capsys, monkeypatch) -> None:
    started(repo, capsys)
    received(repo, capsys)
    jrun(capsys, "review", "S1", "--agent-id", "r", "--verdict", "ACCEPT")
    monkeypatch.setattr(factory, "tests_argv", lambda *_a: [sys.executable, "-c", "import sys; sys.exit(1)"])
    seen: list[tuple[object, bool]] = []
    real = factory.subprocess.run

    def spy(cmd, *a, **kw):
        if cmd and cmd[0] != "git":
            seen.append((cmd, kw.get("shell", False)))
        return real(cmd, *a, **kw)

    monkeypatch.setattr(factory.subprocess, "run", spy)
    code, out = jrun(capsys, "integrate", "S1", "--run-tests")
    assert code == 1 and out["reason"] == "integration tests failed"
    assert len(seen) == 1 and isinstance(seen[0][0], list) and seen[0][1] is False
    st = state_of(repo)["steps"]["S1"]
    assert st["reverted_tips"] == [st["worker_branch"]]
    # a re-merge of the reverted branch would lose changes: integrate detects it and parks with a reason
    st_all = state_of(repo)
    st_all["steps"]["S1"]["status"] = "accepted"
    st_all["steps"]["S1"]["park"] = None
    factory.save_state(st_all)
    code, out = jrun(capsys, "integrate", "S1")
    assert code == 1 and out["reason"] == "branch already merged"
    assert "relaunch" in state_of(repo)["steps"]["S1"]["park"]["reason"]


def test_tests_argv_is_a_list() -> None:
    state = {"main_root": "/m/r", "plan_id": "p", "steps": {"S1": {"tests": "tests/a tests/b"}}}
    argv = factory.tests_argv(state, "S1", "log.txt")
    assert isinstance(argv, list) and argv[argv.index("--") + 1 :][:3] == ["uv", "run", "pytest"]
    assert "tests/a" in argv and "tests/b" in argv


def test_start_warns_when_cursor_attributes_commits(repo: Path, capsys, tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    (home / ".cursor").mkdir(parents=True)
    cfg = home / ".cursor" / "cli-config.json"
    cfg.write_text(json.dumps({"attribution": {"attributeCommitsToAgent": True}}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    before = cfg.read_text(encoding="utf-8")
    code, out = jrun(capsys, "start", str(write_plan(repo, plan_text(step("S1")))))
    assert code == 0 and "attributeCommitsToAgent" in out["warnings"][0]
    assert cfg.read_text(encoding="utf-8") == before


def test_attribution_warning_tolerates_missing_invalid_or_false(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "h"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert factory.attribution_warnings() == []
    (home / ".cursor").mkdir(parents=True)
    cfg = home / ".cursor" / "cli-config.json"
    cfg.write_text("{not json", encoding="utf-8")
    assert factory.attribution_warnings() == []
    cfg.write_text(json.dumps({"attribution": {"attributeCommitsToAgent": False}}), encoding="utf-8")
    assert factory.attribution_warnings() == []
