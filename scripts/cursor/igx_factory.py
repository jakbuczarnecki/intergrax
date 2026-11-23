#!/usr/bin/env python3
"""Cursor-only software factory state machine (stdlib only).

Runs an approved multi-step plan through worker branches that are merged
locally into a per-plan integration branch. No push, PR or remote merge.
See agent-env/plans/cursor-software-factory.md (F1, F2).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CHECKBOX = re.compile(r"^(\s*[-*]\s+\[)( |x|X)(\]\s+)(.*\S)\s*$")
META = re.compile(r"^\s+[-*]\s+(deps|agent|ticket|write|tests):\s*(.*\S)?\s*$")
STEP_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
ESCALATION_TYPES = (
    "unplanned architecture",
    "scope/cost/reversibility decision",
    "plan/rules/code contradiction",
    "forbidden action",
    "missing access",
)
IMPLEMENTER = "igx-implementer"
CAP = 2
CORRECTION_BUDGET = 2
LOST_AGENT = "lost"
MAX_LOST = 2
AI_ATTRIBUTION = re.compile(
    r"(?im)^\s*(co-authored-by:.*(claude|anthropic|openai|gpt|cursor|grok|composer|copilot|gemini|\bai\b)"
    r"|generated[ -]with|generated-by:|assisted-by:)"
)


class FactoryError(RuntimeError):
    """Fail-closed refusal carrying one or more reasons."""

    def __init__(self, *reasons: str) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


# ---------------------------------------------------------------- git helpers
def git(args: list[str], cwd: Path | str, *, check: bool = True) -> str:
    done = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, encoding="utf-8", errors="replace", check=False
    )
    if check and done.returncode:
        raise FactoryError(f"git {' '.join(args[:2])} failed: {(done.stderr or done.stdout).strip()[:300]}")
    return done.stdout.strip() if check else (done.stdout + done.stderr).strip() + f"\n#rc={done.returncode}"


def toplevel(cwd: Path | str) -> Path:
    return Path(git(["rev-parse", "--show-toplevel"], cwd))


def main_root(cwd: Path | str) -> Path:
    common = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd)
    return Path(common).parent


def resolve_root(cwd: Path | str) -> Path:
    """State root: the pointer written into factory worktrees, else the toplevel."""
    top = toplevel(cwd)
    pointer = top / ".tmp" / "factory_root"
    if pointer.is_file():
        return Path(pointer.read_text(encoding="utf-8").strip())
    return top


def slug(text: str, limit: int = 24) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x")[:limit].strip("-")


# ---------------------------------------------------------------------- state
def state_dir(root: Path, plan_id: str) -> Path:
    return Path(root) / ".tmp" / "factory" / plan_id


def load_state(root: Path, plan_id: str) -> dict[str, Any] | None:
    path = state_dir(root, plan_id) / "state.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    directory = state_dir(Path(state["root"]), state["plan_id"])
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix="state.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(tmp, directory / "state.json")


def is_finished(state: dict[str, Any]) -> bool:
    return all(s["status"] in ("integrated", "parked") for s in state["steps"].values())


def find_active_runs(root: Path) -> list[dict[str, Any]]:
    """States under <root>/.tmp/factory that still have unfinished steps."""
    base = Path(root) / ".tmp" / "factory"
    runs: list[dict[str, Any]] = []
    if not base.is_dir():
        return runs
    for entry in sorted(base.iterdir()):
        state = load_state(Path(root), entry.name) if entry.is_dir() else None
        if state and not is_finished(state):
            runs.append(state)
    return runs


def progress_signature(state: dict[str, Any]) -> str:
    rows = [
        [sid, s["status"], s.get("corrections", 0), s.get("relaunches", 0), bool(s.get("background"))]
        for sid, s in sorted(state["steps"].items())
    ]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


# -------------------------------------------------------------------- parsing
def parse_plan(text: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        box = CHECKBOX.match(line)
        if box:
            parts = box.group(4).split(None, 1)
            if not STEP_ID.match(parts[0]):
                current = None
                continue
            current = {
                "id": parts[0],
                "title": parts[1] if len(parts) > 1 else parts[0],
                "checked": box.group(2) in ("x", "X"),
                "deps": [],
                "agent": IMPLEMENTER,
                "ticket": "",
                "write": [],
                "tests": "",
            }
            steps.append(current)
            continue
        meta = META.match(line)
        if meta and current is not None:
            key, value = meta.group(1), (meta.group(2) or "").strip()
            if key in ("deps", "write"):
                current[key] = [v.strip() for v in value.split(",") if v.strip() and v.strip() != "-"]
            else:
                current[key] = value
    return steps


def reachable(deps: dict[str, list[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = list(deps.get(start, []))
    while stack:
        item = stack.pop()
        if item not in seen:
            seen.add(item)
            stack.extend(deps.get(item, []))
    return seen


def find_cycle(deps: dict[str, list[str]]) -> list[str]:
    return [sid for sid in deps if sid in reachable(deps, sid)]


def glob_prefix(pattern: str) -> str | None:
    match = re.search(r"[*?\[]", pattern)
    return pattern[: match.start()] if match else None


def globs_overlap(a: str, b: str) -> bool:
    if a == b or fnmatch.fnmatchcase(a, b) or fnmatch.fnmatchcase(b, a):
        return True
    pa, pb = glob_prefix(a), glob_prefix(b)
    if pa is not None and pb is not None:
        return pa.startswith(pb) or pb.startswith(pa)
    if pa is not None:
        return b.startswith(pa)
    if pb is not None:
        return a.startswith(pb)
    return False


def in_write_set(path: str, globs: list[str]) -> bool:
    for pattern in globs:
        if fnmatch.fnmatchcase(path, pattern):
            return True
        if pattern.endswith("/**") and path.startswith(pattern[:-2]):
            return True
        if pattern.endswith("/") and path.startswith(pattern):
            return True
    return False


def preflight(steps: list[dict[str, Any]], root: Path) -> list[str]:
    errors: list[str] = []
    if not steps:
        return ["plan has no step checkbox lines"]
    ids = [s["id"] for s in steps]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(f"duplicate step id: {dup}")
    known = set(ids)
    deps = {s["id"]: s["deps"] for s in steps}
    for s in steps:
        for dep in s["deps"]:
            if dep not in known:
                errors.append(f"{s['id']}: unknown dependency {dep}")
        if not s["write"]:
            errors.append(f"{s['id']}: missing write set")
        if not s["tests"]:
            errors.append(f"{s['id']}: missing tests")
        ticket = root / s["ticket"] if s["ticket"] else None
        if ticket is None:
            errors.append(f"{s['id']}: missing ticket")
        elif not ticket.is_file():
            errors.append(f"{s['id']}: ticket file not found: {s['ticket']}")
        elif not any(ln.startswith("Approved:") for ln in ticket.read_text(encoding="utf-8").splitlines()):
            errors.append(f"{s['id']}: ticket {s['ticket']} has no 'Approved:' line")
    if errors:
        return errors
    cyc = find_cycle(deps)
    if cyc:
        return [f"dependency cycle involving: {', '.join(sorted(cyc))}"]
    for i, a in enumerate(steps):
        for b in steps[i + 1 :]:
            if b["id"] in reachable(deps, a["id"]) or a["id"] in reachable(deps, b["id"]):
                continue
            for ga in a["write"]:
                for gb in b["write"]:
                    if globs_overlap(ga, gb):
                        errors.append(f"write sets overlap between parallel steps {a['id']} and {b['id']}: {ga} ~ {gb}")
    return errors


# --------------------------------------------------------------------- layout
def wt_path(state: dict[str, Any], suffix: str) -> Path:
    main = Path(state["main_root"])
    return main.parent / "_wt" / main.name / f"f-{slug(state['plan_id'])}-{suffix}"


def write_pointer(worktree: Path, root: str) -> None:
    pointer = worktree / ".tmp" / "factory_root"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(root, encoding="utf-8")


def tests_argv(state: dict[str, Any], sid: str, log: str) -> list[str]:
    step = state["steps"][sid]
    raw = step["tests"].strip()
    if raw.split()[0] in ("uv", "python", "py"):
        inner = shlex.split(raw, posix=True)
    else:
        base = re.sub(r"^pytest\s+", "", raw)
        main = Path(state["main_root"])
        scratch = main.parent / "_wt" / main.name / "_scratch" / f"{slug(state['plan_id'], 12)}-{sid.lower()}"
        inner = ["uv", "run", "pytest", *shlex.split(base, posix=True), "-q", f"--basetemp={scratch}"]
    lock = Path(__file__).resolve().parent / "igx_test_lock.py"
    return [sys.executable, str(lock), "--log", log, "--", *inner]


def tests_command(state: dict[str, Any], sid: str, log: str) -> str:
    """Display form for the worker brief (workers run it in their own shell)."""
    argv = tests_argv(state, sid, log)
    argv[0] = "python"
    return " ".join(argv)


def attribution_warnings() -> list[str]:
    """Non-fatal: the Cursor CLI appends a co-author trailer when this user setting is on. Never edits the file."""
    try:
        cfg = json.loads((Path.home() / ".cursor" / "cli-config.json").read_text(encoding="utf-8"))
        attr = cfg.get("attribution") if isinstance(cfg, dict) else None
        if isinstance(attr, dict) and attr.get("attributeCommitsToAgent") is True:
            return [
                "~/.cursor/cli-config.json has attribution.attributeCommitsToAgent=true: Cursor appends a "
                "'Co-authored-by: Cursor' trailer to worker commits and receive rejects them. "
                "Operator: set it to false."
            ]
    except (OSError, ValueError):
        pass
    return []


def build_brief(state: dict[str, Any], sid: str, findings: str | None = None) -> str:
    s = state["steps"][sid]
    log = f".tmp/factory/{state['plan_id']}/{sid}/tests.log"
    lines = [
        f"ROLE: {s['agent']} - implement exactly one plan step; nothing else.",
        f"PLAN: {state['plan_id']}  STEP: {sid} - {s['title']}",
        f"WORKTREE (absolute): {s['worktree']}",
        "Work only there. First command: cd into that directory. Never touch the main checkout or other worktrees.",
        f"BRANCH: {s['worker_branch']} (already checked out in the worktree)",
        f"TICKET: {s['ticket']} (already approved)",
        "WRITE SET (edit only files matching these globs; anything else is a scope change -> BLOCKED):",
        *[f"  - {g}" for g in s["write"]],
        "TESTS (run once, after the work, exactly this command; it serialises pytest across workers):",
        f"  {tests_command(state, sid, log)}",
        "COMMIT RULES: commit on the worker branch; English message; message body ends with the trailer line",
        f"  Ref: {s['ticket']}",
        "  No AI attribution: no Co-Authored-By naming a tool, no 'Generated with' footer.",
        "  Cursor may append a 'Co-authored-by: Cursor ...' trailer to your commit automatically. Before reporting",
        "  DONE run `git log -1 --format=%B`; if any AI co-author or 'Generated' trailer is present, amend that commit",
        "  with the message rewritten without it (keep the Ref trailer), then re-check with `git log -1 --format=%B`.",
        "FORBIDDEN: git push, gh, PR creation, merge, git reset --hard, --no-verify, touching the main checkout or",
        "  other worktrees, spawning subagents.",
        "DECIDE ALONE: when the plan, ADRs, repo rules or existing code answer a question, decide and continue;",
        "  never ask whether to continue, commit, run tests or retry.",
        "FINAL LINE (exactly one, last line of your reply):",
        "  DONE: <branch> <sha> <tests summary>",
        "  or BLOCKED: <type>: <what, options, recommendation>",
        "  where type is one of: " + "; ".join(ESCALATION_TYPES),
    ]
    if findings:
        lines += [
            "",
            "CORRECTION REQUIRED - fix these findings on the same branch, commit again, rerun tests:",
            findings,
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------- commands
def cmd_start(plan_arg: str, cwd: Path) -> dict[str, Any]:
    root = toplevel(cwd)
    plan_path = Path(plan_arg)
    plan_path = (plan_path if plan_path.is_absolute() else Path(cwd) / plan_path).resolve()
    if not plan_path.is_file():
        raise FactoryError(f"plan file not found: {plan_path}")
    plan_id = plan_path.stem
    existing = load_state(root, plan_id)
    if existing is not None:
        if Path(existing["plan_path"]) == plan_path:
            return {"ok": True, "already_started": True, "plan_id": plan_id}
        raise FactoryError(f"plan id {plan_id} already started from a different file")
    steps = parse_plan(plan_path.read_text(encoding="utf-8"))
    errors = preflight(steps, root)
    if errors:
        raise FactoryError(*errors)
    main = main_root(cwd)
    base = git(["rev-parse", "HEAD"], root)
    state: dict[str, Any] = {
        "plan_id": plan_id,
        "plan_path": str(plan_path),
        "root": str(root),
        "main_root": str(main),
        "base_sha": base,
        "int_branch": f"factory/{plan_id}/int",
        "int_worktree": "",
        "owner_conversation": None,
        "stall": {"signature": None, "count": 0},
        "steps": {},
    }
    state["int_worktree"] = str(wt_path(state, "int"))
    git(["worktree", "add", "-b", state["int_branch"], state["int_worktree"], base], root)
    write_pointer(Path(state["int_worktree"]), str(root))
    for s in steps:
        state["steps"][s["id"]] = {
            **{k: s[k] for k in ("title", "deps", "agent", "ticket", "write", "tests")},
            "status": "integrated" if s["checked"] else "pending",
            "background": False,
            "corrections": 0,
            "relaunches": 0,
            "failures": 0,
            "worker_branch": "",
            "worktree": "",
            "launch_base": "",
            "agent_id": None,
            "review": None,
            "findings": None,
            "park": None,
        }
    save_state(state)
    result: dict[str, Any] = {
        "ok": True,
        "plan_id": plan_id,
        "base_sha": base,
        "int_branch": state["int_branch"],
        "steps": len(steps),
    }
    warnings = attribution_warnings()
    if warnings:
        result["warnings"] = warnings
    return result


def deps_integrated(state: dict[str, Any], sid: str) -> bool:
    return all(state["steps"][d]["status"] == "integrated" for d in state["steps"][sid]["deps"])


def running_implementers(state: dict[str, Any]) -> int:
    return sum(1 for s in state["steps"].values() if s["status"] == "running" and s["agent"] == IMPLEMENTER)


def entry(state: dict[str, Any], sid: str) -> dict[str, Any]:
    s = state["steps"][sid]
    return {"id": sid, "title": s["title"], "agent": s["agent"], "status": s["status"]}


def cmd_next(state: dict[str, Any]) -> dict[str, Any]:
    steps = state["steps"]
    slots = max(0, CAP - running_implementers(state))
    ready_all = [
        sid
        for sid, s in steps.items()
        if (s["status"] == "pending" and deps_integrated(state, sid)) or s["status"] in ("correction", "relaunch")
    ]
    ready = [entry(state, sid) for sid in ready_all][:slots]
    running = [entry(state, sid) for sid, s in steps.items() if s["status"] == "running" and not s["background"]]
    awaiting = [entry(state, sid) for sid, s in steps.items() if s["status"] == "running" and s["background"]]
    actions = [entry(state, sid) for sid, s in steps.items() if s["status"] in ("received", "accepted")]
    parked = [{**entry(state, sid), "park": s["park"]} for sid, s in steps.items() if s["status"] == "parked"]
    return {
        "ready": ready,
        "running": running,
        "awaiting_background": awaiting,
        "pending_actions": actions,
        "parked": parked,
        "done": is_finished(state),
    }


def need_status(state: dict[str, Any], sid: str, *allowed: str) -> dict[str, Any]:
    if sid not in state["steps"]:
        raise FactoryError(f"unknown step: {sid}")
    s = state["steps"][sid]
    if s["status"] not in allowed:
        raise FactoryError(f"step {sid} is {s['status']}; expected one of: {', '.join(allowed)}")
    return s


def cmd_launch(state: dict[str, Any], sid: str, background: bool) -> str:
    s = need_status(state, sid, "pending", "correction", "relaunch")
    if s["status"] == "pending" and not deps_integrated(state, sid):
        raise FactoryError(f"step {sid} has dependencies that are not integrated")
    if s["agent"] == IMPLEMENTER and running_implementers(state) >= CAP:
        raise FactoryError(f"cap reached: {CAP} implementers already running")
    root = Path(state["root"])
    if s["status"] in ("pending", "relaunch"):
        suffix = ""
        if s["status"] == "relaunch":
            suffix = f"-r{s['relaunches']}" + (f"l{s['lost']}" if s.get("lost") else "")
        branch = f"factory/{state['plan_id']}/{sid}{suffix}"
        path = wt_path(state, f"{sid.lower()}{suffix}")
        tip = git(["rev-parse", state["int_branch"]], root)
        git(["worktree", "add", "-b", branch, str(path), tip], root)
        write_pointer(path, str(root))
        s.update(worker_branch=branch, worktree=str(path), launch_base=tip, agent_id=None, review=None)
    findings = s["findings"]
    s["status"] = "running"
    s["background"] = background
    save_state(state)
    return build_brief(state, sid, findings)


def correction_brief(state: dict[str, Any], sid: str, reasons: list[str]) -> str:
    return build_brief(state, sid, "\n".join(f"- {r}" for r in reasons))


def register_failure(state: dict[str, Any], sid: str, reasons: list[str]) -> dict[str, Any]:
    s = state["steps"][sid]
    s["failures"] += 1
    s["corrections"] += 1
    s["findings"] = "\n".join(f"- {r}" for r in reasons)
    out: dict[str, Any] = {"ok": False, "reasons": reasons}
    # budget: 2 corrections, then 1 fresh worker, then park (the fresh worker gets no corrections)
    if s["relaunches"] == 0 and s["corrections"] <= CORRECTION_BUDGET:
        s["status"] = "correction"
        out["correction_brief"] = correction_brief(state, sid, reasons)
    elif s["relaunches"] == 0:
        s["relaunches"] = 1
        s["corrections"] = 0
        s["status"] = "relaunch"
        out["action"] = "relaunch: launch the step again with a fresh worker"
    else:
        do_park(
            state,
            sid,
            "scope/cost/reversibility decision",
            "correction and relaunch budget exhausted: " + "; ".join(reasons),
            "retry with a new plan step; widen write set; drop the step",
            "re-scope the step",
        )
        out["status"] = "parked"
        return out
    out["status"] = s["status"]
    return out


def validate_worker(state: dict[str, Any], sid: str) -> list[str]:
    s = state["steps"][sid]
    root = Path(state["root"])
    base, branch = s["launch_base"], s["worker_branch"]
    reasons: list[str] = []
    commits = git(["rev-list", "--no-merges", f"{base}..{branch}"], root).split()
    if not commits:
        return [f"branch {branch} has no commits beyond the launch base"]
    for name in git(["diff", "--name-only", f"{base}..{branch}"], root).splitlines():
        if not in_write_set(name, s["write"]):
            reasons.append(f"file outside write set: {name}")
    for sha in commits:
        message = git(["log", "-1", "--format=%B", sha], root)
        if f"Ref: {s['ticket']}" not in message:
            reasons.append(f"commit {sha[:9]} lacks trailer 'Ref: {s['ticket']}'")
        if AI_ATTRIBUTION.search(message):
            reasons.append(f"commit {sha[:9]} carries AI attribution")
    rel = Path(".tmp") / "factory" / state["plan_id"] / sid / "tests.log"
    if not any(p.is_file() for p in (Path(s["worktree"]) / rel, root / rel)):
        reasons.append(f"test log missing: {rel.as_posix()} (run tests through igx_test_lock.py --log)")
    return reasons


def cmd_receive(state: dict[str, Any], sid: str, agent_id: str) -> dict[str, Any]:
    s = need_status(state, sid, "running")
    s["agent_id"] = agent_id
    s["background"] = False
    root = Path(state["root"])
    if agent_id == LOST_AGENT and not git(
        ["rev-list", "--no-merges", f"{s['launch_base']}..{s['worker_branch']}"], root
    ):
        # recovery: the worker vanished without committing; relaunch without consuming a correction
        s["lost"] = s.get("lost", 0) + 1
        s["agent_id"] = None
        if s["lost"] > MAX_LOST:
            do_park(
                state,
                sid,
                "missing access",
                f"worker lost {s['lost']} times without committing",
                "check the Cursor agent runtime; re-plan the step",
                "inspect the agent runtime",
            )
            out: dict[str, Any] = {"ok": False, "status": "parked"}
        else:
            s["status"] = "relaunch"
            s["findings"] = None
            out = {
                "ok": False,
                "status": "relaunch",
                "action": "worker lost: launch the step again, no correction used",
            }
        save_state(state)
        return out
    reasons = validate_worker(state, sid)
    if reasons:
        out = register_failure(state, sid, reasons)
    else:
        s["tip"] = git(["rev-parse", s["worker_branch"]], root)
        s["status"] = "received"
        s["findings"] = None
        out = {"ok": True, "status": "received"}
    save_state(state)
    return out


def check_tip(state: dict[str, Any], sid: str) -> None:
    s = state["steps"][sid]
    now = git(["rev-parse", s["worker_branch"]], Path(state["root"]))
    if s.get("tip") and now != s["tip"]:
        raise FactoryError(
            f"worker branch {s['worker_branch']} moved since receive ({s['tip'][:9]} -> {now[:9]}); run receive again"
        )


def cmd_review(state: dict[str, Any], sid: str, agent_id: str, verdict: str, findings: str | None) -> dict[str, Any]:
    s = need_status(state, sid, "received")
    if agent_id == s.get("agent_id"):
        raise FactoryError(f"reviewer {agent_id} is the worker of step {sid}; use an independent reviewer agent")
    check_tip(state, sid)
    s["review"] = {"agent_id": agent_id, "verdict": verdict, "findings": findings}
    if verdict == "ACCEPT":
        s["status"] = "accepted"
        out: dict[str, Any] = {"ok": True, "status": "accepted"}
    else:
        out = register_failure(state, sid, [f"reviewer rejected: {findings or 'no findings given'}"])
        out["ok"] = True
    save_state(state)
    return out


def tick_checkbox(plan_path: Path, sid: str) -> None:
    lines = plan_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        box = CHECKBOX.match(line.rstrip("\r\n"))
        if box and box.group(4).split(None, 1)[0] == sid:
            eol = line[len(line.rstrip("\r\n")) :]
            lines[i] = f"{box.group(1)}x{box.group(3)}{box.group(4)}{eol}"
            break
    plan_path.write_text("".join(lines), encoding="utf-8", newline="")


def cmd_integrate(state: dict[str, Any], sid: str, run_tests: bool) -> dict[str, Any]:
    s = need_status(state, sid, "accepted")
    wt = state["int_worktree"]
    check_tip(state, sid)
    merged = git(["merge-base", "--is-ancestor", s["worker_branch"], "HEAD"], wt, check=False).endswith("#rc=0")
    reverted = any(
        git(["merge-base", "--is-ancestor", t, "HEAD"], wt, check=False).endswith("#rc=0")
        for t in s.get("reverted_tips", [])
    )
    if merged or reverted:
        do_park(
            state,
            sid,
            "plan/rules/code contradiction",
            f"{s['worker_branch']} is already merged"
            + (" and was reverted" if reverted else "")
            + "; a re-merge would be a no-op or lose changes; relaunch the step from the integration tip",
            "relaunch the step as a fresh worker; or re-plan",
            "relaunch the step",
        )
        save_state(state)
        return {"ok": False, "status": "parked", "reason": "branch already merged"}
    done = subprocess.run(
        [
            "git",
            "merge",
            "--no-ff",
            "--no-edit",
            "-m",
            f"factory: integrate {sid}",
            "-m",
            f"Ref: {s['ticket']}",
            s["worker_branch"],
        ],
        cwd=wt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if done.returncode:
        git(["merge", "--abort"], wt, check=False)
        do_park(
            state,
            sid,
            "plan/rules/code contradiction",
            f"merge conflict integrating {s['worker_branch']}: {(done.stdout + done.stderr).strip()[:300]}",
            "resolve by re-planning write sets; rebase the step",
            "re-plan the step",
        )
        save_state(state)
        return {"ok": False, "status": "parked", "reason": "merge conflict"}
    if run_tests:
        log = state_dir(Path(state["root"]), state["plan_id"]) / sid / "integration-tests.log"
        argv = tests_argv(state, sid, str(log))
        rc = subprocess.run(argv, cwd=wt, shell=False, check=False).returncode
        if rc:
            s.setdefault("reverted_tips", []).append(s["worker_branch"])
            git(["revert", "-m", "1", "--no-edit", "HEAD"], wt)
            do_park(
                state,
                sid,
                "plan/rules/code contradiction",
                f"integration tests failed after merge (rc={rc}); merge reverted; log: {log}",
                "fix on a new correction step",
                "inspect the log",
            )
            save_state(state)
            return {"ok": False, "status": "parked", "reason": "integration tests failed"}
    tick_checkbox(Path(state["plan_path"]), sid)
    s["status"] = "integrated"
    save_state(state)
    return {"ok": True, "status": "integrated", "sha": git(["rev-parse", "HEAD"], wt)}


def dependants(state: dict[str, Any], sid: str) -> list[str]:
    deps = {k: v["deps"] for k, v in state["steps"].items()}
    return [k for k in deps if sid in reachable(deps, k)]


def do_park(state: dict[str, Any], sid: str, kind: str, reason: str, options: str, rec: str) -> list[str]:
    if kind not in ESCALATION_TYPES:
        raise FactoryError(f"invalid escalation type: {kind}; allowed: {'; '.join(ESCALATION_TYPES)}")
    if sid not in state["steps"]:
        raise FactoryError(f"unknown step: {sid}")
    if state["steps"][sid]["status"] == "integrated":
        raise FactoryError(f"step {sid} is already integrated")
    parked = [sid]
    state["steps"][sid].update(
        status="parked", park={"type": kind, "reason": reason, "options": options, "recommendation": rec}
    )
    for dep in dependants(state, sid):
        d = state["steps"][dep]
        if d["status"] not in ("integrated", "parked"):
            d.update(
                status="parked", park={"type": kind, "reason": f"blocked by {sid}", "options": "", "recommendation": ""}
            )
            parked.append(dep)
    return parked


def git_sha(path: str) -> str:
    return git(["rev-parse", "HEAD"], path)


def cmd_complete(state: dict[str, Any]) -> tuple[int, str]:
    steps = state["steps"]
    parked = [k for k, s in steps.items() if s["status"] == "parked"]
    if not is_finished(state):
        nxt = cmd_next(state)
        names = [e["id"] for k in ("ready", "running", "awaiting_background", "pending_actions") for e in nxt[k]]
        return 1, "NOT COMPLETE: remaining steps: " + ", ".join(
            names or [k for k, s in steps.items() if s["status"] not in ("integrated", "parked")]
        )
    sha = git_sha(state["int_worktree"])
    head = [
        f"integration branch: {state['int_branch']}",
        f"integration SHA: {sha}",
        f"Operator publish (text only, not run by the factory): git push origin {state['int_branch']}",
    ]
    if not parked:
        return 0, "\n".join([f"PLAN COMPLETE {state['plan_id']}", *head])
    block = ["ESCALATION: parked steps need Operator decisions"]
    for sid in parked:
        p = steps[sid]["park"]
        block += [
            f"- {sid} [{p['type']}]: {p['reason']}",
            f"  options: {p['options'] or '-'}",
            f"  recommendation: {p['recommendation'] or '-'}",
        ]
    return 0, "\n".join([f"PLAN COMPLETE WITH ESCALATIONS {state['plan_id']}", *head, "", *block])


# ------------------------------------------------------------------------ CLI
def pick_state(cwd: Path, plan: str | None) -> dict[str, Any]:
    root = resolve_root(cwd)
    if plan:
        state = load_state(root, plan)
        if state is None:
            raise FactoryError(f"no factory state for plan {plan} under {root}")
        return state
    runs = find_active_runs(root)
    if not runs:
        base = root / ".tmp" / "factory"
        names = [e.name for e in sorted(base.iterdir()) if (e / "state.json").is_file()] if base.is_dir() else []
        runs = [s for s in (load_state(root, n) for n in names) if s]
    if len(runs) != 1:
        raise FactoryError(f"expected exactly one active run under {root}, found {len(runs)}; pass --plan ID")
    return runs[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="igx_factory")
    p.add_argument("--plan", help="plan id (default: the only active run)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start").add_argument("plan_file")
    for name in ("next", "status", "complete"):
        sub.add_parser(name)
    s = sub.add_parser("launch")
    s.add_argument("step")
    s.add_argument("--background", action="store_true")
    s = sub.add_parser("receive")
    s.add_argument("step")
    s.add_argument("--agent-id", required=True)
    s = sub.add_parser("review")
    s.add_argument("step")
    s.add_argument("--agent-id", required=True)
    s.add_argument("--verdict", required=True, choices=["ACCEPT", "REJECT"])
    s.add_argument("--findings")
    s = sub.add_parser("integrate")
    s.add_argument("step")
    s.add_argument("--run-tests", action="store_true")
    s = sub.add_parser("park")
    s.add_argument("step")
    s.add_argument("--type", required=True)
    s.add_argument("--reason", required=True)
    s.add_argument("--options", default="")
    s.add_argument("--recommendation", default="")
    return p


def emit(data: Any) -> None:
    print(json.dumps(data, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = Path.cwd()
    try:
        if args.cmd == "start":
            emit(cmd_start(args.plan_file, cwd))
            return 0
        state = pick_state(cwd, args.plan)
        if args.cmd == "next":
            emit(cmd_next(state))
        elif args.cmd == "status":
            emit(state)
        elif args.cmd == "launch":
            print(cmd_launch(state, args.step, args.background))
        elif args.cmd == "receive":
            out = cmd_receive(state, args.step, args.agent_id)
            emit(out)
            return 0 if out["ok"] else 1
        elif args.cmd == "review":
            emit(cmd_review(state, args.step, args.agent_id, args.verdict, args.findings))
        elif args.cmd == "integrate":
            out = cmd_integrate(state, args.step, args.run_tests)
            emit(out)
            return 0 if out["ok"] else 1
        elif args.cmd == "park":
            parked = do_park(state, args.step, args.type, args.reason, args.options, args.recommendation)
            save_state(state)
            emit({"ok": True, "parked": parked})
        elif args.cmd == "complete":
            code, text = cmd_complete(state)
            print(text)
            return code
    except FactoryError as exc:
        emit({"ok": False, "errors": exc.reasons})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
