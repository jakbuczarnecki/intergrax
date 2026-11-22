# © Artur Czarnecki. All rights reserved.
"""Headless supervisor: drive a factory run with `cursor-agent -p` until it is done, parked or stalled.

Launch it from PowerShell (Git Bash breaks the Cursor hook wrapper). Exit codes:
0 run complete, 2 agent binary missing, 3 everything remaining is parked (escalations),
4 stalled, 5 max iterations or budget exhausted, 6 agent failing (3 consecutive non-zero iterations
that left the state unchanged).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import igx_factory as factory  # noqa: E402

HEADLESS_NOTE = (
    "In headless mode launch parallel workers as foreground Tasks in one message (no --background), "
    "since no completion turn exists headless."
)
CONTINUE_PROMPT = (
    "FACTORY CONTINUE: invoke /igx-native-run-plan and run python scripts/cursor/igx_factory.py next "
    "then follow the skill until igx_factory.py complete is printed; never ask the operator; "
    "park real blockers. " + HEADLESS_NOTE
)
STALL_ITERATIONS = 3
FAIL_ITERATIONS = 3
TAIL_LINES = 20
ID_KEYS = ("session_id", "chatId", "chat_id")


def find_agent(agent_bin: str) -> list[str] | None:
    """Resolve the agent command; a `.py` path is run with the current interpreter."""
    candidate = Path(agent_bin)
    if candidate.suffix == ".py" and candidate.is_file():
        return [sys.executable, str(candidate)]
    names = [agent_bin]
    if sys.platform == "win32" and agent_bin == "cursor-agent":
        names += ["cursor-agent.cmd", "agent"]
    for name in names:
        found = shutil.which(name)
        if found:
            return [found]
    return None


def extract_chat_id(stdout: str) -> str | None:
    docs: list[Any] = []
    try:
        docs.append(json.loads(stdout))
    except ValueError:
        for line in reversed([ln for ln in stdout.splitlines() if ln.strip()]):
            try:
                docs.append(json.loads(line))
                break
            except ValueError:
                continue
    for doc in docs:
        if isinstance(doc, dict):
            for key in ID_KEYS:
                value = doc.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def first_prompt(plan: Path) -> str:
    """Single line without cmd.exe metacharacters (the agent launcher is a .cmd file)."""
    return (
        f"/igx-native-run-plan {plan} - first run python scripts/cursor/igx_factory.py start {plan} "
        "then follow the skill until igx_factory.py complete is printed; never ask the operator; "
        "park real blockers. " + HEADLESS_NOTE
    )


def tail(text: str | None) -> list[str]:
    return (text or "").splitlines()[-TAIL_LINES:]


def summary(state: dict[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {"state": None}
    nxt = factory.cmd_next(state)
    keys = ("ready", "running", "awaiting_background", "pending_actions", "parked")
    out: dict[str, Any] = {k: [e["id"] for e in nxt[k]] for k in keys}
    out["done"] = nxt["done"]
    return out


def log_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="igx_factory_supervisor",
        description="Headless supervisor for a factory run. Launch it from PowerShell, not Git Bash "
        "(Git Bash breaks the Cursor hook wrapper).",
    )
    p.add_argument("--plan", required=True, help="plan markdown file")
    p.add_argument("--model", required=True, help="parent model id")
    p.add_argument("--agent-bin", default="cursor-agent")
    p.add_argument("--max-iterations", type=int, default=40)
    p.add_argument("--budget-minutes", type=float, default=720)
    p.add_argument("--workspace", default=None, help="checkout to run in (default: repo root)")
    return p


def finish(state: dict[str, Any] | None, code: int) -> int:
    if state is not None:
        try:
            print(factory.cmd_complete(state)[1])
        except Exception as exc:  # noqa: BLE001 - report, keep exit code
            print(f"complete output unavailable: {exc}")
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = find_agent(args.agent_bin)
    if agent is None:
        print(f"agent binary not found: {args.agent_bin}", file=sys.stderr)
        return 2
    plan = Path(args.plan).resolve()
    workspace = Path(args.workspace).resolve() if args.workspace else factory.toplevel(Path.cwd())
    root = factory.resolve_root(workspace)
    plan_id = plan.stem
    log_path = factory.state_dir(root, plan_id) / "supervisor.log"
    deadline = time.monotonic() + args.budget_minutes * 60
    chat_id: str | None = None
    last_sig: str | None = None
    same = 0
    failing = 0
    state = factory.load_state(root, plan_id)
    for iteration in range(1, args.max_iterations + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return finish(state, 5)
        prompt = first_prompt(plan) if state is None else CONTINUE_PROMPT
        cmd = [*agent, "-p", "--model", args.model, "--output-format", "json", "--trust", "--workspace", str(workspace)]
        if chat_id:
            cmd += ["--resume", chat_id]
        cmd.append(prompt)
        started = time.monotonic()
        code: int | str
        before = json.dumps(state, sort_keys=True)
        out_tail: list[str] = []
        err_tail: list[str] = []
        try:
            done = subprocess.run(
                cmd, cwd=workspace, text=True, capture_output=True, encoding="utf-8", errors="replace",
                timeout=remaining, check=False,
            )  # fmt: skip
            code = done.returncode
            if code != 0:
                out_tail, err_tail = tail(done.stdout), tail(done.stderr)
            found = extract_chat_id(done.stdout)
            chat_id = found or chat_id
        except subprocess.TimeoutExpired:
            code = "timeout"
        state = factory.load_state(root, plan_id)
        record: dict[str, Any] = {
            "iteration": iteration,
            "exit_code": code,
            "seconds": round(time.monotonic() - started, 2),
            "chat_id": chat_id,
            "next": summary(state),
        }
        if out_tail or err_tail:
            record["stdout_tail"], record["stderr_tail"] = out_tail, err_tail
        log_line(log_path, record)
        if code == "timeout":
            return finish(state, 5)
        unchanged = json.dumps(state, sort_keys=True) == before
        failing = failing + 1 if (code != 0 and unchanged) else 0
        if failing >= FAIL_ITERATIONS:
            return finish(state, 6)
        if state is None:
            continue
        if factory.is_finished(state):
            parked = any(s["status"] == "parked" for s in state["steps"].values())
            return finish(state, 3 if parked else 0)
        if any(st["status"] == "running" for st in state["steps"].values()):
            same, last_sig = 0, None  # workers may still be working; only budget bounds this
            continue
        sig = factory.progress_signature(state)
        same = same + 1 if sig == last_sig else 1
        last_sig = sig
        stall = state.get("stall") or {}
        if same >= STALL_ITERATIONS or (stall.get("stalled") and stall.get("signature") == sig):
            return finish(state, 4)
    return finish(state, 5)


if __name__ == "__main__":
    sys.exit(main())
