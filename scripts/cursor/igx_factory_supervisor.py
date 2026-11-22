# © Artur Czarnecki. All rights reserved.
"""Headless supervisor: drive a factory run with `cursor-agent -p` until it is done, parked or stalled.

Launch it from PowerShell (Git Bash breaks the Cursor hook wrapper). Exit codes:
0 run complete, 2 agent binary missing, 3 everything remaining is parked (escalations),
4 stalled, 5 max iterations or budget exhausted.
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

CONTINUE_PROMPT = (
    "FACTORY CONTINUE: run python scripts/cursor/igx_factory.py next and continue the factory run "
    "per /igx-native-run-plan; never ask the operator; park real blockers."
)
STALL_ITERATIONS = 3
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
    return (
        f"/igx-native-run-plan {plan}\n"
        f"Start the factory run: python scripts/cursor/igx_factory.py start {plan} , then follow the "
        "/igx-native-run-plan skill until the run completes; never ask the operator; park real blockers."
    )


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
    state = factory.load_state(root, plan_id)
    for iteration in range(1, args.max_iterations + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return finish(state, 5)
        prompt = first_prompt(plan) if state is None else CONTINUE_PROMPT
        cmd = [*agent, "-p", "--model", args.model, "--output-format", "json", "--workspace", str(workspace)]
        if chat_id:
            cmd += ["--resume", chat_id]
        cmd.append(prompt)
        started = time.monotonic()
        code: int | str
        try:
            done = subprocess.run(
                cmd, cwd=workspace, text=True, capture_output=True, encoding="utf-8", errors="replace",
                timeout=remaining, check=False,
            )  # fmt: skip
            code = done.returncode
            found = extract_chat_id(done.stdout)
            chat_id = found or chat_id
        except subprocess.TimeoutExpired:
            code = "timeout"
        state = factory.load_state(root, plan_id)
        log_line(
            log_path,
            {
                "iteration": iteration,
                "exit_code": code,
                "seconds": round(time.monotonic() - started, 2),
                "chat_id": chat_id,
                "next": summary(state),
            },
        )
        if code == "timeout":
            return finish(state, 5)
        if state is None:
            continue
        if factory.is_finished(state):
            parked = any(s["status"] == "parked" for s in state["steps"].values())
            return finish(state, 3 if parked else 0)
        sig = factory.progress_signature(state)
        same = same + 1 if sig == last_sig else 1
        last_sig = sig
        stall = state.get("stall") or {}
        if same >= STALL_ITERATIONS or (stall.get("stalled") and stall.get("signature") == sig):
            return finish(state, 4)
    return finish(state, 5)


if __name__ == "__main__":
    sys.exit(main())
