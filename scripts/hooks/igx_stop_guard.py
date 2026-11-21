# © Artur Czarnecki. All rights reserved.
"""Plan-driven Stop hook: keep an agent working while its active plan has open steps (stdlib only).

The active plan is named in the one-line file ``.tmp/session/autonomy/active-plan`` (a path to a
markdown plan with ``- [ ]`` / ``- [x]`` step lists). Without that file the hook allows every stop.

Hook mode (no arguments, Claude Code ``Stop`` event, JSON on stdin):

* allow when the last assistant message starts with ``ESCALATION:``;
* allow and clear the active plan when it starts with ``PLAN COMPLETE``;
* allow when no open step is left (the active plan is cleared);
* otherwise block with the next open step and "continue; stop only with ESCALATION:".

Loop limit: the third consecutive block without any checkbox change is allowed once with a warning
(and the counter resets), so the hook can never spin forever. ``stop_hook_active`` marks a turn that
already continues after a block; it is bounded by this counter instead of being allowed outright,
which would make the hook block only once. Any error allows the stop. ``IGX_HOOKS=off`` disables it.

CLI mode (used by the ``/igx-run-plan`` skill)::

    python scripts/hooks/igx_stop_guard.py start <plan.md>   # write the active-plan file
    python scripts/hooks/igx_stop_guard.py stop              # clear it
    python scripts/hooks/igx_stop_guard.py status            # print the next open step
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ACTIVE = Path(".tmp") / "session" / "autonomy" / "active-plan"
STATE = Path(".tmp") / "session" / "autonomy" / "stop-state.json"
MAX_BLOCKS = 3
_STEP = re.compile(r"^\s*[-*]\s+\[( |x|X)\]\s+(.*\S)\s*$")


def find_root(start: Path | None = None) -> Path | None:
    """Directory that holds ``.tmp/session/autonomy/active-plan``: project dir variable, else cwd and parents."""
    candidates: list[Path] = []
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        candidates.append(Path(env))
    here = (start or Path.cwd()).resolve()
    candidates += [here, *here.parents]
    for base in candidates:
        if (base / ACTIVE).is_file():
            return base
    return None


def project_root() -> Path:
    """Where a new active-plan file goes: project dir variable, else the nearest parent with ``.git``."""
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env)
    here = Path.cwd().resolve()
    return next((p for p in [here, *here.parents] if (p / ".git").exists()), here)


def read_plan(root: Path) -> tuple[Path, list[tuple[bool, str]]] | None:
    raw = (root / ACTIVE).read_text(encoding="utf-8").strip().splitlines()
    if not raw or not raw[0].strip():
        return None
    plan = Path(raw[0].strip())
    plan = plan if plan.is_absolute() else root / plan
    steps = _steps_of(plan)
    return plan, steps


def _load_state(root: Path) -> dict:
    try:
        data = json.loads((root / STATE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(root: Path, state: dict) -> None:
    try:
        (root / STATE).parent.mkdir(parents=True, exist_ok=True)
        (root / STATE).write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def _clear(root: Path) -> None:
    for rel in (ACTIVE, STATE):
        try:
            (root / rel).unlink()
        except OSError:
            pass


def _emit(payload: dict) -> int:
    sys.stdout.write(json.dumps(payload))
    return 0


def decide(root: Path, message: str) -> dict | None:
    """Hook decision as a JSON-able dict, or None to allow silently."""
    loaded = read_plan(root)
    if loaded is None:
        return None
    plan, steps = loaded
    text = (message or "").strip()
    if text.startswith("ESCALATION:"):
        _save_state(root, {})
        return None
    if text.startswith("PLAN COMPLETE"):
        _clear(root)
        return None
    open_steps = [s for done, s in steps if not done]
    if not open_steps:
        _clear(root)
        return None
    signature = f"{plan}|{sum(1 for done, _ in steps if done)}|{len(open_steps)}"
    state = _load_state(root)
    blocks = int(state.get("blocks", 0)) + 1 if state.get("signature") == signature else 1
    if blocks >= MAX_BLOCKS:
        _save_state(root, {})
        return {
            "systemMessage": (
                f"igx stop guard: {MAX_BLOCKS} blocks without progress on '{open_steps[0][:80]}'; "
                "allowing this stop once. The plan is still active (`/igx-run-plan stop` ends it)."
            )
        }
    _save_state(root, {"signature": signature, "blocks": blocks})
    return {
        "decision": "block",
        "reason": (
            f"Open plan step: {open_steps[0][:160]}. Continue; stop only with ESCALATION: <type> or PLAN COMPLETE."
        ),
    }


def hook() -> int:
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        return 0
    try:
        data = json.loads(sys.stdin.read() or "{}")
        root = find_root(Path(str(data.get("cwd"))) if isinstance(data, dict) and data.get("cwd") else None)
        if root is None or not isinstance(data, dict):
            return 0
        result = decide(root, str(data.get("last_assistant_message") or ""))
        return _emit(result) if result else 0
    except Exception:  # noqa: BLE001 - fail open: a broken guard must never trap a session
        return 0


def cli(argv: list[str]) -> int:
    command = argv[1]
    root = find_root() or project_root()
    if command == "start":
        if len(argv) < 3:
            sys.stderr.write("usage: igx_stop_guard.py start <plan.md>\n")
            return 2
        plan = Path(argv[2]).resolve()
        if not plan.is_file() or not any(not done for done, _ in _steps_of(plan)):
            sys.stderr.write(f"{argv[2]}: not found or has no open '- [ ]' step\n")
            return 1
        target = root / ACTIVE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(plan) + "\n", encoding="utf-8")
        _save_state(root, {})
        steps = _steps_of(plan)
        sys.stdout.write(f"active plan: {plan} ({sum(1 for d, _ in steps if not d)} open of {len(steps)} steps)\n")
        return 0
    if command == "stop":
        _clear(root)
        sys.stdout.write("active plan cleared\n")
        return 0
    if command == "status":
        loaded = read_plan(root) if (root / ACTIVE).is_file() else None
        if loaded is None:
            sys.stdout.write("no active plan\n")
            return 0
        nxt = next((s for done, s in loaded[1] if not done), None)
        sys.stdout.write(f"active plan: {loaded[0]}; next open step: {nxt or 'none'}\n")
        return 0
    sys.stderr.write("usage: igx_stop_guard.py [start <plan.md> | stop | status]\n")
    return 2


def _steps_of(plan: Path) -> list[tuple[bool, str]]:
    out = []
    for line in plan.read_text(encoding="utf-8").splitlines():
        m = _STEP.match(line)
        if m:
            out.append((m.group(1) != " ", m.group(2)))
    return out


def main(argv: list[str]) -> int:
    return cli(argv) if len(argv) > 1 else hook()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
