# © Artur Czarnecki. All rights reserved.
"""Cursor ``stop`` hook: keep a factory run going while READY or pending steps exist.

Reads only the hook payload and the factory ``state.json``; never the transcript.
Fail-open: any error prints nothing and exits 0. ``IGX_HOOKS=off`` disables it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

STALL_LIMIT = 3
NEXT_CMD = "python scripts/cursor/igx_factory.py next"


def _import_factory() -> Any:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cursor"))
    import igx_factory  # type: ignore[import-not-found]

    return igx_factory


def discover_root(payload: dict[str, Any]) -> Path | None:
    start = payload.get("cwd")
    if not start:
        roots = payload.get("workspace_roots") or []
        start = roots[0] if roots else None
    if not start:
        return None
    start_path = Path(str(start))
    top: Path | None = None
    done = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start_path,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if done.returncode == 0 and done.stdout.strip():
        top = Path(done.stdout.strip())
    else:
        for parent in (start_path, *start_path.parents):
            if (parent / ".git").exists():
                top = parent
                break
    if top is None:
        return None
    pointer = top / ".tmp" / "factory_root"
    if pointer.is_file():
        text = pointer.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    return top


def _toplevel(path: str) -> Path | None:
    done = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if done.returncode == 0 and done.stdout.strip():
        return Path(done.stdout.strip())
    return None


def may_claim(payload: dict[str, Any], root: Path) -> bool:
    """Claim an unowned run only from a workspace root that is the run's state root.

    A factory worktree (``f-<plan>-*`` or one with a ``.tmp/factory_root`` pointer) never claims.
    """
    for raw in payload.get("workspace_roots") or []:
        try:
            top = _toplevel(str(raw))
            if top is None or (top / ".tmp" / "factory_root").is_file() or top.name.startswith("f-"):
                continue
            if top.resolve() == root.resolve():
                return True
        except OSError:
            continue
    return False


def followup_text(nxt: dict[str, Any]) -> str:
    parts = [f"FACTORY CONTINUE: run {NEXT_CMD}"]
    if nxt["ready"]:
        parts.append("READY: " + ", ".join(e["id"] for e in nxt["ready"]))
    if nxt["pending_actions"]:
        todo = {"received": "review", "accepted": "integrate"}
        items = [f"{e['id']} ({todo.get(e['status'], e['status'])})" for e in nxt["pending_actions"]]
        parts.append("PENDING: " + ", ".join(items))
    return "; ".join(parts)


def decide(payload: dict[str, Any], factory: Any, root: Path) -> str | None:
    if payload.get("status") != "completed":
        return None
    conv = payload.get("conversation_id")
    if not conv:
        return None
    runs = factory.find_active_runs(root)
    mine = [r for r in runs if r.get("owner_conversation") == conv]
    unowned = [r for r in runs if r.get("owner_conversation") is None]
    state = mine[0] if mine else (unowned[0] if unowned and may_claim(payload, root) else None)
    if state is None:
        return None
    if state.get("owner_conversation") is None:
        state["owner_conversation"] = conv
        factory.save_state(state)
    key = [conv, payload.get("generation_id"), payload.get("loop_count")]
    last = state.get("last_decision")
    if last and last.get("key") == key:
        return last.get("output")
    nxt = factory.cmd_next(state)
    output: str | None = None
    if nxt["ready"] or nxt["pending_actions"]:
        sig = factory.progress_signature(state)
        stall = state.get("stall") or {"signature": None, "count": 0}
        stall["count"] = stall.get("count", 0) + 1 if stall.get("signature") == sig else 1
        stall["signature"] = sig
        stall["stalled"] = stall["count"] >= STALL_LIMIT
        state["stall"] = stall
        if not stall["stalled"]:
            output = json.dumps({"followup_message": followup_text(nxt)})
    state["last_decision"] = {"key": key, "output": output}
    factory.save_state(state)
    return output


def main() -> int:
    try:
        if os.environ.get("IGX_HOOKS", "").lower() == "off":
            return 0
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            return 0
        root = discover_root(payload)
        if root is None:
            return 0
        output = decide(payload, _import_factory(), root)
        if output:
            sys.stdout.write(output)
    except Exception:  # noqa: BLE001 - fail-open by contract
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
