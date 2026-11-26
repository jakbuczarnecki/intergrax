#!/usr/bin/env python3
"""Antigravity Stop hook for plan-driven Intergrax autonomy (stdlib only).

Reads `.tmp/session/autonomy/active-plan` (the same active-plan pointer managed by
`python scripts/hooks/igx_stop_guard.py start|stop|status`), without modifying the
hash-pinned Claude/Cursor stop guard script.

Behavior:
- Without `.tmp/session/autonomy/active-plan`, emits `{}` (allows stop silently).
- Only continues when `terminationReason` is `NO_TOOL_CALL` (never loops on ERROR,
  USER_CANCELED, MAX_INVOCATIONS, etc.).
- Allows stop when latest assistant text starts with `ESCALATION:` or `PLAN COMPLETE`.
- Clears `.tmp/session/autonomy/active-plan` when all `- [ ]` steps are ticked.
- Caps consecutive continuations without checkbox progress at 3 (`MAX_BLOCKS`),
  using an Antigravity-specific state file `.tmp/session/autonomy/agy-stop-state.json`.
- `IGX_HOOKS=off` disables the guard; any unexpected error fails open.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ACTIVE = Path(".tmp") / "session" / "autonomy" / "active-plan"
STATE = Path(".tmp") / "session" / "autonomy" / "agy-stop-state.json"
MAX_BLOCKS = 3
_TAIL_BYTES = 65536
_STEP = re.compile(r"^\s*[-*]\s+\[( |x|X)\]\s+(.*\S)\s*$")
_PREFIX_STRIP = re.compile(r"^[\s>*_#-]+")


def _find_root(data: dict[str, Any]) -> Path | None:
    starts: list[Path] = []
    cwd = data.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        starts.append(Path(cwd))
    workspaces = data.get("workspacePaths")
    if isinstance(workspaces, list):
        for item in workspaces:
            if isinstance(item, str) and item.strip():
                starts.append(Path(item))
    starts.append(Path.cwd())
    seen: set[Path] = set()
    for start in starts:
        try:
            resolved = start.resolve()
        except OSError:
            continue
        for candidate in (resolved, *resolved.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / ACTIVE).is_file():
                return candidate
    return None


def _read_plan(root: Path) -> tuple[Path, list[tuple[bool, str]]] | None:
    try:
        lines = (root / ACTIVE).read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None
    if not lines or not lines[0].strip():
        return None
    plan = Path(lines[0].strip())
    if not plan.is_absolute():
        plan = root / plan
    try:
        text = plan.read_text(encoding="utf-8")
    except OSError:
        return None
    steps: list[tuple[bool, str]] = []
    for line in text.splitlines():
        match = _STEP.match(line)
        if match:
            steps.append((match.group(1) != " ", match.group(2)))
    return plan, steps


def _load_state(root: Path) -> dict[str, Any]:
    try:
        parsed = json.loads((root / STATE).read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(root: Path, state: dict[str, Any]) -> None:
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


def _extract_text(obj: object) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, list):
        parts = [_extract_text(item) for item in obj]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(obj, dict):
        for key in ("text", "content", "response", "markdown"):
            val = obj.get(key)
            if isinstance(val, (str, list, dict)):
                found = _extract_text(val)
                if found:
                    return found
    return ""


def _assistant_text_from_entry(entry: object) -> str:
    if not isinstance(entry, dict):
        return ""
    role = str(entry.get("role") or entry.get("type") or entry.get("author") or "").lower()
    if role in ("assistant", "model", "agent", "planner_response"):
        for key in ("content", "text", "response", "message"):
            if key in entry:
                found = _extract_text(entry[key])
                if found:
                    return found
    msg = entry.get("message")
    if isinstance(msg, dict) and str(msg.get("role") or "").lower() in ("assistant", "model"):
        return _extract_text(msg.get("content") or msg.get("text"))
    return ""


def _last_assistant_text(data: dict[str, Any]) -> str:
    transcript = data.get("transcriptPath")
    if not isinstance(transcript, str) or not transcript.strip():
        return ""
    path = Path(transcript)
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    text = ""
    for line in raw[-_TAIL_BYTES:].decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        found = _assistant_text_from_entry(parsed)
        if found:
            text = found
    return text


def _has_marker(text: str, marker: str) -> bool:
    for line in text.splitlines():
        cleaned = _PREFIX_STRIP.sub("", line.strip())
        if cleaned.startswith(marker):
            return True
    return False


def decide(data: dict[str, Any]) -> dict[str, Any]:
    reason_enum = str(data.get("terminationReason") or "")
    if reason_enum and reason_enum != "NO_TOOL_CALL":
        return {}
    root = _find_root(data)
    if root is None:
        return {}
    loaded = _read_plan(root)
    if loaded is None:
        return {}
    plan, steps = loaded
    open_steps = [step for done, step in steps if not done]
    assistant_text = _last_assistant_text(data)
    if assistant_text and _has_marker(assistant_text, "ESCALATION:"):
        _save_state(root, {})
        return {}
    if not open_steps:
        _clear(root)
        return {}
    if assistant_text and _has_marker(assistant_text, "PLAN COMPLETE"):
        return {
            "decision": "continue",
            "reason": (
                f"PLAN COMPLETE was printed, but open step remains: {open_steps[0][:160]}. "
                "Complete and tick `- [x]` or stop with ESCALATION: <type>."
            ),
        }
    signature = f"{plan}|{sum(1 for done, _ in steps if done)}|{len(open_steps)}"
    state = _load_state(root)
    blocks = int(state.get("blocks", 0)) + 1 if state.get("signature") == signature else 1
    if blocks >= MAX_BLOCKS:
        _save_state(root, {})
        return {}
    _save_state(root, {"signature": signature, "blocks": blocks})
    return {
        "decision": "continue",
        "reason": (
            f"Open plan step: {open_steps[0][:160]}. Continue; stop only with ESCALATION: <type> or PLAN COMPLETE."
        ),
    }


def main() -> int:
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        sys.stdout.write("{}")
        return 0
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        result = decide(data) if isinstance(data, dict) else {}
        sys.stdout.write(json.dumps(result))
        return 0
    except Exception:  # noqa: BLE001 - fail open: never trap a session
        sys.stdout.write("{}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
