# © Artur Czarnecki. All rights reserved.
"""Plan-driven Stop hook: keep an agent working while its active plan has open steps (stdlib only).

The active plan is named in the one-line file ``.tmp/session/autonomy/active-plan`` (a path to a
markdown plan with ``- [ ]`` / ``- [x]`` step lists). Without that file the hook allows every stop.

Hook mode (no arguments, Claude Code ``Stop`` event or Cursor native ``stop``, JSON on stdin):

* allow when ``status`` is ``error`` or ``aborted`` (a failed or cancelled run is never continued);
* allow when no visible assistant text exists (payload or transcript tail), because an escalation
  cannot be told apart from an ordinary stop;
* allow when that text starts with ``ESCALATION:``;
* allow and clear the active plan when it starts with ``PLAN COMPLETE``;
* allow when no open step is left (the active plan is cleared);
* otherwise continue with the next open step.

Claude-shaped input (non-empty ``last_assistant_message``) continues with ``decision: block`` and
``reason`` only. Cursor-native input (no such message) is decided by ``igx_cursor_stop.py`` (plan
ownership, end markers, stall states) and continues with ``followup_message`` only; it also reads
Cursor plan files with frontmatter ``todos:``. ``.cursor/hooks.json`` registers ``stop`` with
``loop_limit: null`` so a plan that keeps ticking steps is not cut at Cursor's default of 5.
A numeric ``loop_limit`` that drops further follow-ups leaves the plan active; that is not
``PLAN COMPLETE``.

Loop limit: the third consecutive block without any checkbox change is allowed once with a warning
(and the counter resets), so the hook can never spin forever. A second delivery of the same
``conversation_id`` and ``generation_id`` does not advance the counter. Any error allows the stop.
``IGX_HOOKS=off`` disables it.

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
from urllib.parse import unquote

ACTIVE = Path(".tmp") / "session" / "autonomy" / "active-plan"
STATE = Path(".tmp") / "session" / "autonomy" / "stop-state.json"
KEYS = Path(".tmp") / "session" / "autonomy" / "stop-last-keys.json"
MAX_BLOCKS = 3
_TAIL_BYTES = 65536
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


def _note_keys(data: dict) -> None:
    """Record stdin key names only, under ``.tmp``, so a live stop can be checked without polling."""
    try:
        root = project_root()
        path = root / KEYS
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "keys": sorted(str(key) for key in data),
            "status": data.get("status") if isinstance(data.get("status"), str) else None,
            "loop_count": data.get("loop_count") if isinstance(data.get("loop_count"), int) else None,
            "hook_event_name": data.get("hook_event_name") if isinstance(data.get("hook_event_name"), str) else None,
            "has_transcript": bool(data.get("transcript_path")),
            "has_message": bool(str(data.get("last_assistant_message") or "").strip()),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    except (OSError, TypeError):
        pass


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") if isinstance(item.get("text"), str) else item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                elif text is not None:
                    parts.append(_content_text(text))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        nested = content.get("text") if isinstance(content.get("text"), str) else content.get("content")
        return _content_text(nested)
    return ""


def _assistant_text(obj: object) -> str:
    if not isinstance(obj, dict):
        return ""
    message = obj.get("message")
    role = str(obj.get("role") or obj.get("type") or "")
    if role.lower() == "assistant":
        content = obj.get("content")
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content is None:
            content = obj.get("text")
        return _content_text(content)
    if isinstance(message, dict) and str(message.get("role") or "").lower() == "assistant":
        return _content_text(message.get("content"))
    return ""


def _tail_assistant(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    chunk = raw[-_TAIL_BYTES:]
    text = ""
    for line in chunk.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            found = _assistant_text(json.loads(line))
        except ValueError:
            continue
        if found:
            text = found
    return text


def _visible_message(data: dict) -> str:
    raw = data.get("last_assistant_message")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    transcript = data.get("transcript_path")
    if isinstance(transcript, str) and transcript.strip():
        return _tail_assistant(Path(transcript))
    return ""


def _native_cursor(data: dict) -> bool:
    raw = data.get("last_assistant_message")
    if isinstance(raw, str) and raw.strip():
        return False
    return "loop_count" in data or data.get("hook_event_name") == "stop"


def _event_key(data: dict, signature: str) -> str:
    loop = data.get("loop_count")
    return "|".join(
        (
            str(data.get("conversation_id") or ""),
            str(data.get("generation_id") or ""),
            str(loop if loop is not None else ""),
            signature,
        )
    )


def _cursor_module():
    """Import the Cursor adapter; the script may run through ``runpy.run_path`` without its dir on sys.path."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import igx_cursor_stop

    return igx_cursor_stop


def decide(root: Path, data: dict) -> dict | None:
    """Hook decision as a JSON-able dict, or None to allow silently."""
    if not isinstance(data, dict):
        return None
    loaded = read_plan(root)
    if loaded is None:
        return None
    if _native_cursor(data):
        plan, steps = loaded
        result, clear = _cursor_module().decide_cursor(root, data, plan, steps, root / ACTIVE, MAX_BLOCKS)
        if clear:
            _clear(root)
        return result
    status = str(data.get("status") or "")
    if status.strip().lower() in ("error", "aborted"):
        return None
    text = _visible_message(data)
    if not text:
        return None
    plan, steps = loaded
    if text.startswith("ESCALATION:"):
        _save_state(root, {})
        return None
    if text.startswith("PLAN COMPLETE"):
        _clear(root)
        return None
    open_steps = [step for done, step in steps if not done]
    if not open_steps:
        _clear(root)
        return None
    signature = f"{plan}|{sum(1 for done, _ in steps if done)}|{len(open_steps)}"
    state = _load_state(root)
    event = _event_key(data, signature)
    tracked = bool(data.get("conversation_id") or data.get("generation_id"))
    if tracked and state.get("last_event") == event and isinstance(state.get("last_result"), dict):
        return state["last_result"]
    blocks = int(state.get("blocks", 0)) + 1 if state.get("signature") == signature else 1
    if blocks >= MAX_BLOCKS:
        result = {
            "systemMessage": (
                f"igx stop guard: {MAX_BLOCKS} blocks without progress on '{open_steps[0][:80]}'; "
                "allowing this stop once. The plan is still active (`/igx-run-plan stop` ends it)."
            )
        }
        _save_state(root, {"last_event": event, "last_result": result} if tracked else {})
        return result
    reason = f"Open plan step: {open_steps[0][:160]}. Continue; stop only with ESCALATION: <type> or PLAN COMPLETE."
    result = {"followup_message": reason} if _native_cursor(data) else {"decision": "block", "reason": reason}
    _save_state(
        root,
        {"signature": signature, "blocks": blocks, "last_event": event, "last_result": result},
    )
    return result


def _root_path(value: str) -> Path:
    """Workspace root string to a path; URI-style Windows roots (``/d:/x``, ``file:///d:/x``) become ``D:/x``."""
    text = value.strip()
    if text.lower().startswith("file://"):
        text = unquote(text[7:])
    drive = re.match(r"^/([A-Za-z]):[/\\]", text)
    if drive:
        text = drive.group(1).upper() + text[2:]
    return Path(text)


def _discover_root(data: dict) -> Path | None:
    start = str(data.get("cwd") or "")
    if start:
        return find_root(Path(start))
    roots = data.get("workspace_roots")
    for value in roots if isinstance(roots, list) else []:
        if isinstance(value, str) and value.strip():
            found = find_root(_root_path(value))
            if found is not None:
                return found
    return find_root(None)


def hook() -> int:
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        return 0
    try:
        data = json.loads(sys.stdin.read() or "{}")
        if not isinstance(data, dict):
            return 0
        _note_keys(data)
        root = _discover_root(data)
        if root is None:
            return 0
        result = decide(root, data)
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
    text = plan.read_text(encoding="utf-8")
    for line in text.splitlines():
        m = _STEP.match(line)
        if m:
            out.append((m.group(1) != " ", m.group(2)))
    if not out:
        try:
            return _cursor_module().frontmatter_todos(text)
        except ImportError:
            return []
    return out


def main(argv: list[str]) -> int:
    return cli(argv) if len(argv) > 1 else hook()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
