# © Artur Czarnecki. All rights reserved.
"""Cursor ``stop`` adapter for the plan-driven stop guard (stdlib only).

Loaded by ``igx_stop_guard.py`` when the stop payload is Cursor-native (no ``last_assistant_message``).

Cursor contract (verified on a live install): the ``stop`` hook receives ``conversation_id``,
``generation_id``, ``loop_count``, ``status`` (``completed`` | ``aborted`` | ``error``),
``transcript_path`` and ``workspace_roots`` (no ``cwd``, no assistant text). The only way to continue
is the output key ``followup_message``; ``systemMessage`` is ignored. The transcript is JSONL with
``{"role": ..., "message": {"content": [{"type": "text"|"tool_use", ...}]}}`` lines.

Rules implemented here:

* only a ``completed`` stop of the conversation that owns the plan is continued; the owner is the
  conversation whose transcript shows ``igx_stop_guard.py start`` (the active-plan file is per checkout);
* the final assistant text of the current turn (after the last user line) is checked for the end markers
  ``ESCALATION:``, ``PLAN COMPLETE`` and ``WAITING:`` at the start of any line;
* ``WAITING:`` is honoured only while the transcript holds a ``run_in_background`` Task spawn;
* a stop without checkbox progress is followed up once, warned once, then allowed and the plan is marked
  stalled; a total follow-up cap bounds the whole activation;
* the transcript is scanned tail-first (last ``_SCAN_TAIL`` bytes) with a full streaming scan as fallback,
  so an owner claim or background spawn older than the tail is still found;
* any problem allows the stop. State: ``.tmp/session/autonomy/cursor-stop-state.json``; diagnostics:
  ``cursor-stop-last.json`` (decision and reason code only).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

STATE = Path(".tmp") / "session" / "autonomy" / "cursor-stop-state.json"
DIAG = Path(".tmp") / "session" / "autonomy" / "cursor-stop-last.json"
_TURN_TAIL = 64 * 1024
_SCAN_TAIL = 16 * 1024 * 1024
_MARKER = re.compile(r"^[ \t>*_#-]*(ESCALATION:(?![ \t]*<)|PLAN COMPLETE\b|WAITING:)", re.M)
_BACKGROUND = re.compile(r'"run_in_background"\s*:\s*true')
_START = "igx_stop_guard.py start"
_DONE_STATUS = ("completed", "cancelled")


def frontmatter_todos(text: str) -> list[tuple[bool, str]]:
    """Cursor plan todos from the YAML frontmatter as ``(done, content)``; ``[]`` when there are none."""
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return []
    block = lines[1:end]
    start = next((i for i, line in enumerate(block) if line.rstrip() == "todos:"), None)
    if start is None:
        return []
    items: list[dict[str, str]] = []
    item_indent: int | None = None
    field_indent = 0
    pending: tuple[dict[str, str], str, int] | None = None  # (item, key, indent) of an open block scalar
    folded: list[str] = []

    def flush() -> None:
        nonlocal pending, folded
        if pending is not None:
            pending[0][pending[1]] = " ".join(folded)
        pending, folded = None, []

    for line in block[start + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if pending is not None:
            if indent > pending[2]:
                folded.append(line.strip())
                continue
            flush()
        if not line[0].isspace() and not line.startswith("-"):
            break
        item_start = re.match(r"^\s*-\s+(\w+):\s*(.*?)\s*$", line)
        if item_start and (item_indent is None or indent <= item_indent):
            item_indent = indent
            field_indent = line.index(item_start.group(1))
            key, value = item_start.group(1), item_start.group(2)
            items.append({key: value})
        else:
            field = re.match(r"^\s+(\w+):\s*(.*?)\s*$", line)
            if item_start or not (field and items) or indent > field_indent:
                continue
            key, value = field.group(1), field.group(2)
            items[-1][key] = value
        if value in (">", ">-", "|", "|-"):
            pending = (items[-1], key, indent)
    flush()
    out: list[tuple[bool, str]] = []
    for item in items:
        content = _unquote(item.get("content") or item.get("id") or "")
        if content:
            out.append((_unquote(item.get("status", "")).lower() in _DONE_STATUS, content))
    return out


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        return inner.replace('\\"', '"') if value[0] == '"' else inner.replace("''", "'")
    return value


def _tail(path: Path, size: int) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        length = handle.tell()
        handle.seek(max(0, length - size))
        return handle.read().decode("utf-8", errors="ignore")


def _text_of(obj: dict) -> str:
    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else obj.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type", "text") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(part for part in parts if part.strip()).strip()


def _role(obj: dict) -> str:
    message = obj.get("message")
    role = obj.get("role") or (message.get("role") if isinstance(message, dict) else "")
    return str(role or "").lower()


def turn_text(path: Path) -> str:
    """Last non-empty assistant text segment of the current turn (after the last user line)."""
    text = ""
    for line in _tail(path, _TURN_TAIL).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        role = _role(obj)
        if role == "user":
            text = ""
        elif role == "assistant":
            found = _text_of(obj)
            if found:
                text = found
    return text


def _tool_use_inputs(line: str) -> list[str]:
    """Serialised ``input`` of every assistant ``tool_use`` item on a transcript line."""
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict) or _role(obj) != "assistant":
        return []
    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else obj.get("content")
    return [
        json.dumps(item.get("input"))
        for item in (content if isinstance(content, list) else [])
        if isinstance(item, dict) and item.get("type") == "tool_use"
    ]


def _is_start_claim(line: str) -> bool:
    return any(_START in text for text in _tool_use_inputs(line))


def _is_background(line: str) -> bool:
    return any(_BACKGROUND.search(text) for text in _tool_use_inputs(line))


def _find_line(path: Path, needle: str, accept: Callable[[str], bool]) -> bool:
    """True when an accepted line holds ``needle``: the tail first, then a full streaming scan."""
    for line in _tail(path, _SCAN_TAIL).splitlines():
        if needle in line and accept(line):
            return True
    if path.stat().st_size <= _SCAN_TAIL:
        return False
    key = needle.encode()
    with path.open("rb") as handle:
        for raw in handle:
            if len(raw) <= _SCAN_TAIL and key in raw and accept(raw.decode("utf-8", errors="ignore")):
                return True
    return False


def _claims_start(path: Path) -> bool:
    """True when an assistant ``tool_use`` input holds the start command (plain text does not claim)."""
    return _find_line(path, _START, _is_start_claim)


def _markers(text: str) -> set[str]:
    return {found.rstrip(":") for found in _MARKER.findall(text)}


def _load(root: Path) -> dict:
    try:
        data = json.loads((root / STATE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(root: Path, state: dict) -> None:
    try:
        (root / STATE).parent.mkdir(parents=True, exist_ok=True)
        (root / STATE).write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def _diag(root: Path, data: dict, result: dict | None, reason: str, open_count: int) -> None:
    loop = data.get("loop_count")
    status = data.get("status")
    payload = {
        "decision": "followup" if result else "allow",
        "reason": reason,
        "loop_count": loop if isinstance(loop, int) else None,
        "status": status if isinstance(status, str) else None,
        "open": open_count,
    }
    try:
        (root / DIAG).parent.mkdir(parents=True, exist_ok=True)
        (root / DIAG).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def decide_cursor(
    root: Path,
    data: dict,
    plan: Path,
    steps: list[tuple[bool, str]],
    active_path: Path,
    max_blocks: int,
) -> tuple[dict | None, bool]:
    """Return ``(result_or_None, clear)``; ``clear`` asks the caller to drop the active plan."""
    open_steps = [step for done, step in steps if not done]

    def allow(reason: str) -> tuple[dict | None, bool]:
        _diag(root, data, None, reason, len(open_steps))
        return None, False

    completed = str(data.get("status") or "").strip().lower() == "completed"
    conversation = str(data.get("conversation_id") or data.get("session_id") or "")
    transcript = data.get("transcript_path")
    if not conversation or not isinstance(transcript, str) or not transcript.strip():
        return allow("no_conversation" if completed else "status")
    path = Path(transcript)
    if not path.is_file():
        return allow("no_transcript" if completed else "status")
    try:
        activation = f"{plan}|{active_path.stat().st_mtime_ns}"
    except OSError:
        return allow("no_activation" if completed else "status")
    state: dict[str, object] = _load(root)
    if state.get("activation") != activation:
        state = {"activation": activation}
    owner = state.get("owner")
    if not owner:
        if not _claims_start(path):
            return allow("not_owner" if completed else "status")
        state["owner"] = conversation
        _save(root, state)
    elif owner != conversation:
        return allow("other_conversation" if completed else "status")
    if not completed:
        return allow("status")

    done_count = len(steps) - len(open_steps)
    signature = f"{done_count}|{len(open_steps)}"
    generation = str(data.get("generation_id") or "")
    event = "|".join((conversation, generation, str(data.get("loop_count", "")), signature))
    if generation and state.get("last_event") == event:
        replay = state.get("last_result")
        result = replay if isinstance(replay, dict) else None
        _diag(root, data, result, "replay", len(open_steps))
        return result, False

    text = turn_text(path)
    if not text:
        return allow("no_text")
    found = _markers(text)
    if "ESCALATION" in found:
        return allow("escalation")
    if not open_steps:
        try:
            (root / STATE).unlink()
        except OSError:
            pass
        _diag(root, data, None, "plan_done", 0)
        return None, True
    if "WAITING" in found and "PLAN COMPLETE" not in found and _find_line(path, "run_in_background", _is_background):
        return allow("waiting")

    step = open_steps[0]
    if state.get("signature") == signature:
        count = int(str(state.get("n", 0))) + 1
    else:
        count = 1
        state["stalled"] = False
    followups = int(str(state.get("followups", 0)))
    state.update({"signature": signature, "n": count, "followups": followups, "last_event": event})

    def finish(result: dict | None, reason: str) -> tuple[dict | None, bool]:
        state["last_result"] = result
        _save(root, state)
        _diag(root, data, result, reason, len(open_steps))
        return result, False

    if state.get("stalled"):
        return finish(None, "stalled")
    if followups >= max(10, 3 * len(steps)):
        state["stalled"] = True
        return finish(None, "cap")
    if count >= max_blocks:
        state["stalled"] = True
        return finish(None, "stall")
    state["followups"] = followups + 1
    if count == 1:
        if "PLAN COMPLETE" in found:
            reason, lead = "premature_complete", f"PLAN COMPLETE is premature: {len(open_steps)} step(s) still open. "
        elif "WAITING" in found:
            reason, lead = "waiting_unbacked", "WAITING needs a background worker spawned in this conversation. "
        else:
            reason, lead = "followup", ""
        message = (
            f"{lead}Open plan step: {step[:160]}. Continue. End the turn only with ESCALATION: <type>, "
            "WAITING: <background task ids> while background workers run, or PLAN COMPLETE when every step is ticked."
        )
    else:
        reason = "stall_warning"
        message = (
            f"No progress on '{step[:80]}' in {plan.name}. Tick verified steps in the plan file or end the turn "
            "with ESCALATION: <type>. The next stop without progress is allowed and marks the plan stalled."
        )
    return finish({"followup_message": message}, reason)
