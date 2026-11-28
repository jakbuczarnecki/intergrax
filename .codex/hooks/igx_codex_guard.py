# © Artur Czarnecki. All rights reserved.
"""Codex-only lifecycle adapter. Stdlib only; state belongs to one Codex session.

Hook decisions are guardrails, not an OS sandbox or a shell security parser.
No Claude, Cursor, or Antigravity state is read or written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SECRET_NAMES = {".env", ".env.local", ".env.production"}
AGENTS = {"igx-researcher", "igx-architect", "igx-implementer", "igx-reviewer", "ci-investigator"}
PLAN_COMMAND = re.compile(
    r'\$(?:intergrax-codex:)?codex-igx-run-plan(?:[ \t]+("[^"\r\n]+"|\x27[^\x27\r\n]+\x27|[^\s`]+))?'
)
STEP = re.compile(r"^\s*[-*]\s+\[( |x|X)\]\s+(.*\S)\s*$", re.MULTILINE)
SHELL_DENY = (
    (r"\bgit(?:\.exe)?\b[^\n;&|]*\bpush\b", "Push remains a human action under the shared core."),
    (r"\bgit(?:\.exe)?\b[^\n;&|]*\breset\b[^\n;&|]*--hard\b", "Hard reset is forbidden."),
    (r"--no-verify\b|\bINTERGRAX_SKIP_", "Governance bypass is forbidden."),
    (
        r"\bgit(?:\.exe)?\b[^\n;&|]*\b(?:checkout|switch)\b",
        "Use the approved worktree procedure; do not switch the shared checkout.",
    ),
    (
        r"\bgh(?:\.exe)?\b[^\n;&|]*\b(?:pr|repo)\b[^\n;&|]*\b(?:create|new|merge|edit|delete)\b",
        "Remote delivery/settings remain a human action under the shared core.",
    ),
)


def project_root(data: dict[str, Any]) -> Path:
    here = Path(str(data.get("cwd") or Path.cwd())).resolve()
    return next((p for p in (here, *here.parents) if (p / ".git").exists()), Path(__file__).resolve().parents[2])


def state_path(root: Path, session: str) -> Path:
    key = hashlib.sha256(session.encode("utf-8")).hexdigest()[:32]
    return root / ".tmp" / "session" / "codex" / key / "plan.json"


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("Invalid Codex plan state")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(path)


def deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def context(event: str, message: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": message}}


def secret_path(raw: str) -> bool:
    return raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower() in SECRET_NAMES


def pre_tool(data: dict[str, Any], root: Path) -> dict[str, Any]:
    name = str(data.get("tool_name") or "").rsplit(".", 1)[-1]
    args = data.get("tool_input") or {}
    if not isinstance(args, dict):
        return deny("Invalid tool arguments; correct the call before retrying.")
    if name in {"spawn_agent", "Agent"}:
        agent = str(args.get("agent_type") or args.get("subagent_type") or "")
        if agent not in AGENTS:
            return deny("Use a registered Intergrax Codex profile and the operator-authorized delegation scope.")
    if name in {"Read", "read_file"} or name.endswith("__read_file"):
        if secret_path(str(args.get("file_path") or args.get("path") or "")):
            return deny("Reading secret .env files is forbidden by project policy.")
    if name in {"apply_patch", "Edit", "Write"}:
        patch = str(args.get("command") or args.get("patch") or args.get("input") or "")
        paths = re.findall(r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$", patch, re.MULTILINE)
        paths.append(str(args.get("file_path") or args.get("path") or ""))
        if any(secret_path(path) for path in paths):
            return deny("Secret environment files are outside the Codex edit scope.")
    if name in {"Bash", "exec_command", "shell", "shell_command"}:
        command = str(args.get("command") or args.get("cmd") or "")
        for pattern, reason in SHELL_DENY:
            if re.search(pattern, command, re.IGNORECASE):
                return deny(reason)
        if re.search(r"(?<![\w.])\.env(?:\.local|\.production)?(?![\w.])", command, re.IGNORECASE):
            return deny("Do not access secret environment files through the shell.")
        # Reuse the existing pure dump check without invoking another harness's hook.
        path = root / "scripts" / "hooks" / "igx_bash_guard.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("igx_codex_dump_check", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # PowerShell bounded Get-Content reads are the Codex equivalent of Read ranges.
            for statement in re.split(r"[;\r\n]|&&|\|\|", command):
                bound = re.search(r"(?i)(?<!\w)(?:-TotalCount|-Head|-First|-Last|-Tail)\s+(\d+)", statement)
                bounded = bound is not None and 0 < int(bound.group(1)) <= 600
                reason = "" if bounded else module.check(statement, str(args.get("workdir") or root), posix=False)
                if reason:
                    return deny(reason)
    return {}


def plan_prompt(data: dict[str, Any], root: Path, path: Path) -> dict[str, Any]:
    match = PLAN_COMMAND.search(str(data.get("prompt") or ""))
    if not match:
        return {}
    argument = (match.group(1) or "").strip("\"'")
    if argument == "stop":
        path.unlink(missing_ok=True)
        return context("UserPromptSubmit", "Codex plan execution stopped for this session.")
    if argument == "status":
        return context("UserPromptSubmit", f"Codex plan state: {json.dumps(load_state(path))}")
    if not argument:
        return context(
            "UserPromptSubmit", "Use $intergrax-codex:codex-igx-run-plan <approved-plan.md>, stop, or status."
        )
    plan = Path(argument)
    plan = (root / plan).resolve() if not plan.is_absolute() else plan.resolve()
    if not plan.is_relative_to(root):
        return context("UserPromptSubmit", "Plan not activated: select an approved plan inside this worktree.")
    steps = STEP.findall(plan.read_text(encoding="utf-8"))
    if not any(flag == " " for flag, _ in steps):
        return context("UserPromptSubmit", "Plan not activated: no open checklist steps.")
    save_state(path, {"plan": str(plan.relative_to(root)), "stalls": 0})
    return context("UserPromptSubmit", f"Codex plan activated for this session: {plan.relative_to(root)}")


def stop(data: dict[str, Any], root: Path, path: Path) -> dict[str, Any]:
    state = load_state(path)
    if not state or state.get("paused"):
        return {}
    plan = (root / state["plan"]).resolve()
    if not plan.is_relative_to(root):
        raise ValueError("Active plan escaped the worktree")
    text = plan.read_text(encoding="utf-8")
    steps = STEP.findall(text)
    if not steps:
        raise ValueError("Active plan no longer has a checklist")
    remaining = [title for flag, title in steps if flag == " "]
    if not remaining:
        path.unlink(missing_ok=True)
        return {}
    message = str(data.get("last_assistant_message") or "")
    if re.search(r"(?m)^\s*(?:[>#*_-]\s*)*ESCALATION:", message):
        return {}
    fingerprint = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    # Codex may continue within the same turn_id. Count every Stop invocation;
    # treating the turn id as a unique delivery id would defeat the stall cap.
    stalls = int(state.get("stalls", 0)) + 1 if state.get("fingerprint") == fingerprint else 1
    response = (
        {"systemMessage": "Codex plan stalled after three stops without checklist progress; inspect before resuming."}
        if stalls >= 3
        else {"decision": "block", "reason": f"Continue the approved plan. Next open step: {remaining[0]}"}
    )
    state.update(fingerprint=fingerprint, stalls=stalls)
    save_state(path, state)
    return response


def handle(data: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        return {}
    root = project_root(data)
    event = str(data.get("hook_event_name") or "")
    if event == "PreToolUse":
        return pre_tool(data, root)
    if event == "SubagentStart":
        return context(
            event, "Follow .codex/AGENTS.md and your typed profile. Never spawn workers or declare G6/IIAC PASS."
        )
    session = str(data.get("session_id") or "")
    if not session:
        return {"systemMessage": "Codex session id unavailable; session-owned plan enforcement is inactive."}
    path = state_path(root, session)
    if event == "UserPromptSubmit":
        return plan_prompt(data, root, path)
    if event == "Stop":
        return stop(data, root, path)
    if event == "Interrupt" and path.is_file():
        state = load_state(path)
        state["paused"] = True
        save_state(path, state)
    return {}


def main() -> int:
    event = ""
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict):
            raise ValueError("Hook input must be an object")
        event = str(data.get("hook_event_name") or "")
        response = handle(data)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        message = (
            f"Codex guard could not validate {event or 'event'} ({type(exc).__name__}); "
            "inspect setup before continuing."
        )
        response = deny(message) if event == "PreToolUse" else {"systemMessage": message}
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
