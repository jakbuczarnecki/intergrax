# © Artur Czarnecki. All rights reserved.
"""Cursor ``subagentStart`` guard: deny the untyped ``generalPurpose`` subagent (stdlib only).

Claude Code denies ``general-purpose`` through ``permissions.deny``; Cursor does not enforce that
file, so ``.cursor/hooks.json`` registers this script for ``subagentStart`` with the matcher
``^generalPurpose$`` and ``failClosed: true``. With ``failClosed`` an empty stdout, a crash or a
non-zero exit would block the spawn, therefore this script always prints one valid JSON object.

* ``IGX_HOOKS=off`` -> ``{"permission": "allow"}``;
* ``subagent_type`` exactly ``generalPurpose`` (or missing / empty) -> ``deny`` naming the typed roles;
* any other non-empty type (``explore``, ``igx-*``, ``general-purpose``, ``GeneralPurpose`` ...) -> ``allow``.

Matching is exact on purpose: another spelling is added to ``DENIED_TYPES`` only after a probe shows
that exact string in the hook log.
"""

from __future__ import annotations

import json
import os
import sys

DENIED_TYPES = frozenset({"generalPurpose", "general-purpose", "GeneralPurpose"})
ROLES = ("igx-architect", "igx-implementer", "igx-researcher", "igx-reviewer")


def decide(data: object) -> dict[str, str]:
    subagent_type = ""
    if isinstance(data, dict):
        raw = data.get("subagent_type")
        subagent_type = raw if isinstance(raw, str) else ""
    if subagent_type and subagent_type not in DENIED_TYPES:
        return {"permission": "allow"}
    return {
        "permission": "deny",
        "user_message": (
            "generalPurpose subagents are not allowed in this repository. "
            f"Use a typed role instead: {', '.join(ROLES)} (or explore for read-only search)."
        ),
    }


def main() -> int:
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        sys.stdout.write(json.dumps({"permission": "allow"}))
        return 0
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except (ValueError, OSError):
        data = {}
    sys.stdout.write(json.dumps(decide(data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
