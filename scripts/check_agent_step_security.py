#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — Tier-2 agent step security: gateway-only I/O surface (ACP-CON-7)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = REPO_ROOT / "agents"

FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "intergrax.integrations.providers",
    "boto3",
    "httpx",
    "requests",
    "aiohttp",
    "openai",
    "anthropic",
    "slack_sdk",
    "google.cloud",
    "azure.",
)

AGENT_ENTRY_CALL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bRuntimeEngine\s*\("), "RuntimeEngine()"),
    (re.compile(r"\brun_pipeline_step\s*\("), "run_pipeline_step()"),
    (re.compile(r"\bsocket\.(socket|create_connection)\b"), "raw socket"),
    (re.compile(r"\bsubprocess\.(run|Popen|call)\b"), "subprocess"),
)

ALLOWLIST_RELATIVE: frozenset[str] = frozenset(
    {
        "agents/lab/mock_agents.py",
    }
)


def main() -> int:
    violations: list[str] = []

    for path in sorted(AGENTS_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWLIST_RELATIVE:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("import ", "from ")):
                for prefix in FORBIDDEN_IMPORT_PREFIXES:
                    if prefix in stripped:
                        violations.append(f"{rel}:{line_no}: forbidden import ({prefix})")
            if not path.name.endswith("_agent.py"):
                continue
            for pattern, label in AGENT_ENTRY_CALL_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{rel}:{line_no}: forbidden {label}")

    if violations:
        print("Agent step security violations:")
        print("\n".join(violations))
        return 1

    print("Agent step security gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
