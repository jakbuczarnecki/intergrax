# © Artur Czarnecki. All rights reserved.

"""CE-01-R1: canonical ContextEngine must not read semantic runtime handles."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_engine.py"

SEMANTIC_LITERALS = (
    "runtime_config",
    "messages",
    "max_output_tokens",
    "context_optimization_policy",
    "nexus_ucl_runtime",
)

HANDLE_GET_PATTERN = re.compile(
    r"ctx\.handles\.get\(\s*[\"']([^\"']+)[\"']"
)


def main() -> int:
    text = ENGINE_PATH.read_text(encoding="utf-8")
    violations: list[str] = []
    for match in HANDLE_GET_PATTERN.finditer(text):
        key = match.group(1)
        if key in SEMANTIC_LITERALS or key in {"event_bus", "node_id", "agent_id"}:
            violations.append(f"context_engine.py: ctx.handles.get({key!r})")
    if violations:
        print("CE canonical semantic handle reads detected:", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
