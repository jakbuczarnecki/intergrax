#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI guard: context preflight must default to adapter token counting (M-LLM-X.3.4)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_preflight.py"


def main() -> int:
    source = PREFLIGHT.read_text(encoding="utf-8")
    required = (
        "adapter.count_messages_tokens",
        "if count_tokens is None:",
    )
    missing = [token for token in required if token not in source]
    if missing:
        print(f"FAIL: context_preflight.py missing required patterns: {missing}", file=sys.stderr)
        return 1
    print("OK: verify_context_preflight defaults to adapter.count_messages_tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
