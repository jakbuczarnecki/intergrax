# © Artur Czarnecki. All rights reserved.
"""CI gate: architecture hubs must stay token-efficient (F4)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "architecture"

MAX_HUB_LINES = 1200
MAX_HUB_TOKENS = 12_000
SPLIT_MARKER = "## Architecture satellites (read on demand)"


def main() -> int:
    errors: list[str] = []

    for path in sorted(ARCH.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if SPLIT_MARKER not in text:
            continue
        lines = text.splitlines()
        tokens = len(text) // 4
        if len(lines) > MAX_HUB_LINES:
            errors.append(
                f"{path.relative_to(ROOT)}: {len(lines)} lines exceeds arch hub max {MAX_HUB_LINES}"
            )
        if tokens > MAX_HUB_TOKENS:
            errors.append(
                f"{path.relative_to(ROOT)}: ~{tokens} tokens exceeds arch hub max {MAX_HUB_TOKENS}"
            )

    if errors:
        print("check_arch_hub_size: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_arch_hub_size: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
