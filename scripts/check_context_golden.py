#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-16.1 — context golden fixture gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from intergrax.runtime.context.context_golden_harness import load_context_golden_cases

    cases = load_context_golden_cases(REPO_ROOT)
    if not cases:
        print("no context golden cases", file=sys.stderr)
        return 1
    print(f"OK: {len(cases)} context golden cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
