#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-32.2 — architecture debt register presence gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    register = REPO_ROOT / "docs" / "guides" / "ARCHITECTURE_DEBT_REGISTER.md"
    if not register.is_file():
        print(f"missing register: {register}", file=sys.stderr)
        return 1
    text = register.read_text(encoding="utf-8")
    if "DEBT-" not in text:
        print("register missing DEBT- entries", file=sys.stderr)
        return 1
    print("OK: architecture debt register present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
