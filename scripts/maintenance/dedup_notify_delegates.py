#!/usr/bin/env python3
"""Remove duplicate sync notify delegates when async notify exists."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "intergrax" / "integrations" / "providers" / "notification_channel"
SYNC_NOTIFY = re.compile(
    r"\n    def notify\(self, message\):\n        return self\._require_client\(\)\.notify\(message\)\n",
)


def main() -> None:
    fixed = 0
    for path in ROOT.glob("*/integration.py"):
        src = path.read_text(encoding="utf-8")
        if "async def notify" not in src:
            continue
        new, count = SYNC_NOTIFY.subn("\n", src)
        if count:
            path.write_text(new, encoding="utf-8")
            fixed += 1
    print(f"fixed: {fixed}")


if __name__ == "__main__":
    main()
