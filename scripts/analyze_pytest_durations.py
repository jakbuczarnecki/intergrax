#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Summarize pytest --durations=0 output (slowest tests + per-file totals)."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(
    r"^(?P<sec>\d+(?:\.\d+)?)s\s+(?P<phase>\w+)\s+(?P<nodeid>.+)$"
)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/ci-gate-durations.txt")
    if not path.is_file():
        print(f"missing duration log: {path}", file=sys.stderr)
        return 1
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")
    rows: list[tuple[float, str, str]] = []
    in_section = False
    for line in text.splitlines():
        if line.strip().startswith("=== slowest"):
            in_section = True
            continue
        if in_section and line.startswith("="):
            break
        if not in_section:
            continue
        m = LINE.match(line.strip())
        if not m:
            continue
        sec = float(m.group("sec"))
        nodeid = m.group("nodeid")
        rows.append((sec, m.group("phase"), nodeid))

    if not rows:
        print("no duration rows found (run pytest with --durations=0 first)")
        return 1

    total = sum(sec for sec, _, _ in rows)
    by_file: dict[str, float] = defaultdict(float)
    by_test: dict[str, float] = defaultdict(float)
    for sec, _, nodeid in rows:
        file_part = nodeid.split("::", 1)[0]
        by_file[file_part] += sec
        by_test[nodeid] += sec

    print(f"tests timed: {len(by_test)}")
    print(f"sum(call/setup/teardown durations): {total:.1f}s")
    print()
    print("TOP 25 slowest tests:")
    for nodeid, sec in sorted(by_test.items(), key=lambda x: x[1], reverse=True)[:25]:
        print(f"  {sec:7.2f}s  {nodeid}")
    print()
    print("TOP 20 slowest files (sum of test durations):")
    for file_part, sec in sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {sec:7.1f}s  {file_part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
