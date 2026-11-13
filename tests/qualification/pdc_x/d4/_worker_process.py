# © Artur Czarnecki. All rights reserved.

"""Self-bounded OS worker used by the PDC-D4 crash/restart qualification.

Run as ``python _worker_process.py <max_seconds>``. The process prints ``ready`` and then
idles. It always exits on its own after ``max_seconds`` so that a failed test can never
leave a long-lived stray process behind.
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    max_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    deadline = time.monotonic() + max_seconds
    print("ready", flush=True)
    while time.monotonic() < deadline:
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
