# © Artur Czarnecki. All rights reserved.

"""OBS-DELIVERY-QOS-SCALE-R1 — static gates for atomic non-critical admission."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SINK_PATH = (
    REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "event_delivery"
    / "bounded_event_sink.py"
)


def main() -> int:
    text = SINK_PATH.read_text(encoding="utf-8")
    violations: list[str] = []

    if text.count("_non_critical_buffered") < 3:
        violations.append("missing unified non-critical buffered coordination state")

    if "_admit_non_critical(" not in text:
        violations.append("BEST_EFFORT and IMPORTANT must share _admit_non_critical")

    if "_non_critical_at_capacity" in text:
        violations.append(
            "IMPORTANT must not use separate pre-check _non_critical_at_capacity"
        )

    if "threading.Condition" not in text:
        violations.append("non-critical quota wait must use threading.Condition")

    if "_buffered_count_lock" in text:
        violations.append("legacy split lock must not remain alongside quota authority")

    if violations:
        for item in violations:
            print(item, file=sys.stderr)
        return 1

    print("OK: OBS-DELIVERY-QOS-SCALE-R1 atomic non-critical admission gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
