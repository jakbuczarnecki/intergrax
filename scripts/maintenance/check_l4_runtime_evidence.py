#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-AHI.1 — L4 closed-loop evidence gate (harness baseline)."""

from __future__ import annotations

import sys

from intergrax.runtime.adaptive.l4_runtime_evidence import build_harness_baseline_l4_evidence


def main() -> int:
    report = build_harness_baseline_l4_evidence()
    if report.scenarios_passed_count < 3:
        print("L4 evidence requires >=3 golden scenarios", file=sys.stderr)
        return 1
    if not report.runtime_l4_closed_loop_passed:
        print("harness baseline L4 evidence must pass closeout thresholds", file=sys.stderr)
        return 1
    if report.apply_rollback_rate >= 0.10:
        print("apply_rollback_rate must stay below 0.10", file=sys.stderr)
        return 1
    print(
        "OK: L4 runtime evidence "
        f"(scenarios={report.scenarios_passed_count}, rollback_rate={report.apply_rollback_rate:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
