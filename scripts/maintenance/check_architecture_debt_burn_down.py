#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-32.1 — living architecture debt burn-down tied to milestones."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from intergrax.runtime.architecture.debt_burn_down import load_debt_burn_down_report  # noqa: E402


def main() -> int:
    report = load_debt_burn_down_report(REPO_ROOT)
    if not report.records:
        print("debt register must include DEBT rows linked to AUDIT-IDEAL", file=sys.stderr)
        return 1
    if report.unresolved_debt_ids:
        print(
            "unresolved debt for Done AUDIT-IDEAL items: "
            + ", ".join(report.unresolved_debt_ids),
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: architecture debt burn-down "
        f"({len(report.records)} debt rows, {len(report.done_audit_ids)} done audit ids)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
