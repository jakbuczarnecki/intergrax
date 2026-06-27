#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — §40.12 reference mutating checklist (ACP-CLOSE-PROD-7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path = str(_entry)
    if path not in sys.path:
        sys.path.insert(0, path)

from intergrax.agents.readiness.section_40_12_checklist import (  # noqa: E402
    Section4012ItemStatus,
    build_section_40_12_reference_report,
    write_section_40_12_reference_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check §40.12 reference mutating checklist")
    parser.add_argument("--write", action="store_true", help="Write build/acp_section_40_12_reference.json")
    args = parser.parse_args()

    report = (
        write_section_40_12_reference_report()
        if args.write
        else build_section_40_12_reference_report()
    )

    failures = [
        item
        for item in report.items
        if item.status == Section4012ItemStatus.FAIL
    ]
    if failures:
        print("§40.12 reference checklist failures:")
        for item in failures:
            print(f"  {item.item_id}: {item.requirement}")
        return 1

    if not report.all_passed:
        print("§40.12 reference checklist: unexpected non-pass state")
        return 1

    print(
        f"§40.12 reference checklist: OK ({len(report.items)} items, "
        f"capability={report.reference_capability})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
