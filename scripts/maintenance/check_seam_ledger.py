#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail CI when product imports cross boundary seams not declared in governance/architecture/seam_ledger.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

from scripts.maintenance.seam_ledger_lib import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    error_findings,
    run_seam_ledger_check,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mechanical seam ledger gate (PT-G01).")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=_REPO / DEFAULT_LEDGER_PATH,
        help="Path to seam_ledger.yaml",
    )
    args = parser.parse_args()
    findings = run_seam_ledger_check(repo_root=_REPO, ledger_path=args.ledger)
    errors = error_findings(findings)
    if errors:
        print("seam_ledger: FAIL")
        for item in errors:
            loc = ""
            if item.path:
                loc = item.path
                if item.line is not None:
                    loc += f":{item.line}"
                loc += ": "
            prefix = f"[{item.kind}] "
            if item.rule_id:
                prefix += f"({item.rule_id}) "
            print(f"  - {prefix}{loc}{item.message}")
        return 1
    print("check_seam_ledger: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
