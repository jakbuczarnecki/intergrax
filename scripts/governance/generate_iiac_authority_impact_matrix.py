#!/usr/bin/env python3
"""Regenerate IIAC G0 global authority impact matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.governance.iiac_g0_inventory import (
    DEFAULT_MATRIX,
    REPO,
    build_matrix_payload,
)

OUTPUT = DEFAULT_MATRIX


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate IIAC G0 authority impact matrix")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="JSON output path",
    )
    parser.add_argument(
        "--inventory-source-ref",
        default="HEAD",
        help="Git ref for governed-file snapshot (default: HEAD)",
    )
    parser.add_argument(
        "--inventory-publication-sha",
        default=None,
        help="Optional publication SHA (omit in same commit as matrix write to avoid self-ref)",
    )
    args = parser.parse_args()

    try:
        payload = build_matrix_payload(
            args.inventory_source_ref,
            inventory_publication_sha=args.inventory_publication_sha,
        )
    except RuntimeError as exc:
        print(f"generate_iiac_authority_impact_matrix: FAIL — {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rel = args.output.relative_to(REPO)
    print(
        f"IIAC authority impact matrix: {payload['entry_count']} entries, "
        f"{payload['covered_path_count']} covered paths @ {payload['inventory_source_sha']} → {rel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
