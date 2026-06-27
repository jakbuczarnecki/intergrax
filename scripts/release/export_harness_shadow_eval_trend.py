#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Export harness shadow evaluation trend report from the file registry (W-OPS.11)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True, help="Release label for this export batch")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "build" / "architecture_hardening" / "shadow_evaluation_trend_report.json",
    )
    parser.add_argument(
        "--keep-registry",
        action="store_true",
        help="Do not clear observation registry after export",
    )
    args = parser.parse_args()

    from intergrax.runtime.architecture.online_evaluation_trend import export_shadow_evaluation_trend

    report = export_shadow_evaluation_trend(
        args.release_id.strip(),
        clear_registry_after_export=not args.keep_registry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"shadow evaluation trend: {len(report.snapshots)} snapshot(s), {len(report.comparisons)} comparison(s)")
    print(f"artifact: {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
