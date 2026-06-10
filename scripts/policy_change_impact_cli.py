#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-20.2 — policy change impact visualization CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.runtime.architecture import (  # noqa: E402
    build_capability_impact_report,
    build_catalog_capability_graph,
)
from intergrax.runtime.architecture.policy_change_impact import (  # noqa: E402
    render_policy_change_impact_visualization,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of highest-impact nodes to visualize.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write CapabilityImpactReport JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    graph = build_catalog_capability_graph()
    report = build_capability_impact_report(graph)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(render_policy_change_impact_visualization(report, top_n=args.top))
    print(f"\nnodes analyzed: {len(report.impacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
