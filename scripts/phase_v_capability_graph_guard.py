#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Phase V capability graph lineage, impact, and compatibility guard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Protocol

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.runtime.architecture import (
    CapabilityGraph,
    build_capability_impact_report,
    build_capability_lineage_report,
    build_catalog_capability_graph,
    evaluate_capability_graph_compatibility,
)


class ReportWriter(Protocol):
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        ...


class JsonReportWriter:
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _load_previous_graph(path: Path) -> CapabilityGraph | None:
    if not path.exists():
        return None
    return CapabilityGraph.model_validate_json(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Return non-zero when compatibility report has errors.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = REPO_ROOT / "build" / "architecture_hardening"
    baseline_path = output_dir / "capability_graph_baseline.json"
    writer: ReportWriter = JsonReportWriter()

    current_graph = build_catalog_capability_graph()
    previous_graph = _load_previous_graph(baseline_path)
    if previous_graph is None:
        previous_graph = current_graph

    lineage_report = build_capability_lineage_report(current_graph)
    impact_report = build_capability_impact_report(current_graph)
    compatibility_report = evaluate_capability_graph_compatibility(
        previous=previous_graph,
        current=current_graph,
    )

    writer.write(output_path=output_dir / "capability_graph.json", payload=current_graph)
    writer.write(output_path=output_dir / "capability_lineage_report.json", payload=lineage_report)
    writer.write(output_path=output_dir / "capability_impact_report.json", payload=impact_report)
    writer.write(output_path=output_dir / "capability_compatibility_report.json", payload=compatibility_report)
    writer.write(output_path=baseline_path, payload=current_graph)

    print("phase-v capability graph guard: OK")
    print(f"compatible: {compatibility_report.compatible}")
    print(f"artifacts: {output_dir.as_posix()}")

    if args.enforce and not compatibility_report.compatible:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
