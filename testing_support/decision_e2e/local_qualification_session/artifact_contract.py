# © Artur Czarnecki. All rights reserved.

"""Central canonical artifact set for local qualification sessions."""

from __future__ import annotations

DEFAULT_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "runs.json",
    "summary.json",
    "report.md",
    "analysis.json",
    "artifact-manifest.txt",
    "final-report.md",
)
