# © Artur Czarnecki. All rights reserved.

"""AHI-MAINT-03 — CI fixture signal trends for adaptive evidence."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "build" / "ahi" / "signal_trends.json"


def build_signal_trends_fixture() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "signals": [
            {"name": "routing_latency_p95", "value": 0.42, "window": "ci_fixture"},
            {"name": "retriever_switch_rate", "value": 0.08, "window": "ci_fixture"},
        ],
    }


def export_signal_trends_fixture(path: Path | None = None) -> Path:
    out = path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_signal_trends_fixture()
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out
