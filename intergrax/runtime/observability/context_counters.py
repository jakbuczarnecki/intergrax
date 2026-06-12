# © Artur Czarnecki. All rights reserved.

"""Opt-in context assembly metrics (CE-9.4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def context_metrics_enabled() -> bool:
    return os.environ.get("INTERGRAX_CONTEXT_METRICS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@dataclass
class ContextAssemblyCounters:
    assemble_total: int = 0
    candidate_collected_total: int = 0
    candidate_dropped_total: int = 0
    validation_failed_total: int = 0
    fragments_by_engine: dict[str, int] = field(default_factory=dict)

    def record_assemble(self, engine_id: str) -> None:
        if not context_metrics_enabled():
            return
        self.assemble_total += 1
        self.fragments_by_engine[engine_id] = self.fragments_by_engine.get(engine_id, 0) + 1


_COUNTERS = ContextAssemblyCounters()


def get_context_counters() -> ContextAssemblyCounters:
    return _COUNTERS
