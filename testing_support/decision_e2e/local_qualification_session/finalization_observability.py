# © Artur Czarnecki. All rights reserved.

"""Qualification finalization observability counters (non-decision metrics)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FinalizationObservability:
    artifact_generation_started: int = 0
    artifact_generation_completed: int = 0
    artifact_validation_failed: int = 0
    finalization_blocked: int = 0
    events: list[str] = field(default_factory=list)

    def record(self, name: str) -> None:
        self.events.append(name)
        if name == "artifact_generation_started":
            self.artifact_generation_started += 1
        elif name == "artifact_generation_completed":
            self.artifact_generation_completed += 1
        elif name == "artifact_validation_failed":
            self.artifact_validation_failed += 1
        elif name == "finalization_blocked":
            self.finalization_blocked += 1


_GLOBAL = FinalizationObservability()


def reset_finalization_observability() -> None:
    global _GLOBAL
    _GLOBAL = FinalizationObservability()


def get_finalization_observability() -> FinalizationObservability:
    return _GLOBAL


def record_finalization_metric(name: str) -> None:
    get_finalization_observability().record(name)
