# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive analyzer registry — ordering, isolation metadata (PREDICTIVE R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from intergrax.contracts.predictive_analyzer import PredictiveAnalyzer

_T = TypeVar("_T")


class PredictiveRegistryConfigurationError(Exception):
    """Invalid analyzer registration — fail fast at bootstrap."""


@dataclass(frozen=True, slots=True)
class _OrderedAnalyzer:
    priority: int
    namespace: str
    stable_id: str
    analyzer: PredictiveAnalyzer


def _analyzer_sort_key(item: _OrderedAnalyzer) -> tuple[int, str, str]:
    return (-item.priority, item.namespace, item.stable_id)


class PredictiveAnalyzerRegistry:
    """Discovery, validation, deterministic ordering, and isolation metadata."""

    def __init__(self, analyzers: tuple[PredictiveAnalyzer, ...] = ()) -> None:
        self._analyzers = _order_analyzers(analyzers)

    @property
    def analyzers(self) -> tuple[PredictiveAnalyzer, ...]:
        return self._analyzers

    @classmethod
    def empty(cls) -> PredictiveAnalyzerRegistry:
        return cls()


def _order_analyzers(
    analyzers: tuple[PredictiveAnalyzer, ...],
) -> tuple[PredictiveAnalyzer, ...]:
    ordered = _dedupe_and_sort(
        [
            _OrderedAnalyzer(
                priority=analyzer.priority,
                namespace=analyzer.analyzer_namespace,
                stable_id=analyzer.analyzer_id,
                analyzer=analyzer,
            )
            for analyzer in analyzers
        ],
        label="predictive analyzer",
    )
    return tuple(item.analyzer for item in ordered)


def _dedupe_and_sort(
    items: list[_OrderedAnalyzer],
    *,
    label: str,
) -> list[_OrderedAnalyzer]:
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.namespace, item.stable_id)
        if key in seen:
            raise PredictiveRegistryConfigurationError(
                f"duplicate {label}: {item.namespace}/{item.stable_id}",
            )
        seen.add(key)
    return sorted(items, key=_analyzer_sort_key)


__all__ = [
    "PredictiveAnalyzerRegistry",
    "PredictiveRegistryConfigurationError",
]
