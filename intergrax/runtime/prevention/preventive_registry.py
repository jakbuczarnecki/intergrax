# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive analyzer registry — deterministic ordering (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.preventive.preventive_analyzer import PreventiveAnalyzer


class PreventiveRegistryConfigurationError(Exception):
    """Invalid analyzer registration — fail fast at bootstrap."""


@dataclass(frozen=True, slots=True)
class _OrderedPreventiveAnalyzer:
    priority: int
    namespace: str
    stable_id: str
    analyzer: PreventiveAnalyzer


def _sort_key(item: _OrderedPreventiveAnalyzer) -> tuple[int, str, str]:
    return (-item.priority, item.namespace, item.stable_id)


class PreventiveAnalyzerRegistry:
    def __init__(self, analyzers: tuple[PreventiveAnalyzer, ...] = ()) -> None:
        self._analyzers = _order(analyzers)

    @property
    def analyzers(self) -> tuple[PreventiveAnalyzer, ...]:
        return self._analyzers

    @classmethod
    def empty(cls) -> PreventiveAnalyzerRegistry:
        return cls()


def _order(analyzers: tuple[PreventiveAnalyzer, ...]) -> tuple[PreventiveAnalyzer, ...]:
    seen: set[tuple[str, str]] = set()
    ordered: list[_OrderedPreventiveAnalyzer] = []
    for analyzer in analyzers:
        key = (analyzer.analyzer_namespace, analyzer.analyzer_id)
        if key in seen:
            raise PreventiveRegistryConfigurationError(
                f"duplicate preventive analyzer: {analyzer.analyzer_namespace}/{analyzer.analyzer_id}",
            )
        descriptor = analyzer.descriptor
        if descriptor.analyzer_id != analyzer.analyzer_id:
            raise PreventiveRegistryConfigurationError("descriptor.analyzer_id mismatch")
        if descriptor.namespace != analyzer.analyzer_namespace:
            raise PreventiveRegistryConfigurationError("descriptor.namespace mismatch")
        if descriptor.version != analyzer.analyzer_version:
            raise PreventiveRegistryConfigurationError("descriptor.version mismatch")
        seen.add(key)
        ordered.append(
            _OrderedPreventiveAnalyzer(
                priority=analyzer.priority,
                namespace=analyzer.analyzer_namespace,
                stable_id=analyzer.analyzer_id,
                analyzer=analyzer,
            ),
        )
    return tuple(item.analyzer for item in sorted(ordered, key=_sort_key))


__all__ = ["PreventiveAnalyzerRegistry", "PreventiveRegistryConfigurationError"]
