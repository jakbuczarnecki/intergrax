# © Artur Czarnecki. All rights reserved.

"""Project frozen ``(label, pytest targets)`` declarations onto R1 qualification contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from testing_support.execution_qualification.contracts import (
    QualificationRunManifest,
    QualificationSuite,
)

FrozenPytestSuiteSource = tuple[tuple[str, list[str]], ...]


@dataclass(frozen=True, slots=True)
class AdaptedMandatorySuite:
    """One frozen mandatory label with its typed subprocess definition."""

    display_label: str
    suite: QualificationSuite


def adapt_frozen_pytest_suites(
    source: FrozenPytestSuiteSource,
    *,
    label_to_suite_id: Mapping[str, str],
    exclusive_resource_by_label: Mapping[str, str] | None = None,
) -> tuple[AdaptedMandatorySuite, ...]:
    """Build manifest entries 1:1 with ``source`` order; labels must map to unique suite IDs."""
    if not source:
        raise ValueError("source must contain at least one mandatory suite")
    exclusive_map = exclusive_resource_by_label if exclusive_resource_by_label is not None else {}
    adapted: list[AdaptedMandatorySuite] = []
    seen_suite_ids: set[str] = set()
    for display_label, pytest_targets in source:
        if display_label not in label_to_suite_id:
            raise ValueError(f"no suite_id mapping for mandatory label: {display_label!r}")
        suite_id = label_to_suite_id[display_label]
        if suite_id in seen_suite_ids:
            raise ValueError(f"duplicate suite_id mapping for label: {display_label!r}")
        seen_suite_ids.add(suite_id)
        exclusive = exclusive_map.get(display_label)
        suite = QualificationSuite(
            suite_id=suite_id,
            pytest_arguments=tuple(pytest_targets),
            exclusive_resource_id=exclusive,
        )
        adapted.append(AdaptedMandatorySuite(display_label=display_label, suite=suite))
    return tuple(adapted)


def manifest_from_adapted_suites(
    adapted: tuple[AdaptedMandatorySuite, ...],
) -> QualificationRunManifest:
    return QualificationRunManifest(suites=tuple(entry.suite for entry in adapted))
