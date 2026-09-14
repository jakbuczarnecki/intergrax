# © Artur Czarnecki. All rights reserved.

"""Legacy orchestrator expansion over catalog mandatory sources."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    LEGACY_ORCHESTRATOR_EXPANSION_PATHS,
    orchestrator_mandatory_lookup,
)
from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)


@dataclass(frozen=True, slots=True)
class CatalogRequiredTarget:
    display_label: str
    pytest_arguments: tuple[str, ...]

    @classmethod
    def from_targets(
        cls, display_label: str, targets: list[str]
    ) -> CatalogRequiredTarget:
        return cls(
            display_label=display_label,
            pytest_arguments=normalize_pytest_arguments(targets),
        )


def _orchestrator_expansion(targets: list[str]) -> FrozenPytestSuiteSource | None:
    if len(targets) != 1:
        return None
    normalized = normalize_pytest_arguments(targets)[0]
    if normalized not in LEGACY_ORCHESTRATOR_EXPANSION_PATHS:
        return None
    return orchestrator_mandatory_lookup()[normalized]


def expand_mandatory_subprocesses(
    source: FrozenPytestSuiteSource,
) -> tuple[CatalogRequiredTarget, ...]:
    expanded: list[CatalogRequiredTarget] = []
    for display_label, targets in source:
        expanded.extend(_expand_one_mandatory(display_label, targets))
    return tuple(expanded)


def _expand_one_mandatory(
    display_label: str,
    targets: list[str],
) -> tuple[CatalogRequiredTarget, ...]:
    nested = _orchestrator_expansion(targets)
    if nested is None:
        return (CatalogRequiredTarget.from_targets(display_label, targets),)
    expanded: list[CatalogRequiredTarget] = []
    for child_label, child_targets in nested:
        child_path = f"{display_label}>{child_label}"
        expanded.extend(_expand_one_mandatory(child_path, child_targets))
    return tuple(expanded)


def unique_required_leaf_targets(
    source: FrozenPytestSuiteSource,
) -> tuple[CatalogRequiredTarget, ...]:
    subprocesses = expand_mandatory_subprocesses(source)
    seen: set[tuple[str, ...]] = set()
    unique: list[CatalogRequiredTarget] = []
    for entry in subprocesses:
        if entry.pytest_arguments in seen:
            continue
        seen.add(entry.pytest_arguments)
        unique.append(entry)
    return tuple(unique)


def is_nested_orchestrator_leaf(pytest_arguments: tuple[str, ...]) -> bool:
    return (
        len(pytest_arguments) == 1
        and pytest_arguments[0] in LEGACY_ORCHESTRATOR_EXPANSION_PATHS
    )
