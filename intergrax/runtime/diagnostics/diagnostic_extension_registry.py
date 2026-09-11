# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Central extension registry — deterministic discovery and validation (R5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from intergrax.contracts.diagnostic_analyzer import DiagnosticAnalyzer
from intergrax.contracts.diagnostic_evidence_contributor import (
    DiagnosticEvidenceContributor,
)
from intergrax.contracts.diagnostic_taxonomy_contributor import (
    DiagnosticTaxonomyContributor,
    validate_taxonomy_kind_token,
)

_T = TypeVar("_T")


class DiagnosticExtensionConfigurationError(Exception):
    """Invalid extension registration — fail fast at bootstrap."""


@dataclass(frozen=True, slots=True)
class _OrderedExtension:
    priority: int
    namespace: str
    stable_id: str
    extension: object


def _extension_sort_key(item: _OrderedExtension) -> tuple[int, str, str]:
    return (item.priority, item.namespace, item.stable_id)


class DiagnosticExtensionRegistry:
    """Discovery, validation, deterministic ordering, and isolation metadata."""

    def __init__(
        self,
        *,
        evidence_contributors: tuple[DiagnosticEvidenceContributor, ...] = (),
        analyzers: tuple[DiagnosticAnalyzer, ...] = (),
        taxonomy_contributors: tuple[DiagnosticTaxonomyContributor, ...] = (),
    ) -> None:
        self._evidence_contributors = _order_contributors(evidence_contributors)
        self._analyzers = _order_analyzers(analyzers)
        self._taxonomy = _order_taxonomy(taxonomy_contributors)
        self._taxonomy_catalog = _build_taxonomy_catalog(self._taxonomy)

    @property
    def evidence_contributors(self) -> tuple[DiagnosticEvidenceContributor, ...]:
        return self._evidence_contributors

    @property
    def analyzers(self) -> tuple[DiagnosticAnalyzer, ...]:
        return self._analyzers

    @property
    def taxonomy_catalog(self) -> tuple[str, ...]:
        return self._taxonomy_catalog

    @classmethod
    def empty(cls) -> DiagnosticExtensionRegistry:
        return cls()


def _order_contributors(
    contributors: tuple[DiagnosticEvidenceContributor, ...],
) -> tuple[DiagnosticEvidenceContributor, ...]:
    ordered = _dedupe_and_sort(
        [
            _OrderedExtension(
                priority=contributor.priority,
                namespace=contributor.evidence_namespace,
                stable_id=contributor.contributor_id,
                extension=contributor,
            )
            for contributor in contributors
        ],
        label="evidence contributor",
    )
    return tuple(item.extension for item in ordered)


def _order_analyzers(
    analyzers: tuple[DiagnosticAnalyzer, ...],
) -> tuple[DiagnosticAnalyzer, ...]:
    ordered = _dedupe_and_sort(
        [
            _OrderedExtension(
                priority=analyzer.priority,
                namespace=analyzer.analyzer_namespace,
                stable_id=analyzer.analyzer_id,
                extension=analyzer,
            )
            for analyzer in analyzers
        ],
        label="analyzer",
    )
    return tuple(item.extension for item in ordered)


def _order_taxonomy(
    contributors: tuple[DiagnosticTaxonomyContributor, ...],
) -> tuple[DiagnosticTaxonomyContributor, ...]:
    ordered = _dedupe_and_sort(
        [
            _OrderedExtension(
                priority=contributor.priority,
                namespace=contributor.taxonomy_namespace,
                stable_id=contributor.contributor_id,
                extension=contributor,
            )
            for contributor in contributors
        ],
        label="taxonomy contributor",
    )
    return tuple(item.extension for item in ordered)


def _dedupe_and_sort(
    items: list[_OrderedExtension],
    *,
    label: str,
) -> list[_OrderedExtension]:
    seen: set[str] = set()
    for item in items:
        if item.stable_id in seen:
            raise DiagnosticExtensionConfigurationError(
                f"duplicate {label} stable_id: {item.stable_id!r}",
            )
        seen.add(item.stable_id)
    return sorted(items, key=_extension_sort_key)


def _build_taxonomy_catalog(
    contributors: tuple[DiagnosticTaxonomyContributor, ...],
) -> tuple[str, ...]:
    kinds: list[str] = []
    seen: set[str] = set()
    for contributor in contributors:
        for kind in contributor.registered_kinds():
            normalized = validate_taxonomy_kind_token(
                kind,
                expected_namespace=contributor.taxonomy_namespace,
            )
            if normalized in seen:
                raise DiagnosticExtensionConfigurationError(
                    f"duplicate taxonomy kind: {normalized!r}",
                )
            seen.add(normalized)
            kinds.append(normalized)
    return tuple(kinds)


__all__ = [
    "DiagnosticExtensionConfigurationError",
    "DiagnosticExtensionRegistry",
]
